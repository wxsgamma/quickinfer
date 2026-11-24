from copy import copy
from datetime import timedelta
import traceback

import torch
import torch.distributed as dist

from quickinfer.llmengine.model_runner import ModelRunner


def written_ranges(seqs, block_size):
    ranges = []
    for seq in seqs:
        pos = seq.num_cached_tokens
        end = pos + seq.num_scheduled_tokens
        while pos < end:
            block, offset = divmod(pos, block_size)
            length = min(block_size - offset, end - pos)
            slot = seq.block_table[block] * block_size + offset
            ranges.append((slot, slot + length))
            pos += length
    return ranges


def block_ranges(blocks, size):
    return [(b * size, (b + 1) * size) for b in blocks]


class KVTransfer:
    def __init__(self, url, rank, device):
        self.device = torch.device("cuda", device)
        self.peer = 1 - rank
        self.stream = torch.cuda.Stream(device=self.device)
        self.layers = []
        # D 是 rank 0，P 是 rank 1。模型继续用原来的单卡 TP 组，另一组只传 KV。
        timeout = timedelta(seconds=120)
        store, _, _ = next(dist.rendezvous(url, rank, 2, timeout=timeout))
        store.set_timeout(timeout)
        self.group = dist.ProcessGroupNCCL(store, rank, 2, timeout)
        try:
            # 两个方向各传一个字节，先把 NCCL 的 P2P 通道和缓冲区建好，再给模型分 KV。
            with torch.cuda.stream(self.stream):
                probe = torch.zeros(2, dtype=torch.uint8, device=self.device)
                self.group._start_coalescing()
                self.group.send([probe[:1]], self.peer, 0)
                self.group.recv([probe[1:]], self.peer, 0)
                self.group._end_coalescing().wait()
            self.stream.synchronize()
        except BaseException:
            self.group.abort()
            raise

    def set_caches(self, caches):
        # 主模型、草稿模型按相同顺序展开，每个视图是一层 K 或 V 的连续槽位。
        self.layers = [layer for cache in caches for layer in cache.flatten(2, 3).flatten(0, 1).unbind()]

    def transfer(self, ranges, *, send):
        if not ranges:
            return
        spans = []
        for start, end in sorted(ranges):
            if spans and start <= spans[-1][1]:
                spans[-1] = (spans[-1][0], max(end, spans[-1][1]))
            else:
                spans.append((start, end))

        self.transfer_spans(spans, send=send)

    def transfer_spans(self, spans, *, send):
        # 两端块号可以不同，这里的顺序和每段长度由发送方和接收方共同确定。
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        op = self.group.send if send else self.group.recv
        with torch.cuda.stream(self.stream):
            for start, end in spans:
                # 两端按同一顺序收发，直接读写各自的 KV 槽，一次提交这个区间的所有层。
                self.group._start_coalescing()
                for layer in self.layers:
                    op([layer[start:end]], self.peer, 0)
                self.group._end_coalescing().wait()

    def close(self, abort=False):
        if abort:
            self.group.abort()
        else:
            self.group.shutdown()
        self.layers.clear()


def run_prefill(config, connection, link_url):
    runner = link = None
    clean_exit = False
    try:
        link = KVTransfer(link_url, 1, config.prefill_gpu)
        runner = ModelRunner(config, 0, [], device=config.prefill_gpu, prefill_only=True)
        link.set_caches(runner.kv_caches)
        connection.send((True, (runner.kv_cache.device.index, config.num_kvcache_blocks)))
        while (batch := connection.recv()) is not None:
            seqs, restore, seed = batch
            torch.cuda.manual_seed(seed)
            size = config.kvcache_block_size
            if restore:
                link.transfer(block_ranges(restore, size), send=False)
            # 等回传的前缀可读，也等上一次发送结束，才能复用 P 上的槽位。
            torch.cuda.current_stream(link.device).wait_stream(link.stream)
            token_ids = runner.run(seqs, True)
            # 先通知 D 准备接收，再发 KV。D 等自己的通信流完成后才使用这些块。
            connection.send((True, token_ids))
            link.transfer(written_ranges(seqs, size), send=True)
        link.stream.synchronize()
        clean_exit = True
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()
        if link is not None:
            link.close(abort=not clean_exit)
        if runner is not None:
            runner.exit()
        elif dist.is_initialized():
            dist.destroy_process_group()


class PrefillWorker:
    def __init__(self, config, ctx):
        prefill_config = copy(config)
        prefill_config.distributed_init_method += "-prefill"
        link_url = config.distributed_init_method + "-kv"
        self.connection, child = ctx.Pipe()
        self.process = ctx.Process(target=run_prefill, args=(prefill_config, child, link_url))
        self.link = None
        self.block_size = config.kvcache_block_size
        self.pending = []
        self.restore = set()
        self.token_ids = None
        try:
            self.process.start()
            child.close()
            self.link = KVTransfer(link_url, 0, config.decode_gpu)
            self.device, self.num_blocks = self.receive()
            if self.device != config.prefill_gpu:
                raise RuntimeError("Prefill worker initialized on the wrong CUDA device")
        except BaseException:
            child.close()
            self.close(abort=True)
            raise

    def receive(self):
        try:
            ok, value = self.connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError("Prefill worker disconnected") from error
        if not ok:
            raise RuntimeError(f"Prefill worker failed:\n{value}")
        return value

    def receive_kv(self):
        if self.token_ids is None:
            self.token_ids = self.receive()
            self.link.transfer(written_ranges(self.pending, self.block_size), send=False)

    def schedule(self, scheduler):
        while True:
            if self.pending and self.token_ids is None and self.connection.poll():
                self.receive_kv()
            if self.token_ids is not None and self.link.stream.query():
                break
            # 先给现有 D 请求留好本轮的槽，避免刚来的 prompt 抢走最后一个空块。
            decode_seqs = scheduler.schedule_decode()
            if not self.pending:
                self.restore = set()
                seqs = scheduler.schedule_prefill(self.restore)
                if seqs:
                    # 每个 P 批次从调用方 RNG 取一个种子，P、D 不复用同一串采样噪声。
                    seed = torch.randint(2**63 - 1, (), device="cpu").item()
                    try:
                        self.connection.send((seqs, self.restore, seed))
                    except (EOFError, OSError) as error:
                        raise RuntimeError("Prefill worker disconnected") from error
                    self.pending = seqs
                    self.link.transfer(block_ranges(self.restore, self.block_size), send=True)
            # P 计算或 NCCL 传输时，D 可以继续跑已有请求。
            if decode_seqs:
                return decode_seqs, False, None
            if self.pending:
                self.receive_kv()
                self.link.stream.synchronize()
                break
            if not scheduler.waiting:
                raise RuntimeError("No requests to schedule")

        seqs, token_ids = self.pending, self.token_ids
        self.pending, self.token_ids = [], None
        for block_id in self.restore:
            scheduler.block_manager.blocks[block_id].prefill_kv_ready = True
        scheduler.admit_to_decode(seqs)
        return seqs, True, token_ids

    def close(self, abort=False):
        if self.connection.closed:
            return
        try:
            if not abort:
                if self.pending:
                    # P 可能正在等匹配的 recv，正常退出时先收完已发出的批次。
                    self.receive_kv()
                    self.link.stream.synchronize()
                self.connection.send(None)
        except BaseException:
            abort = True
            raise
        finally:
            self.connection.close()
            try:
                if self.link is not None:
                    self.link.close(abort)
                    self.link = None
            finally:
                if self.process.pid is not None:
                    if abort and self.process.is_alive():
                        self.process.terminate()
                    self.process.join()
                self.pending, self.token_ids = [], None

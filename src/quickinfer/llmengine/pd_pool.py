from collections import deque
from copy import copy
from multiprocessing import get_context
from multiprocessing.connection import wait
import hashlib
import os
import time
import traceback

from quickinfer.llmengine.modelconfig import ModelConfig
from quickinfer.llmengine.scheduler import Scheduler
from quickinfer.llmengine.sequence import Sequence
from quickinfer.sampling_params import SamplingParams


def handoff_spans(src, dst, tokens, size):
    if tokens <= 0 or size <= 0 or len(src) != (tokens + size - 1) // size or len(src) != len(dst):
        raise ValueError("Invalid KV handoff size")
    spans = []
    for a, b in zip(src, dst, strict=True):
        n = min(tokens, size)
        a, b = a * size, b * size
        if spans and spans[-1][0] + spans[-1][2] == a and spans[-1][1] + spans[-1][2] == b:
            spans[-1][2] += n
        else:
            spans.append([a, b, n])
        tokens -= n
    return spans


def check_argmax(logits, expected, ties):
    chosen = logits.argmax(dim=-1)
    for row, token in enumerate(expected):
        if token is None:
            continue
        got = chosen[row].item()
        if got != token:
            best, reference = logits[row, got].item(), logits[row, token].item()
            if reference != best:
                raise ValueError(f"Reference token {token} scores {reference}, below token {got} at {best}")
            # BF16 可能并列第一。只在分数完全相等时续用单卡 token，以便下个位置比较同一个前缀。
            ties.append((got, token, best))
            chosen[row] = token
    return chosen


def reference_rows(seqs, references, is_prefill):
    rows = []
    for seq, reference in zip(seqs, references, strict=True):
        drafts = [] if is_prefill else seq.draft_tokens
        for pos in range(1 + len(drafts)):
            token = reference[pos] if pos < len(reference) else None
            rows.append(token)
            if pos < len(drafts) and drafts[pos] != token:
                reference = ()  # 草稿从这里分叉，后面的 logits 已经属于另一个前缀。
    return rows


def gpu_worker(conn, config, gpu, prefill, decode, cpus):
    runner, links, pending = None, {}, None
    clean = False
    try:
        if cpus:
            os.sched_setaffinity(0, cpus)
        import torch
        from torch._dynamo.utils import counters
        from quickinfer.llmengine.model_runner import ModelRunner
        from quickinfer.llmengine.prefill_worker import KVTransfer

        torch.cuda.set_device(gpu)
        url = config.distributed_init_method
        for p in prefill:
            for d in decode:
                if gpu in (p, d):
                    peer = d if gpu == p else p
                    links[peer] = KVTransfer(f"{url}-kv-{p}-{d}", int(gpu == p), gpu)
        config.distributed_init_method = f"{url}-gpu-{gpu}"
        runner = ModelRunner(config, 0, [], device=gpu, prefill_only=gpu in prefill)
        checking, ties = False, []
        def sample_reference(logits, temperatures):
            return check_argmax(logits, reference_rows(seqs, expected, is_prefill), ties)

        for link in links.values():
            link.set_caches(runner.kv_caches)
        bytes_per_token = sum(cache.numel() * cache.element_size() // (config.num_kvcache_blocks * config.kvcache_block_size)
                              for cache in runner.kv_caches)
        conn.send(("ready", config.num_kvcache_blocks, bytes_per_token, os.getpid(),
                   dict(torch=str(torch.__version__), cuda=torch.version.cuda, nccl=torch.cuda.nccl.version(),
                        dtype=str(config.hf_config.dtype), uuid=str(torch.cuda.get_device_properties(gpu).uuid))))
        while True:
            if pending:
                begin, end, link, spans = pending
                if end.query():
                    checksum = None
                    if checking:
                        # 校验模式才把 KV 读回 CPU。正式请求全程走 NCCL。
                        h = hashlib.sha256()
                        for a, b in spans:
                            for layer in link.layers:
                                h.update(layer[a:b].view(torch.uint8).cpu().numpy())
                        checksum = h.hexdigest()
                    conn.send(("kv", (begin.elapsed_time(end), checksum)))
                    pending = None
                elif not conn.poll(.0003):
                    continue
            cmd, *args = conn.recv()
            if cmd == "run":
                seqs, is_prefill = args[:2]
                if checking:
                    expected = args[2]
                verify = not is_prefill and any(s.num_scheduled_tokens > 1 for s in seqs)
                tokens = runner.run(seqs, is_prefill, verify, sample_reference if checking else None)
                if verify:
                    # 动态早停在 D 进程内决定实际长度，和本轮结果一起传回调度器。
                    conn.send(("tokens", tokens, [len(s.draft_tokens) for s in seqs]))
                else:
                    conn.send(("tokens", tokens))
            elif cmd == "transfer":
                peer, spans, send = args
                link = links[peer]
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record(link.stream)
                link.transfer_spans(spans, send=send)
                end.record(link.stream)
                pending = begin, end, link, spans
            elif cmd == "check":
                checking = args[0]
                conn.send(("checked", ties))
                ties = []
            elif cmd == "reset":
                torch.manual_seed(args[0])
                graphs = len(getattr(runner, "graphs", {}))
                if runner.draft is not None:
                    graphs += len(runner.draft.graphs)
                conn.send(("reset", graphs, counters["stats"]["unique_graphs"]))
            elif cmd == "close":
                clean = True
                break
            else:
                raise ValueError(cmd)
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        conn.close()
        for link in links.values():
            link.close(abort=not clean)
        if runner is not None:
            runner.exit()


class Stage:
    def __init__(self, gpu, pipe, process):
        self.gpu, self.pipe, self.process = gpu, pipe, process
        self.batch = self.transfer = None
        self.prefill = False
        self.reserved = self.prompt_work = self.admissions = 0


class PDPool:
    def __init__(self, model, prefill_gpus, decode_gpus, *, cpu_sets=None, **options):
        self.prefill = list(prefill_gpus)
        self.decode = list(decode_gpus)
        devices = self.prefill + self.decode
        if not self.decode or len(set(devices)) != len(devices) or any(type(g) is not int or g < 0 for g in devices):
            raise ValueError("Select distinct nonnegative GPU indices and at least one decode GPU")
        self.config = ModelConfig(model, **options)
        if self.config.tensor_parallel_size != 1 or self.config.prefill_gpu is not None:
            raise ValueError("PDPool uses one model replica per assigned GPU")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model, local_files_only=True)
        self.config.eos = tokenizer.eos_token_id
        if self.config.draft_hf_config is not None:
            draft = AutoTokenizer.from_pretrained(self.config.draft_model, local_files_only=True)
            if tokenizer.get_vocab() != draft.get_vocab():
                raise ValueError("Target and draft tokenizers must use the same token IDs")
        self.size = Sequence.block_size = self.config.kvcache_block_size
        self.stages = {}
        self.checking = False
        self.context = get_context("spawn")
        try:
            for g in devices:
                parent, child = self.context.Pipe()
                proc = self.context.Process(target=gpu_worker, args=(child, self.config, g, self.prefill,
                                            self.decode, (cpu_sets or {}).get(g)))
                proc.start()
                child.close()
                self.stages[g] = Stage(g, parent, proc)
            for st in self.stages.values():
                tag, st.blocks, st.bytes_per_token, st.pid, st.runtime = self.message(st)
                if tag != "ready":
                    raise RuntimeError("Worker did not register")
            if len({s.bytes_per_token for s in self.stages.values()}) != 1:
                raise RuntimeError("Workers disagree on the KV layout")
            self.reset()
        except BaseException:
            self.close(abort=True)
            raise

    def message(self, st):
        try:
            msg = st.pipe.recv()
        except (EOFError, OSError) as err:
            raise RuntimeError(f"GPU {st.gpu} worker disconnected") from err
        if msg[0] == "error":
            raise RuntimeError(f"GPU {st.gpu}: {msg[1]}")
        return msg

    def graph_counts(self, seed):
        for st in self.stages.values():
            st.pipe.send(("reset", seed + st.gpu))
        return {g: self.message(st)[1:] for g, st in self.stages.items()}

    def check_mode(self, enabled):
        if self.live:
            raise RuntimeError("Drain requests before changing KV verification mode")
        for st in self.stages.values():
            st.pipe.send(("check", enabled))
        ties = []
        for st in self.stages.values():
            tag, found = self.message(st)
            if tag != "checked":
                raise RuntimeError("Worker did not switch verification mode")
            ties.extend((st.gpu, *x) for x in found)
        self.checking = enabled
        return ties

    def reset(self, seed=0):
        if getattr(self, "live", 0) or any(st.batch or st.transfer for st in self.stages.values()):
            raise RuntimeError("Drain the pool before resetting it")
        self.live = 0
        self.ready = deque()
        self.events = []
        for st in self.stages.values():
            config = copy(self.config)
            config.num_kvcache_blocks = st.blocks
            st.scheduler = Scheduler(config)
            st.reserved = st.prompt_work = st.admissions = 0
            st.stats = dict.fromkeys(("prefill_steps", "decode_steps", "prefill_tokens_computed",
                "decode_steps_with_prefill_pending", "preemptions", "kv_p_to_d_bytes", "kv_d_to_p_bytes",
                "kv_p_to_d_stream_ms", "kv_d_to_p_stream_ms", "handoffs", "verify_steps",
                "draft_tokens_verified", "draft_tokens_accepted", "decode_tokens"), 0)
        self.routes = {}
        self.verified = []
        return self.graph_counts(seed)

    def capacity(self, seq):
        return (seq.num_prompt_tokens + seq.max_tokens - 2) // self.size + 1

    def add_request(self, token_ids, sampling_params):
        if not isinstance(sampling_params, SamplingParams) or not isinstance(token_ids, list) or not token_ids:
            raise ValueError("Provide nonempty token IDs and sampling parameters")
        if any(type(t) is not int or not 0 <= t < self.config.hf_config.vocab_size for t in token_ids):
            raise ValueError("Invalid token ID")
        seq = Sequence(token_ids, sampling_params)
        if len(seq) + seq.max_tokens > self.config.max_model_len or not any(
                self.capacity(seq) <= self.stages[d].blocks for d in self.decode):
            raise ValueError("Request exceeds context or decode KV capacity")
        candidates = [self.stages[g] for g in self.prefill or self.decode
                      if (seq.num_blocks if self.prefill else self.capacity(seq)) <= self.stages[g].blocks]
        if not candidates:
            raise ValueError("Request exceeds prefill KV capacity")
        # 输入长度在到达时已知。选剩余 prompt 工作少的卡，平手时先照顾较空的副本。
        st = min(candidates, key=lambda s: (s.prompt_work, len(s.scheduler.running), s.admissions))
        st.admissions += 1
        st.prompt_work += len(seq)
        seq.pd_prompt_work = len(seq)
        seq.pd_p_gpu, seq.pd_d_gpu = st.gpu, None if self.prefill else st.gpu
        seq.pd_dispatch_ns = None
        seq.pd_prefill_done_ns = seq.pd_transfer_ns = None
        st.scheduler.add(seq)
        self.live += 1
        return seq

    def schedule_compute(self):
        for g, st in self.stages.items():
            if st.batch:
                continue
            s = st.scheduler
            seqs = s.schedule_prefill() if (not self.prefill or g in self.prefill) else []
            is_prefill = bool(seqs)
            if is_prefill:
                if not self.prefill:
                    s.admit_to_decode(seqs)
                now = time.monotonic_ns()
                for seq in seqs:
                    if seq.pd_dispatch_ns is None:
                        seq.pd_dispatch_ns = now
                st.stats["prefill_steps"] += 1
                st.stats["prefill_tokens_computed"] += sum(seq.num_scheduled_tokens for seq in seqs)
            elif g in self.decode:
                before = len(s.waiting)
                seqs = s.schedule_decode()
                st.stats["preemptions"] += len(s.waiting) - before
                if self.prefill and s.waiting:
                    raise RuntimeError("Reserved decode KV was exhausted")
                if seqs:
                    st.stats["decode_steps"] += 1
                    st.stats["decode_steps_with_prefill_pending"] += bool(self.prefill and any(
                        self.stages[p].batch or self.stages[p].scheduler.waiting or self.stages[p].transfer
                        for p in self.prefill))
            if seqs:
                st.batch, st.prefill = seqs, is_prefill
                command = ("run", seqs, is_prefill)
                if self.checking:
                    expected = []
                    for seq in seqs:
                        ref = getattr(seq, "pd_expected", ())
                        start = seq.num_completion_tokens
                        if is_prefill and seq.num_cached_tokens + seq.num_scheduled_tokens < len(seq):
                            ref = ()
                        expected.append(ref[start:start + (1 if is_prefill else seq.num_scheduled_tokens)])
                    command += (expected,)
                st.pipe.send(command)

    def finish_batch(self, st, tokens, draft_counts=None):
        seqs, st.batch = st.batch, None
        if st.prefill:
            for seq in seqs:
                remaining = max(0, seq.num_prompt_tokens - seq.num_cached_tokens - seq.num_scheduled_tokens)
                st.prompt_work -= seq.pd_prompt_work - remaining
                seq.pd_prompt_work = remaining
        if st.gpu in self.prefill:
            for seq, token in zip(seqs, tokens, strict=True):
                st.scheduler.block_manager.hash_blocks(seq, True)
                seq.num_cached_tokens += seq.num_scheduled_tokens
                seq.num_scheduled_tokens = 0
                if seq.num_cached_tokens == len(seq):
                    seq.pd_prefill_done_ns = time.monotonic_ns()
                    self.ready.append((st, seq, token))
        else:
            self.commit(st, seqs, tokens, st.prefill, draft_counts)

    def commit(self, st, seqs, tokens, prefill, draft_counts=None):
        old = [s.num_completion_tokens for s in seqs]
        emitted = st.scheduler.postprocess(seqs, tokens, prefill)
        now = time.monotonic_ns()
        if not prefill:
            st.stats["decode_tokens"] += emitted
        if draft_counts is not None and any(draft_counts):
            st.stats["verify_steps"] += 1
            st.stats["draft_tokens_verified"] += sum(draft_counts)
            # 每段最后一枚是纠正或奖励 token；遇到 EOS 只计实际提交的草稿前缀。
            st.stats["draft_tokens_accepted"] += sum(min(len(result) - 1, seq.num_completion_tokens - count)
                                                    for seq, count, result in zip(seqs, old, tokens, strict=True))
        for seq, count in zip(seqs, old):
            if seq.num_completion_tokens > count:
                self.events.append((seq, now))
            if seq.is_finished:
                self.live -= 1
                if self.prefill:
                    st.reserved -= self.capacity(seq)

    def schedule_transfers(self):
        # P/D 各有自己的块表。接收前 D 留好本请求的增长空间，传完才允许首个 decode。
        delayed = deque()
        while self.ready:
            p, seq, token = self.ready.popleft()
            need = self.capacity(seq)
            candidates = [self.stages[g] for g in self.decode if not self.stages[g].transfer
                          and self.stages[g].reserved + need <= self.stages[g].blocks]
            if p.transfer or not candidates:
                delayed.append((p, seq, token))
                continue
            d = min(candidates, key=lambda st: (len(st.scheduler.running), st.reserved, st.admissions))
            d.admissions += 1
            d.reserved += need
            seq.pd_d_gpu = d.gpu
            # 暂存 P 的块引用。D 分配自己的物理块，双方各自完成 NCCL 后才能释放或使用。
            held, seq.block_table = seq.block_table, []
            d.scheduler.block_manager.allocate(seq, 0)
            spans = handoff_spans(held, seq.block_table, len(seq), self.size)
            job = (p, d, seq, token, held, {})
            p.transfer = d.transfer = job
            d.stats["handoffs"] += 1
            d.stats["kv_p_to_d_bytes"] += len(seq) * d.bytes_per_token
            edge = (p.gpu, d.gpu)
            self.routes[edge] = self.routes.get(edge, 0) + 1
            seq.pd_transfer_ns = time.monotonic_ns()
            p.pipe.send(("transfer", d.gpu, [(a, a + n) for a, b, n in spans], True))
            d.pipe.send(("transfer", p.gpu, [(b, b + n) for a, b, n in spans], False))
        self.ready = delayed

    def finish_transfer(self, st, result):
        p, d, seq, token, held, finished = st.transfer
        elapsed, checksum = result
        finished[st.gpu] = checksum
        if st is d:
            d.stats["kv_p_to_d_stream_ms"] += elapsed
        if len(finished) < 2:
            return
        if finished[p.gpu] != finished[d.gpu]:
            raise RuntimeError(f"KV checksum differs on GPU {p.gpu} and GPU {d.gpu}")
        if checksum is not None:
            self.verified.append((p.gpu, d.gpu, len(seq), checksum))
        p.transfer = d.transfer = None
        p.scheduler.block_manager.release_blocks(held)
        seq.num_scheduled_tokens = len(seq)
        d.scheduler.admit_to_decode([seq])
        self.commit(d, [seq], [token], True)

    def step(self, timeout=None):
        if not self.live:
            return []
        self.events = []
        try:
            self.schedule_transfers()
            self.schedule_compute()
            active = {s.pipe: s for s in self.stages.values() if s.batch or s.transfer}
            if not active:
                raise RuntimeError("No runnable batch or KV transfer")
            for pipe in wait(active, timeout=timeout):
                st = active[pipe]
                message = self.message(st)
                if message[0] == "tokens":
                    self.finish_batch(st, *message[1:])
                elif message[0] == "kv":
                    self.finish_transfer(st, message[1])
                else:
                    raise RuntimeError(f"Unexpected worker event {message[0]}")
            return self.events
        except BaseException:
            self.close(abort=True)
            raise

    def close(self, abort=False):
        abort = abort or bool(getattr(self, "live", 0))
        stages = list(self.stages.values())
        for st in stages:
            if st.process.is_alive():
                if abort:
                    st.process.terminate()
                else:
                    try:
                        st.pipe.send(("close",))
                    except (BrokenPipeError, OSError):
                        pass
        for st in stages:
            st.process.join(15)
            if st.process.is_alive():
                st.process.kill()
                st.process.join()
            st.pipe.close()
        self.stages.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(abort=exc_type is not None)

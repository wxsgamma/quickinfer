import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
from threading import Event, Thread
from unittest.mock import patch

import torch
from quickinfer import LLM, SamplingParams
from quickinfer.llmengine.block_manager import BlockManager
from quickinfer.llmengine.sequence import Sequence
from quickinfer.server import create_server
from quickinfer.utils.context import reset_context


@torch.inference_mode()
def check_kv(llm, actual, cached_blocks):
    runner, manager = llm.model_runner, llm.scheduler.block_manager
    length = actual.num_prompt_tokens
    ref = Sequence(actual.prompt_token_ids)
    # 已命中的前缀用同一块；在 D 的独立空块重算后缀，逐层核对从 P 收到的 K 和 V。
    manager.allocate(ref, cached_blocks)
    try:
        while ref.num_cached_tokens < length:
            start = ref.num_cached_tokens
            ref.num_scheduled_tokens = min(length - start, llm.config.max_num_batched_tokens)
            ids, pos = runner.prepare_prefill([ref])
            if runner.draft is not None:
                runner.draft.model(ids, pos)
            runner.model(ids, pos)
            ref.num_cached_tokens += ref.num_scheduled_tokens
        for cache in runner.kv_caches:
            for i in range(cached_blocks, ref.num_blocks):
                end = min(manager.block_size, length - i * manager.block_size)
                got = cache[:, :, actual.block_table[i], :end]
                expected = cache[:, :, ref.block_table[i], :end]
                torch.testing.assert_close(got, expected, rtol=0, atol=0)
    finally:
        reset_context()
        manager.deallocate(ref)


def check_handoff(llm):
    runner, worker = llm.model_runner, llm.prefill_worker
    p, d = llm.config.prefill_gpu, llm.config.decode_gpu
    assert worker.device == p and all(cache.device.index == d for cache in runner.kv_caches)
    manager = llm.scheduler.block_manager
    with torch.inference_mode():
        for cache in runner.kv_caches:
            cache.fill_(19)
        torch.cuda.synchronize(d)
    tokens = llm.tokenizer.encode("The seasons change, and the rain gives way to clear skies. " * 80)
    llm.add_request(tokens[:255], SamplingParams(.01, 64, True))
    first = llm.scheduler.waiting[-1]
    chunks = []
    while not first.num_completion_tokens:
        chunks.append(llm.step()[1])
    assert chunks == [128, 127], chunks
    for cache in runner.kv_caches:
        assert (cache[:, :, first.block_table[0], 255] == 19).all()
    check_kv(llm, first, 0)
    llm.step()
    shared = first.block_table[0]
    assert not manager.blocks[shared].prefill_kv_ready

    llm.add_request(first.token_ids[:256] + tokens[256:698], SamplingParams(.01, 8, True))
    second = llm.scheduler.waiting[-1]
    restores, overlap = [], []
    prefill, decode = llm.scheduler.schedule_prefill, runner.run

    def schedule(restore):
        seqs = prefill(restore)
        if seqs:
            restores.append(set(restore))
        return seqs

    def run(seqs, is_prefill, is_draft=False):
        assert not is_prefill and torch.cuda.current_device() == d
        assert second not in seqs
        if worker.pending and not worker.connection.poll():
            overlap.append(len(seqs))
        return decode(seqs, is_prefill, is_draft)

    with patch.object(llm.scheduler, "schedule_prefill", schedule), patch.object(runner, "run", run):
        with torch.cuda.nvtx.range("PD-overlap"), torch.cuda.device(p):
            while not second.num_completion_tokens:
                llm.step()
    assert restores == [{shared}, set(), set(), set()], restores
    assert overlap and manager.blocks[shared].prefill_kv_ready
    check_kv(llm, second, 1)
    while not llm.is_finished():
        llm.step()
    assert (first.num_completion_tokens, second.num_completion_tokens) == (64, 8)
    assert len(manager.free_block_ids) == len(manager.blocks)
    print(f"P=cuda:{p} D=cuda:{d} KV exact ({len(runner.kv_caches)} models), "
          f"decode steps started during prefill={len(overlap)}", flush=True)


def check_http(llm):
    decoding = Event()
    overlaps = []
    run = llm.model_runner.run

    def decode(seqs, is_prefill, is_draft):
        assert not is_prefill
        decoding.set()
        if llm.prefill_worker.pending:
            overlaps.append(len(seqs))
        return run(seqs, is_prefill, is_draft)

    with create_server(llm, port=0) as server, patch.object(llm.model_runner, "run", decode):
        thread = Thread(target=server.serve_forever)
        thread.start()

        def request(prompts, limit):
            conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=180)
            try:
                conn.request("POST", "/generate", json.dumps(dict(prompts=prompts, max_tokens=limit, ignore_eos=True)))
                reply = conn.getresponse()
                return reply.status, json.loads(reply.read())
            finally:
                conn.close()

        try:
            with ThreadPoolExecutor(2) as clients:
                a = clients.submit(request, ["The capital of France is"], 48)
                assert decoding.wait(120)
                b = clients.submit(request, [[37] * 400, [12, 18]], 2)
                for future, lengths in ((b, [2, 2]), (a, [48])):
                    status, data = future.result()
                    assert status == 200 and [len(o["token_ids"]) for o in data["outputs"]] == lengths, data
            assert overlaps and request([[999999999]], 1)[0] == 400
        finally:
            server.shutdown()
            thread.join()
    print("two concurrent HTTP clients passed", flush=True)


def check_pressure(llm):
    manager = BlockManager(2, llm.config.kvcache_block_size)
    with patch.object(llm.scheduler, "block_manager", manager):
        prompts = [[token] * 255 for token in (7, 13, 19)]
        outputs = llm.generate(prompts, SamplingParams(.01, 8, True), use_tqdm=False)
        assert [len(o["token_ids"]) for o in outputs] == [8, 8, 8]
        assert sorted(manager.free_block_ids) == [0, 1]
    print("three cross-block requests with two KV blocks passed", flush=True)


def check_pending_close(llm):
    llm.add_request([17], SamplingParams(.01, 32, True))
    llm.step()
    llm.add_request([23] * 300, SamplingParams(.01, 2, True))
    llm.step()
    assert llm.prefill_worker.pending and llm.prefill_worker.token_ids is None
    # 这时还没给 P 匹配 recv，退出也要把在途的 KV 收完。
    llm.close()
    assert llm.prefill_worker.process.exitcode == 0
    print("closed with a prefill transfer in flight", flush=True)


if __name__ in ("__main__", "__mp_main__"):
    # 逐元素重算需要 P、D 使用相同算子；各进程的 torch.compile 缓存可能不同。D 仍重放 CUDA Graph。
    torch.compiler.set_stance("force_eager")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument("--prefill-gpu", type=int, default=0)
    parser.add_argument("--decode-gpu", type=int, default=1)
    args = parser.parse_args()
    with LLM(**vars(args), max_model_len=768, max_num_batched_tokens=128, max_num_seqs=4,
             num_kvcache_blocks=12, enabled_draft=bool(args.draft_model), draft_method="modeldraft") as llm:
        child = llm.prefill_worker.process
        check_handoff(llm)
        check_http(llm)
        check_pressure(llm)
        check_pending_close(llm)
    assert child.exitcode == 0

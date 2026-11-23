"""Opt-in GPU integration test, run with uv run python tests/gpu_smoke.py --model PATH."""

import argparse
import http.client
import json
from itertools import accumulate
from threading import Thread
from unittest.mock import patch

import torch
from quickinfer import LLM, SamplingParams
from quickinfer.server import create_server
from quickinfer.benchmark import run_benchmark
from quickinfer.cli import build_parser
from quickinfer.utils.context import reset_context, set_context


def check_http(llm):
    # HTTP is serviced on the engine's main thread. Only the client uses a thread.
    results = []
    with create_server(llm, port=0) as server:
        def client():
            try:
                conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=120)
                conn.request("POST", "/generate", json.dumps({
                    "prompts": ["The capital of France is"], "max_tokens": 4, "ignore_eos": True,
                }), {"Content-Type": "application/json"})
                response = conn.getresponse()
                results.append((response.status, json.loads(response.read())))
                conn.close()
            except Exception as error:
                results.append(error)
        thread = Thread(target=client, daemon=True)
        thread.start()
        server.timeout = 120
        server.handle_request()
        thread.join(timeout=125)
    assert len(results) == 1 and isinstance(results[0], tuple), results
    status, data = results[0]
    assert status == 200, results
    assert len(data["outputs"][0]["token_ids"]) == 4, results


@torch.inference_mode()
def check_draft_graph(llm):
    runner = llm.model_runner
    scheduler = llm.scheduler
    torch.manual_seed(731)
    for prompt in ([7] * 255, [11, 12, 13]):
        llm.add_request(prompt, SamplingParams(max_tokens=8, ignore_eos=True))
    while scheduler.waiting:
        llm.step()
    seqs = list(scheduler.running)
    seqs[0].draft_tokens = [17, 23, 29, 31]
    for seq in seqs:
        seq.num_scheduled_tokens = 1 + len(seq.draft_tokens)
        scheduler.block_manager.may_append(seq, seq.num_scheduled_tokens)

    def verify():
        ids, pos = runner.prepare_decode(seqs, True)
        return runner.run_model(ids, pos, False).float()

    # 第一条跨块，第二条没有草稿。补图时不能覆盖已有的 KV。
    prefix = runner.kv_cache[:, :, seqs[0].block_table[0], 0].clone()
    try:
        runner.enforce_eager = True
        expected = verify()
        runner.enforce_eager = False
        actual = verify()
        assert actual.size(0) == 6 and runner.graph_size(6) in runner.graphs
        # 两种 attention 的 BF16 舍入会有差别，逐行比较采样分布。
        distance = (actual.softmax(-1) - expected.softmax(-1)).abs().sum(-1) / 2
        assert torch.all(distance < 0.1), distance.tolist()
        assert torch.equal(prefix, runner.kv_cache[:, :, seqs[0].block_table[0], 0])
        # 改最后一个草稿，前面四行和另一条请求的 logits 应该完全一样。
        seqs[0].draft_tokens[-1] = 71
        graph, _ = runner.graphs[runner.graph_size(6)]
        with patch.object(graph, "replay", wraps=graph.replay) as replay:
            changed = verify()
            replay.assert_called_once_with()
        assert torch.equal(changed[:4], actual[:4]) and torch.equal(changed[5], actual[5])
        assert not torch.equal(changed[4], actual[4])
    finally:
        runner.enforce_eager = False
        reset_context()
    scheduler.postprocess(seqs, [[19], [21]], False)
    assert len(seqs[0].block_table) == 1
    while not llm.is_finished():
        llm.step()
    print(json.dumps({"draft_graph_cross_block": "passed"}), flush=True)


@torch.inference_mode()
def check_model_draft(model, draft_model, eager, dynamic, early=False):
    with LLM(model, draft_model=draft_model, enabled_draft=True, draft_method="modeldraft", num_draft=4,
             dynamic_verify=dynamic, draft_early_stop=early, enforce_eager=eager, max_model_len=512,
             max_num_batched_tokens=128, max_num_seqs=4,
             num_kvcache_blocks=8) as llm:
        torch.manual_seed(731)
        runner = llm.model_runner
        manager = llm.scheduler.block_manager
        verify = runner.draft.verify
        totals = dict(target_rows=0, draft_rows=0, full=0, rejected=0)
        max_tv = dict(target_rows=0.0, draft_rows=0.0)
        draft_graphs = {id(graph): key for key, graph in runner.draft.graphs.items()}
        replays = {True: 0, False: 0}
        original_replay = torch.cuda.CUDAGraph.replay
        original_gen = runner.draft.gen
        original_step = runner.draft.forward_step
        trace = []

        def record(state):
            trace.append([tuple(row) for row in state[:, 1:].tolist() if row[1] >= 0])

        def replay(graph):
            if id(graph) in draft_graphs:
                size, sample = draft_graphs[id(graph)]
                replays[sample] += 1
                if early:
                    record(runner.draft.graph_state[:size])
            return original_replay(graph)

        def forward_step(state, *args):
            if early and eager:
                record(state)
            return original_step(state, *args)

        def generate(seqs, tables, temperatures, cost=None):
            limits = [seq.num_scheduled_tokens - 1 for seq in seqs]
            count = max(limits)
            if not eager and not early and count and not replays[True]:
                # 重放要产生新的随机数，重设种子后同一前缀又能复现。
                with torch.random.fork_rng(devices=[tables.device]):
                    hot = torch.full_like(temperatures, 100.)
                    draws = []
                    for seed in (17, None, 17):
                        if seed is not None:
                            torch.cuda.manual_seed(seed)
                        original_gen(seqs, tables, hot)
                        draws.append([seq.draft_tokens.copy() for seq in seqs])
                    assert draws[0] == draws[2] and draws[0] != draws[1]
            before = replays.copy()
            trace.clear()
            probs = original_gen(seqs, tables, temperatures, cost)
            if cost is not None:
                lengths = [len(seq.draft_tokens) for seq in seqs]
                offsets = list(accumulate(limits, initial=0))
                # 按真实前向行逐步对照。某条停在第 k 枚后，只能再看到别的请求。
                assert len(trace) == 1 + max(lengths)
                for step, actual in enumerate(trace):
                    expected = [(len(seq) - 1 + step, limit - step, offset)
                                for seq, limit, length, offset in zip(seqs, limits, lengths, offsets)
                                if step <= length]
                    assert sorted(actual) == sorted(expected), (actual, expected)
            elif not eager:
                assert replays[True] - before[True] == count and replays[False] - before[False] == 1
            return probs

        def check(drafts, p, q, draft_rows=None):
            outputs = verify(drafts, p, q, draft_rows)
            if draft_rows is not None:
                q = q[draft_rows]
            seqs = gen.call_args.args[0]
            offset = 0
            for i, (seq, draft, output) in enumerate(zip(seqs, drafts, outputs)):
                ids = torch.tensor(seq.token_ids + draft, device=p.device)
                n = len(ids)
                positions = torch.arange(n, device=p.device)
                cu = torch.tensor([0, n], dtype=torch.int32, device=p.device)
                # 完整前缀从头计算，用 -1 禁止这次对照前向改写线上 KV。
                set_context(True, True, cu, cu, n, n, slot_mapping=torch.full_like(positions, -1, dtype=torch.int32))
                k = len(draft)
                start = offset + i
                for name, model, actual in [
                    ("target_rows", runner.model, p[start:start + k + 1]),
                    ("draft_rows", runner.draft.model, q[offset:offset + k]),
                ]:
                    if len(actual) == 0:
                        continue
                    first = len(seq) - 1
                    hidden = model(ids, positions)[first:first + len(actual)]
                    expected = (model.compute_logits(hidden).float() / seq.temperature).softmax(-1)
                    distance = (actual - expected).abs().sum(-1).max().item() / 2
                    assert distance < 0.1, (name, len(seq), draft, distance)
                    max_tv[name] = max(max_tv[name], distance)
                    totals[name] += len(actual)
                if k:
                    totals["full" if len(output) == k + 1 else "rejected"] += 1
                offset += k
            return outputs

        prompt = llm.tokenizer.encode("A traveler found an old map in a library. ")
        llm.add_request((prompt * 256)[:255], SamplingParams(0.8, 20, True))
        llm.add_request("Name a color.", SamplingParams(1.1, 2, True))
        llm.add_request("Write a short story about the sea.", SamplingParams(0.9, 14, True))
        with (
            patch.object(torch.cuda.CUDAGraph, "replay", replay),
            patch.object(runner.draft, "forward_step", side_effect=forward_step),
            patch.object(runner.draft, "gen", side_effect=generate) as gen,
            patch.object(runner.draft, "verify", check),
        ):
            while llm.scheduler.waiting:
                llm.step()
            first = llm.scheduler.running[0]
            if dynamic:
                assert runner.verify_cost is not None and sum(runner.verify_cost) > 0
                runner.verify_cost = (0., 1.)  # 全裁空，锚点仍须写在第一个块的末槽。
            llm.step()  # 第一条草稿跨第 256 个位置，第二条没有草稿。
            if dynamic:
                assert first.num_cached_tokens == 256 and len(first.block_table) == 1
                runner.verify_cost = (4., 1.)
            llm.add_request(first.token_ids[:256] + [42], SamplingParams(0.7, 9, True))
            assert manager.can_allocate(llm.scheduler.waiting[0]) == 1
            while not llm.is_finished():
                llm.step()
        assert len(manager.free_block_ids) == llm.config.num_kvcache_blocks
        assert totals["draft_rows"] > 0
        print(json.dumps(dict(model_draft="eager" if eager else "cuda_graph", dynamic=dynamic, early=early,
                              **totals, max_tv=max_tv,
                              sample_replays=replays[True], kv_replays=replays[False])), flush=True)


def run(model, eager):
    with LLM(model, enforce_eager=eager, max_model_len=1024,
             max_num_batched_tokens=128, max_num_seqs=3,
             num_kvcache_blocks=16, gpu_memory_utilization=0.15) as llm:
        if not eager:
            check_draft_graph(llm)
        events = []
        original_schedule = llm.scheduler.schedule

        def schedule():
            seqs, prefill = original_schedule()
            events.append({"prefill": prefill, "batch": len(seqs),
                           "cached": [s.num_cached_tokens for s in seqs],
                           "scheduled": [s.num_scheduled_tokens for s in seqs]})
            return seqs, prefill

        llm.scheduler.schedule = schedule
        torch.manual_seed(123)
        prefix = [42] * 600
        prompts = [prefix, prefix[:-1] + [43], prefix[:-1] + [44]]
        outputs = llm.generate(prompts, SamplingParams(max_tokens=8, ignore_eos=True), use_tqdm=False)
        assert len(outputs) == 3 and all(len(o["token_ids"]) == 8 for o in outputs)
        assert any(e["prefill"] and max(e["cached"]) >= 512 for e in events), events
        assert any(not e["prefill"] and e["batch"] == 3 for e in events), events
        assert max(sum(e["scheduled"]) for e in events if e["prefill"]) <= 128
        assert len(llm.scheduler.block_manager.free_block_ids) == llm.config.num_kvcache_blocks
        check_http(llm)
        prompt = llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is 2 plus 2? Answer briefly."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        answer = llm.generate([prompt], SamplingParams(temperature=0.6, max_tokens=32), use_tqdm=False)[0]
        result = {"mode": "eager" if eager else "cuda_graph", "sequences": 3,
                  "tokens_per_sequence": 8, "prefix_cached_tokens": 512, "decode_batch": 3,
                  "http_generation": "passed", "answer": answer["text"],
                  "kv_blocks": llm.config.num_kvcache_blocks}
    llm.exit()  # Explicit cleanup and the atexit handler must be safely repeatable.
    assert not torch.distributed.is_initialized()
    result["cleanup"] = "passed"
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model")
    args = parser.parse_args()
    if args.draft_model:
        for eager in (True, False):
            for dynamic, early in ((False, False), (True, False), (True, True)):
                check_model_draft(args.model, args.draft_model, eager, dynamic, early)
        raise SystemExit(0)
    for eager in (True, False):
        run(args.model, eager)
    # Regression: a valid two-token context must not fail in benchmark warm-up.
    short_args = build_parser().parse_args([
        "benchmark", "--model", args.model, "--num-seqs", "1",
        "--input-len", "1", "--output-len", "1", "--max-model-len", "2",
        "--max-num-seqs", "1", "--max-num-batched-tokens", "2",
        "--num-kvcache-blocks", "2", "--gpu-memory-utilization", "0.15", "--enforce-eager",
    ])
    result = run_benchmark(short_args)
    assert result["output_tokens"] == 1, result
    print(json.dumps({"short_context_benchmark": "passed"}), flush=True)

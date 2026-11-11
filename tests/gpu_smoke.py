"""Opt-in GPU integration test, run with uv run python tests/gpu_smoke.py --model PATH."""

import argparse
import http.client
import json
from threading import Thread

import torch
from quickinfer import LLM, SamplingParams
from quickinfer.server import create_server
from quickinfer.benchmark import run_benchmark
from quickinfer.cli import build_parser


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


def run(model, eager):
    with LLM(model, enforce_eager=eager, max_model_len=1024,
             max_num_batched_tokens=128, max_num_seqs=3,
             num_kvcache_blocks=16, gpu_memory_utilization=0.15) as llm:
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
    args = parser.parse_args()
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

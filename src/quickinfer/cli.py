import argparse
import json


ENGINE_OPTIONS = (
    "dtype", "max_model_len", "max_num_seqs", "max_num_batched_tokens",
    "gpu_memory_utilization", "tensor_parallel_size", "enforce_eager",
    "kvcache_block_size", "num_kvcache_blocks",
)


def engine_kwargs(args):
    return {name: getattr(args, name) for name in ENGINE_OPTIONS if hasattr(args, name)}


def run_generate(args):
    from quickinfer import LLM, SamplingParams
    with LLM(args.model, **engine_kwargs(args)) as llm:
        prompts = args.prompt
        if args.chat:
            prompts = [llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            ) for prompt in prompts]
        outputs = llm.generate(prompts, SamplingParams(args.temperature, args.max_tokens), use_tqdm=False)
        print(json.dumps(outputs, ensure_ascii=False))
    return 0


def run_server(args):
    from quickinfer import LLM
    from quickinfer.server import serve
    with LLM(args.model, **engine_kwargs(args)) as llm:
        serve(llm, host=args.host, port=args.port)
    return 0


def run_bench(args):
    from quickinfer.benchmark import run_benchmark
    print(json.dumps(run_benchmark(args), indent=2))
    return 0


def add_engine_args(parser):
    parser.add_argument("--model", required=True, help="Local Qwen3 safetensors directory")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="auto")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graph capture/replay")
    parser.add_argument("--kvcache-block-size", type=int, default=256)
    parser.add_argument("--num-kvcache-blocks", type=int, default=-1, help="Optional KV cache block cap")


def build_parser():
    parser = argparse.ArgumentParser(prog="quickinfer", description="Lightweight Qwen3 inference")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="Generate text offline")
    add_engine_args(generate)
    generate.add_argument("--prompt", action="append", required=True, help="Repeat to batch prompts")
    generate.add_argument("--max-tokens", type=int, default=64)
    generate.add_argument("--temperature", type=float, default=0.6)
    generate.add_argument("--chat", action="store_true", help="Apply Qwen3 chat template without thinking")
    generate.set_defaults(func=run_generate)
    serve_parser = subparsers.add_parser("serve", help="Serve a local JSON generation API")
    add_engine_args(serve_parser)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.set_defaults(func=run_server)
    bench = subparsers.add_parser("benchmark", help="Measure synthetic offline generation throughput")
    add_engine_args(bench)
    bench.add_argument("--num-seqs", type=int, default=256)
    bench.add_argument("--input-len", type=int, default=1024)
    bench.add_argument("--output-len", type=int, default=1024)
    bench.add_argument("--seed", type=int, default=0)
    bench.set_defaults(func=run_bench)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

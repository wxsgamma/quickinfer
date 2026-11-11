"""Synthetic fixed-length offline generation benchmark."""

from random import Random
from time import perf_counter

import torch

from quickinfer import LLM, SamplingParams
from quickinfer.cli import engine_kwargs


def run_benchmark(args):
    for name in ("num_seqs", "input_len", "output_len"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.input_len + args.output_len > args.max_model_len:
        raise ValueError("input_len + output_len exceeds max_model_len")
    rng = Random(args.seed)
    torch.manual_seed(args.seed)
    with LLM(args.model, **engine_kwargs(args)) as llm:
        vocab_size = llm.config.hf_config.vocab_size
        prompts = [[rng.randrange(vocab_size) for _ in range(args.input_len)] for _ in range(args.num_seqs)]
        llm.generate([[0]], SamplingParams(max_tokens=1), use_tqdm=False)
        torch.cuda.synchronize()
        start = perf_counter()
        outputs = llm.generate(prompts, SamplingParams(
            temperature=0.6, ignore_eos=True, max_tokens=args.output_len), use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = perf_counter() - start
        total_tokens = sum(len(output["token_ids"]) for output in outputs)
    return {
        "num_sequences": args.num_seqs, "input_length": args.input_len,
        "output_length": args.output_len, "output_tokens": total_tokens,
        "seconds": elapsed, "output_tokens_per_second": total_tokens / elapsed,
        "seed": args.seed, "enforce_eager": args.enforce_eager,
    }

"""Run with uv run python benchmark/bench.py --model /path/to/Qwen3 ..."""

import sys
from quickinfer.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["benchmark", *sys.argv[1:]]))

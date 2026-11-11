"""Run with uv run python example.py --model /path/to/Qwen3."""

import sys
from quickinfer.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["generate", "--chat", "--prompt", "Introduce yourself.", *sys.argv[1:]]))

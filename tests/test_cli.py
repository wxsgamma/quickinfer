import contextlib
import io
import unittest

from quickinfer.cli import build_parser


class CLITest(unittest.TestCase):
    def test_generate_command(self):
        args = build_parser().parse_args([
            "generate", "--model", "/models/qwen3", "--prompt", "Hello",
            "--dtype", "bf16", "--max-tokens", "8", "--enforce-eager",
        ])
        self.assertEqual(args.prompt, ["Hello"])
        self.assertEqual(args.max_tokens, 8)
        self.assertTrue(args.enforce_eager)

    def test_serve_keeps_existing_entrypoint(self):
        args = build_parser().parse_args(["serve", "--model", "/models/qwen3"])
        self.assertEqual(args.port, 8080)
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.dtype, "auto")

    def test_benchmark_accepts_model_and_workload(self):
        args = build_parser().parse_args([
            "benchmark", "--model", "/models/qwen3", "--num-seqs", "3",
            "--input-len", "16", "--output-len", "8",
        ])
        self.assertEqual(args.num_seqs, 3)
        self.assertEqual(args.output_len, 8)

    def test_unsupported_flash_attention_dtypes_are_rejected(self):
        for dtype in ("fp8", "fp32"):
            with self.subTest(dtype=dtype), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    build_parser().parse_args(["serve", "--model", "unused", "--dtype", dtype])


if __name__ == "__main__":
    unittest.main()

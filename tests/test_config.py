import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from quickinfer import SamplingParams
from quickinfer.llmengine.modelconfig import ModelConfig
from quickinfer.llmengine.llm_engine import LLMEngine


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        # Config parsing only: no model is loaded from this fixture.
        (self.path / "model.safetensors").touch()
        (self.path / "config.json").write_text(json.dumps({
            "model_type": "qwen3", "vocab_size": 128, "hidden_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "intermediate_size": 256, "num_hidden_layers": 2,
            "max_position_embeddings": 1024, "torch_dtype": "bfloat16",
        }))

    def tearDown(self):
        self.temp.cleanup()

    def test_dtype_and_context_limit(self):
        config = ModelConfig(str(self.path), dtype="fp16", max_model_len=2048)
        self.assertEqual(config.hf_config.dtype, torch.float16)
        self.assertEqual(config.max_model_len, 1024)
        self.assertEqual(ModelConfig(str(self.path)).hf_config.dtype, torch.bfloat16)

    def test_invalid_configuration_is_rejected(self):
        for kwargs in ({"dtype": "fp8"}, {"max_num_seqs": 0}, {"kvcache_block_size": 128},
                       {"gpu_memory_utilization": 0}, {"tensor_parallel_size": 3},
                       {"num_kvcache_blocks": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ModelConfig(str(self.path), **kwargs)

    def test_instances_do_not_share_ipc_endpoints(self):
        first, second = ModelConfig(str(self.path)), ModelConfig(str(self.path))
        self.assertNotEqual(first.shm_name, second.shm_name)
        self.assertNotEqual(first.distributed_init_method, second.distributed_init_method)

    def test_draft_model_config(self):
        options = dict(enabled_draft=True, draft_method="modeldraft")
        with self.assertRaisesRegex(ValueError, "draft_model"):
            ModelConfig(str(self.path), **options)
        self.assertIsNone(ModelConfig(str(self.path), **options, num_draft=0).draft_hf_config)
        draft = self.path / "draft"
        draft.mkdir()
        (draft / "model.safetensors").touch()
        data = json.loads((self.path / "config.json").read_text())
        data["max_position_embeddings"] = 512
        (draft / "config.json").write_text(json.dumps(data))
        config = ModelConfig(str(self.path), draft_model=str(draft), **options)
        self.assertEqual(config.max_model_len, 512)
        self.assertEqual(config.draft_hf_config.dtype, config.hf_config.dtype)
        data["vocab_size"] += 1
        (draft / "config.json").write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "vocabulary size"):
            ModelConfig(str(self.path), draft_model=str(draft), **options)

    def test_bad_batch_is_rejected_before_enqueue(self):
        # Validation is CPU-only; bypass CUDA construction and use actual generate logic.
        engine = LLMEngine.__new__(LLMEngine)
        engine._closed = False
        engine.config = SimpleNamespace(max_model_len=16, num_kvcache_blocks=1,
                                       kvcache_block_size=256, hf_config=SimpleNamespace(vocab_size=100))
        queued = []
        engine.scheduler = SimpleNamespace(add=queued.append)
        with self.assertRaises(ValueError):
            engine.generate([[1, 2], []], SamplingParams(max_tokens=2), use_tqdm=False)
        self.assertEqual(queued, [])
        with self.assertRaises(ValueError):
            engine.generate([[1]], [], use_tqdm=False)
        with self.assertRaises(ValueError):
            engine._prepare_request([1] * 15, SamplingParams(max_tokens=2))


if __name__ == "__main__":
    unittest.main()

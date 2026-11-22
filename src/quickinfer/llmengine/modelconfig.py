from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from uuid import uuid4

import torch
from transformers import AutoConfig


@dataclass(slots=True)
class ModelConfig:
    model: str
    dtype: str = "auto"
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    hf_config: object = field(init=False, repr=False)
    eos: int = field(init=False, default=-1)
    distributed_init_method: str = field(init=False)
    shm_name: str = field(init=False)
    enabled_draft: bool = False
    draft_method: str = "ngram"
    num_draft: int = 4
    draft_model: str | None = None
    draft_hf_config: object = field(init=False, default=None, repr=False)

    def __post_init__(self):
        if self.draft_method == "n-gram":
            self.draft_method = "ngram"
        if self.enabled_draft and self.draft_method not in ("ngram", "modeldraft"):
            raise ValueError(f"Unsupported draft method: {self.draft_method}")
        if isinstance(self.num_draft, bool) or not isinstance(self.num_draft, int) or self.num_draft < 0:
            raise ValueError("num_draft must be a non-negative integer")
        if self.dtype not in ("auto", "fp16", "bf16"):
            raise ValueError("dtype must be auto, fp16 or bf16 (FlashAttention requires FP16/BF16)")
        for name in ("max_num_batched_tokens", "max_num_seqs", "max_model_len"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.kvcache_block_size <= 0 or self.kvcache_block_size % 256:
            raise ValueError("kvcache_block_size must be a positive multiple of 256")
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError("tensor_parallel_size must be between 1 and 8")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.num_kvcache_blocks != -1 and self.num_kvcache_blocks <= 0:
            raise ValueError("num_kvcache_blocks must be -1 or positive")
        self.model, self.hf_config = self.load_config(self.model)
        resolved = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.dtype)
        if resolved is None:
            resolved = getattr(self.hf_config, "dtype", None)
            if resolved not in (torch.float16, torch.bfloat16):
                resolved = torch.bfloat16
        self.hf_config.dtype = resolved
        if self.enabled_draft and self.draft_method == "modeldraft" and self.num_draft:
            if not self.draft_model:
                raise ValueError("modeldraft requires a local draft_model directory")
            self.draft_model, self.draft_hf_config = self.load_config(self.draft_model)
            if self.draft_hf_config.vocab_size != self.hf_config.vocab_size:
                raise ValueError("Target and draft models must have the same vocabulary size")
            self.draft_hf_config.dtype = resolved
        instance = uuid4().hex
        self.distributed_init_method = Path(tempfile.gettempdir(), f"quickinfer-dist-{instance}").as_uri()
        self.shm_name = f"quickinfer-{instance}"

    def load_config(self, model):
        path = Path(model).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"Model must be a local directory: {path}")
        if not any(path.glob("*.safetensors")):
            raise ValueError(f"No safetensors weights found in {path}")
        hf_config = AutoConfig.from_pretrained(path, local_files_only=True)
        if hf_config.model_type != "qwen3":
            raise ValueError("This engine currently supports dense Qwen3 models only")
        for name in ("num_key_value_heads", "num_attention_heads", "vocab_size", "intermediate_size"):
            if getattr(hf_config, name) % self.tensor_parallel_size:
                raise ValueError(f"{name} must be divisible by tensor_parallel_size")
        self.max_model_len = min(self.max_model_len, hf_config.max_position_embeddings)
        return str(path), hf_config

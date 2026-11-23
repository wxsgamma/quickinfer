import atexit
import gc
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import torch

from quickinfer.llmengine.modelconfig import ModelConfig as Config
from quickinfer.sampling_params import SamplingParams
from quickinfer.llmengine.sequence import Sequence
from quickinfer.llmengine.scheduler import Scheduler
from quickinfer.llmengine.model_runner import ModelRunner
from quickinfer.llmengine.prefill_worker import PrefillWorker


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config) if field.init and field.name != "model"}
        unknown = set(kwargs) - config_fields
        if unknown:
            raise TypeError(f"Unknown engine options: {', '.join(sorted(unknown))}")
        config = self.config = Config(model, **kwargs)
        if not torch.cuda.is_available():
            raise RuntimeError("Quickinfer requires an NVIDIA CUDA GPU")
        if config.tensor_parallel_size > torch.cuda.device_count():
            raise ValueError("tensor_parallel_size exceeds visible CUDA devices")
        if config.prefill_gpu is not None and max(config.prefill_gpu, config.decode_gpu) >= torch.cuda.device_count():
            raise ValueError("prefill_gpu or decode_gpu exceeds visible CUDA devices")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, local_files_only=True)
        if config.draft_hf_config is not None:
            draft_tokenizer = AutoTokenizer.from_pretrained(config.draft_model, use_fast=True, local_files_only=True)
            if draft_tokenizer.get_vocab() != self.tokenizer.get_vocab():
                raise ValueError("Target and draft tokenizers must use the same token IDs")
            del draft_tokenizer
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        self.prefill_worker = None
        ctx = mp.get_context("spawn")# spawn 启动一个干净的python解释器进程，避免fork带来的问题
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        try:
            if config.prefill_gpu is not None:
                # 先建 NCCL 通信组再算 KV 显存，D 的块数上限取 P 已分到的块数。
                self.prefill_worker = PrefillWorker(config, ctx)
                config.num_kvcache_blocks = self.prefill_worker.num_blocks
            self.model_runner = ModelRunner(config, 0, self.events, device=config.decode_gpu)
            if self.prefill_worker is not None:
                self.prefill_worker.link.set_caches(self.model_runner.kv_caches)
        except BaseException:
            if self.prefill_worker is not None:
                self.prefill_worker.close(abort=True)
            for process in self.ps:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
            if hasattr(self, "model_runner") and config.prefill_gpu is not None:
                self.model_runner.exit()
                del self.model_runner
                from quickinfer.layers.rotary_embedding import get_rope
                get_rope.cache_clear()
            elif torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            raise
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self._closed = False
        atexit.register(self.exit)

    def exit(self):
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.exit)
        try:
            if self.prefill_worker is not None:
                self.prefill_worker.close()
        finally:
            self.model_runner.call("exit")
            del self.model_runner
            for p in self.ps:
                p.join()# 等待子进程结束
            from quickinfer.layers.rotary_embedding import get_rope
            get_rope.cache_clear()
            gc.collect()
            torch.cuda.empty_cache()

    close = exit

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit()

    def _prepare_request(self, prompt, sampling_params):
        if self._closed:
            raise RuntimeError("Engine is closed")
        if not isinstance(sampling_params, SamplingParams):
            raise TypeError("sampling_params must be a SamplingParams instance")
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not isinstance(prompt, list) or not prompt:
            raise ValueError("Each prompt must contain at least one token")
        if any(isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < self.config.hf_config.vocab_size for t in prompt):
            raise ValueError("Prompt token IDs must be integers within the model vocabulary")
        if len(prompt) + sampling_params.max_tokens > self.config.max_model_len:
            raise ValueError("Prompt length plus max_tokens exceeds max_model_len")
        capacity = self.config.num_kvcache_blocks * self.config.kvcache_block_size
        if len(prompt) + sampling_params.max_tokens > capacity:
            raise ValueError("Request exceeds KV cache capacity; increase num_kvcache_blocks or GPU memory budget")
        return prompt

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):# 编码prompt
        prompt = self._prepare_request(prompt, sampling_params)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def _enqueue_requests(self, prompts, sampling_params):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError("Provide one SamplingParams per prompt")
        # 整批校验通过再入队，HTTP 用 seq_id 把各自的结果交回原请求。
        seqs = [Sequence(self._prepare_request(p, sp), sp) for p, sp in zip(prompts, sampling_params)]
        for seq in seqs:
            self.scheduler.add(seq)
        return [seq.seq_id for seq in seqs]

    def step(self):
        if self._closed:
            raise RuntimeError("Engine is closed")
        if self.prefill_worker is None:
            return self._run_step(*self.scheduler.schedule())
        try:
            with torch.cuda.device(self.config.decode_gpu):
                return self._run_step(*self.prefill_worker.schedule(self.scheduler))
        except BaseException:
            try:
                self.prefill_worker.close(abort=True)
            finally:
                self.exit()
            raise

    def _run_step(self, seqs, is_prefill, token_ids=None):
        prefill_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else 0
        if token_ids is None:
            is_draft = not is_prefill and any(seq.num_scheduled_tokens > 1 for seq in seqs)
            token_ids = self.model_runner.call("run", seqs, is_prefill, is_draft)
        num_emitted = self.scheduler.postprocess(seqs, token_ids, is_prefill)
        num_tokens = prefill_tokens if is_prefill else -num_emitted
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished() and not (self.prefill_worker and self.prefill_worker.pending)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        if self._closed:
            raise RuntimeError("Engine is closed")
        self._enqueue_requests(prompts, sampling_params)
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs

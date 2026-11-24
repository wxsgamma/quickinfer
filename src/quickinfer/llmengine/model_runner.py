import pickle
from datetime import timedelta
from math import prod
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from quickinfer.llmengine.modelconfig import ModelConfig as Config
from quickinfer.llmengine.draft import Model_Draft
from quickinfer.llmengine.sequence import Sequence
from quickinfer.models.qwen3 import Qwen3ForCausalLM
from quickinfer.layers.sampler import Sampler
from quickinfer.utils.context import set_context, get_context, reset_context
from quickinfer.utils.loader import load_model

class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event], *, device=None, prefill_only=False):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.prefill_only = prefill_only
        self.enforce_eager = prefill_only or config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        torch.cuda.set_device(rank if device is None else device)
        dist.init_process_group("nccl", config.distributed_init_method,
                                world_size=self.world_size, rank=rank, timeout=timedelta(seconds=120))
        default_dtype = torch.get_default_dtype()
        default_device = torch.get_default_device()
        try:
            torch.set_default_dtype(hf_config.dtype)
            torch.set_default_device("cuda")
            self.model = Qwen3ForCausalLM(hf_config)
            load_model(self.model, config.model)
            self.draft = None
            self.verify_cost = None
            if config.draft_hf_config is not None:
                draft_model = Qwen3ForCausalLM(config.draft_hf_config)
                load_model(draft_model, config.draft_model)
                self.draft = Model_Draft(draft_model, self.block_size)
            self.sampler = Sampler()
            self.warmup_model()
            self.allocate_kv_cache()
            if not self.enforce_eager:
                self.capture_cudagraph()
                if self.draft is not None:
                    self.draft.capture_cudagraph(config)
            if self.draft is not None and config.dynamic_verify and not prefill_only:
                self.verify_cost = self.profile_verify()
        finally:
            torch.set_default_device(default_device)
            torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name=config.shm_name, create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=config.shm_name)
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
            if self.draft is not None:
                self.draft.graphs.clear()
        torch.cuda.synchronize(self.kv_cache.device)
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        if n + 4 > self.shm.size:
            raise ValueError("Tensor-parallel command exceeds shared-memory capacity; reduce batch size")
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True, False)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        workspace = peak - current
        models = [(self.model, config.hf_config)]
        if self.draft is not None:
            models.append((self.draft.model, config.draft_hf_config))
            # 验证时 rank 0 同时持有主模型 logits、草稿概率，以及 FP32 归一化和补采的临时行。
            if self.rank == 0 and not self.prefill_only:
                rows = min(config.max_num_batched_tokens, config.max_num_seqs * (1 + config.num_draft))
                batch_size = min(rows, config.max_num_seqs)
                vocab = config.hf_config.vocab_size
                fp32 = torch.float32.itemsize
                verify_bytes = rows * (config.hf_config.dtype.itemsize + 3 * fp32) + batch_size * 4 * fp32
                workspace = max(workspace, vocab * verify_bytes)
                if not self.enforce_eager:
                    # 草稿图的词表概率、logits 和采样临时显存在重放后还会保留。
                    workspace += batch_size * vocab * (4 * fp32 + config.draft_hf_config.dtype.itemsize * 2)
        shapes = []
        block_bytes = 0
        for _, hf in models:
            heads = hf.num_key_value_heads // self.world_size
            head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
            shape = (hf.num_hidden_layers, self.block_size, heads, head_dim)
            shapes.append(shape)
            block_bytes += 2 * prod(shape) * hf.dtype.itemsize
        available_blocks = int(total * config.gpu_memory_utilization - used - workspace) // block_bytes
        if self.world_size > 1:
            blocks = torch.tensor(available_blocks, dtype=torch.long, device="cuda")
            dist.all_reduce(blocks, op=dist.ReduceOp.MIN)
            available_blocks = blocks.item()
        requested_blocks = config.num_kvcache_blocks
        config.num_kvcache_blocks = min(available_blocks, requested_blocks) if requested_blocks > 0 else available_blocks
        if config.num_kvcache_blocks <= 0:
            raise ValueError("Insufficient free GPU memory for KV cache; select a free GPU or adjust the memory budget")
        self.kv_caches = []
        for (model, hf), (layers, size, heads, head_dim) in zip(models, shapes):
            # 两个模型共用块号和回收时机，各层的 K、V 存在各自的显存里。
            cache = torch.empty(2, layers, config.num_kvcache_blocks, size, heads, head_dim, dtype=hf.dtype)
            self.kv_caches.append(cache)
            if model is self.model:
                self.kv_cache = cache
            layer_id = 0
            for module in model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = cache[0, layer_id]
                    module.v_cache = cache[1, layer_id]
                    layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, False, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence], is_draft: bool = False):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        if is_draft:
            graph_seqs = []
            cu_seqlens_q = [0]
            cu_seqlens_k = [0]
            max_seqlen_q = 0
            max_seqlen_k = 0
            for seq in seqs:
                start = len(seq) - 1
                tokens = [seq.last_token] + seq.draft_tokens
                q_len = len(tokens)
                k_len = start + q_len

                input_ids.extend(tokens)
                positions.extend(range(start, start + q_len))

                if self.enforce_eager:
                    cu_seqlens_q.append(cu_seqlens_q[-1] + q_len)
                    cu_seqlens_k.append(cu_seqlens_k[-1] + k_len)
                    max_seqlen_q = max(max_seqlen_q, q_len)
                    max_seqlen_k = max(max_seqlen_k, k_len)
                else:
                    # 每个位置占 decode 图的一行，共用请求的块表，只读到自己的位置。
                    graph_seqs.extend([seq] * q_len)
                    context_lens.extend(range(start + 1, k_len + 1))

                for pos in range(start, start + q_len):
                    block_idx = pos // self.block_size
                    offset = pos % self.block_size
                    block_id = seq.block_table[block_idx]
                    slot_mapping.append(block_id * self.block_size + offset)
            input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            if self.enforce_eager:
                cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
                cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
                block_tables = self.prepare_block_tables(seqs)
                set_context(True, True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                            slot_mapping=slot_mapping, block_tables=block_tables)
            else:
                size = self.graph_size(len(graph_seqs))
                if size not in self.graphs:
                    self.capture_cudagraph([size])
                context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
                block_tables = self.prepare_block_tables(graph_seqs)
                set_context(False, True, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
            return input_ids, positions
        else:
            for seq in seqs:
                input_ids.append(seq.last_token)
                positions.append(len(seq) - 1)
                context_lens.append(len(seq))
                # 动态送验可能把整段草稿裁空，预留的末块不一定是当前 token 所在的块。
                block, offset = divmod(len(seq) - 1, self.block_size)
                slot_mapping.append(seq.block_table[block] * self.block_size + offset)
            input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            block_tables = self.prepare_block_tables(seqs)
            set_context(False, False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
            return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence], is_draft: bool = False):
        if is_draft:
            # 每条请求有 1 + 实际草稿数 行 logits，空草稿也有末 token 的一行。
            temperatures = [seq.temperature for seq in seqs for _ in range(1 + len(seq.draft_tokens))]
        else:
            temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    def graph_size(self, num_tokens: int) -> int:
        if num_tokens <= self.graph_bs[-1]:
            return next(size for size in self.graph_bs if size >= num_tokens)
        # 草稿展开后可能超过最大请求数，较大的图按需捕获，多个相近长度共用一张。
        return min(1 << (num_tokens - 1).bit_length(), self.config.max_num_batched_tokens)

    @torch.inference_mode()
    def profile_verify(self):
        config = self.config
        width = min(1 + config.num_draft, config.max_model_len)
        limit = min(128, config.max_num_batched_tokens, config.max_num_seqs * width)
        if limit == 1:
            return 1., 0.
        # 在当前模型和硬件上量出固定耗时和每多验一行的成本。只在启动时测两档。
        times = []
        prefix = min(128, self.block_size, config.max_model_len - width + 1)
        self.kv_cache[:, :, 0].zero_()
        for rows in (1, limit):
            seqs = []
            for offset in range(0, rows, width):
                seq = Sequence([0] * prefix)
                seq.draft_tokens = [0] * (min(width, rows - offset) - 1)
                seq.block_table = [0] * ((prefix + len(seq.draft_tokens) - 1) // self.block_size + 1)
                seqs.append(seq)
            ids, pos = self.prepare_decode(seqs, True)
            get_context().slot_mapping.fill_(-1)
            temperatures = torch.ones(rows, dtype=torch.float32, device=ids.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for i in range(7):
                if i == 2:  # 前两次预热，不计入耗时。
                    start.record()
                logits = self.run_model(ids, pos, False)
                if logits is not None:
                    Sampler.probabilities(logits, temperatures)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) / 5)
        reset_context()
        row_ms = max(0., (times[1] - times[0]) / (limit - 1))
        return max(0., times[0] - row_ms), row_ms

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        context = get_context()
        bs = input_ids.size(0)
        if is_prefill or self.enforce_eager or (not context.is_draft and bs > self.graph_bs[-1]):
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            graph, graph_vars = self.graphs[self.graph_size(bs)]
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool, is_draft: bool = False, target_sampler=None) -> list[int] | list[list[int]] | None:
        try:
            draft_probs = None
            draft_rows = None
            if self.draft is not None and not is_prefill:
                temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
                tables = self.prepare_block_tables(seqs)
                if target_sampler is not None:
                    # 逐 token 对照时让单卡和 D 猜同一段，送验宽度也不依赖两张卡的启动测速。
                    torch.cuda.manual_seed(sum(len(seq) for seq in seqs))
                dynamic = is_draft and self.verify_cost is not None and target_sampler is None
                if dynamic and self.config.draft_early_stop:
                    draft_probs, draft_rows = self.draft.gen(seqs, tables, temperatures, self.verify_cost)
                else:
                    draft_probs = self.draft.gen(seqs, tables, temperatures)
                    if dynamic:
                        draft_rows = self.draft.select_for_verify(seqs, draft_probs, self.verify_cost, tables.device)
                if draft_rows == []:  # 所有请求都只验锚点，直接走普通 decode。
                    is_draft = False
                    draft_probs = None
            input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs, is_draft)
            if self.draft is not None and is_prefill:
                # prompt 的每个分块也给小模型过一遍，后续可直接从末 token 开始猜。
                self.draft.model(input_ids, positions)
            temperatures = self.prepare_sample(seqs, is_draft) if self.rank == 0 else None
            logits = self.run_model(input_ids, positions, is_prefill)
            if self.rank != 0:
                return None
            if draft_probs is not None and target_sampler is None:
                probs = Sampler.probabilities(logits, temperatures)
                drafts = [seq.draft_tokens for seq in seqs]
                if draft_rows is not None:
                    return self.draft.verify(drafts, probs, draft_probs, draft_rows)
                return self.draft.verify(drafts, probs, draft_probs)
            # 对照模式用主模型的 greedy token 验同一批草稿；正常 modeldraft 走上面的 p/q 拒绝采样。
            sample = self.sampler if target_sampler is None else target_sampler
            token_ids = sample(logits, temperatures).tolist()
            if not is_draft:
                return token_ids

            results = []
            start = 0
            for seq in seqs:
                end = start + len(seq.draft_tokens) + 1
                target_tokens = token_ids[start:end]
                accepted = 0
                for draft, target in zip(seq.draft_tokens, target_tokens):
                    if draft != target:
                        break
                    accepted += 1
                # 多留一位：首次猜错时是纠正 token，全部猜中时是额外生成的 token。
                results.append(target_tokens[:accepted + 1])
                start = end
            return results
        finally:
            reset_context()

    @torch.inference_mode()
    def capture_cudagraph(self, sizes: list[int] | None = None):
        config = self.config
        if sizes is None:
            max_bs = min(config.max_num_seqs, 512)
            self.graph_bs = sorted({bs for bs in [1, 2, 4, 8] + list(range(16, max_bs + 1, 16)) if bs <= max_bs} | {max_bs})
            self.graphs = {}
            self.graph_pool = None
            sizes = self.graph_bs
        max_bs = max(sizes)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        device = self.kv_cache.device
        input_ids = torch.zeros(max_bs, dtype=torch.int64, device=device)
        positions = torch.zeros_like(input_ids)
        # 运行中补图时已有请求在用 KV，预热和捕获都跳过写入。
        slot_mapping = torch.full((max_bs,), -1, dtype=torch.int32, device=device)
        context_lens = torch.zeros_like(slot_mapping)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=device)
        outputs = self.kv_cache.new_empty((max_bs, config.hf_config.hidden_size))
        graph_vars = dict(input_ids=input_ids, positions=positions, slot_mapping=slot_mapping,
                          context_lens=context_lens, block_tables=block_tables, outputs=outputs)

        for bs in reversed(sizes):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph, graph_vars
            torch.cuda.synchronize()
            reset_context()

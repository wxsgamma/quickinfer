from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from itertools import accumulate

import torch
import torch.distributed as dist

from quickinfer.layers.sampler import Sampler
from quickinfer.utils.context import set_context, reset_context

@dataclass
class Draft:
    method:str 
    def gen():
        pass
@dataclass
class N_gram(Draft):
    method: str = "ngram"
    n_min:int=2
    n_max:int=4
    k:int=4
    
    def gen(self, token_ids: list[int]) -> list[int]:
        N = len(token_ids)

        if not (1 <= self.n_min <= self.n_max) or self.k <= 0:
            raise ValueError("Invalid n-gram parameters")

        max_n = min(self.n_max, N - 1)
        if max_n < self.n_min:
            return []

        # 1. 反转 token 序列
        rev = token_ids[::-1]
        pattern = rev[:max_n]

        # 2. 构造 KMP 的 LPS 数组
        lps = [0] * max_n
        j = 0

        for i in range(1, max_n):
            while j > 0 and pattern[i] != pattern[j]:
                j = lps[j - 1]

            if pattern[i] == pattern[j]:
                j += 1

            lps[i] = j

        # 3. 使用 KMP 搜索
        matched = 0
        best_len = 0
        best_end = -1

        for i in range(1, N):
            while matched > 0 and rev[i] != pattern[matched]:
                matched = lps[matched - 1]

            if rev[i] == pattern[matched]:
                matched += 1

            # 记录最长匹配
            if matched >= self.n_min and matched >= best_len:
                best_len = matched
                best_end = i

            # 达到最大匹配长度，利用 LPS 回退
            if matched == max_n:
                matched = lps[max_n - 1]

        if best_len < self.n_min:
            return []

        # 4. 将反转序列的匹配位置映射回原序列
        start = N - 1 - best_end + best_len

        # 5. 复制匹配片段后面的 k 个 token
        return token_ids[start:start + self.k]
    
class Model_Draft(Draft):
    def __init__(self, model, block_size: int):
        super().__init__("modeldraft")
        self.model = model
        self.block_size = block_size
        self.graphs = {}

    def gen(self, seqs, block_tables: torch.Tensor, temperatures: torch.Tensor | None, cost=None):
        if cost is not None:
            return self.gen_early(seqs, block_tables, temperatures, cost)
        if self.graphs:
            return self.gen_graph(seqs, block_tables, temperatures)
        device = block_tables.device
        counts = [seq.num_scheduled_tokens - 1 for seq in seqs]
        offsets = list(accumulate(counts, initial=0))
        max_count = max(counts)
        starts = torch.tensor([len(seq) - 1 for seq in seqs], device=device)
        tokens = torch.zeros(len(seqs), max_count + 1, dtype=torch.long, device=device)
        tokens[:, 0] = torch.tensor([seq.last_token for seq in seqs], device=device)
        draft_probs = None

        for step in range(max_count + 1):
            # 第 0 步吃已确认的末 token，最后一步把最后一个草稿也写入小模型的 KV。
            active = [i for i, count in enumerate(counts) if step <= count]
            positions = starts[active] + step
            tables = block_tables[active]
            blocks = tables.gather(1, (positions // self.block_size).unsqueeze(1)).squeeze(1)
            slots = blocks * self.block_size + positions % self.block_size
            set_context(False, slot_mapping=slots.int(), context_lens=(positions + 1).int(), block_tables=tables)
            hidden = self.model(tokens[active, step], positions)

            rows = [row for row, i in enumerate(active) if step < counts[i]]
            if not rows:
                break
            requests = [active[row] for row in rows]
            logits = self.model.compute_logits(hidden[rows])
            if logits is not None:  # TP 只在 rank 0 上保留完整词表和草稿概率。
                probs = Sampler.probabilities(logits, temperatures[requests])
                if draft_probs is None:
                    draft_probs = probs.new_empty((offsets[-1], probs.size(1)))
                draft_probs[[offsets[i] + step for i in requests]] = probs
                next_ids = Sampler.sample(probs)
            else:
                next_ids = torch.empty(len(rows), dtype=torch.long, device=device)
            if dist.get_world_size() > 1:
                dist.broadcast(next_ids, 0)
            tokens[requests, step + 1] = next_ids

        if max_count:
            for seq, row, count in zip(seqs, tokens[:, 1:].tolist(), counts):
                seq.draft_tokens = row[:count]
        return draft_probs

    def graph_size(self, n):
        return min(1 << (n - 1).bit_length(), self.graph_state.size(0))

    def forward_step(self, state, tables, temperatures, sample, out=None):
        # 剩 0 枚还要写最后一个 token，负数是填充行，不能写 KV。
        active = state[:, 2] >= 0
        pos = torch.where(active, state[:, 1], 0)
        blocks = tables.gather(1, (pos // self.block_size).unsqueeze(1)).squeeze(1)
        slots = torch.where(active, blocks * self.block_size + pos % self.block_size, -1)
        lens = torch.where(active, pos + 1, 0)
        set_context(False, slot_mapping=slots.int(), context_lens=lens.int(), block_tables=tables)
        hidden = self.model(state[:, 0], pos)
        if not sample:
            return
        logits = self.model.compute_logits(hidden)
        probs = None
        if logits is not None:
            probs = Sampler.probabilities(logits, temperatures)
            if out is not None:
                probs = out.copy_(probs)
            next_ids = Sampler.sample(probs)
        else:
            next_ids = torch.empty_like(pos)
        if dist.get_world_size() > 1:
            dist.broadcast(next_ids, 0)
        state[:, 0].copy_(next_ids)
        state[:, 1].add_(1)
        state[:, 2].sub_(1)
        return probs

    @torch.inference_mode()
    def capture_cudagraph(self, config):
        max_bs = min(config.max_num_seqs, config.max_num_batched_tokens)
        max_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        device = self.model.lm_head.weight.device
        # 每行依次是当前 token、位置、剩余草稿数、该请求在验证概率里的起点。
        self.graph_state = torch.zeros(max_bs, 4, dtype=torch.long, device=device)
        self.graph_state[:, 2].fill_(-1)
        self.graph_tables = torch.zeros(max_bs, max_blocks, dtype=torch.int32, device=device)
        rank = dist.get_rank()
        if rank == 0:
            self.graph_temperatures = torch.ones(max_bs, dtype=torch.float32, device=device)
            self.graph_probs = torch.empty(max_bs, config.draft_hf_config.vocab_size, dtype=torch.float32, device=device)

        def forward(bs, sample):
            temperatures = self.graph_temperatures[:bs] if rank == 0 else None
            probs = self.graph_probs[:bs] if rank == 0 else None
            self.forward_step(self.graph_state[:bs], self.graph_tables[:bs], temperatures, sample, probs)

        sizes = {min(1 << i, max_bs) for i in range((max_bs - 1).bit_length() + 1)}
        pool = None
        # 预热、捕获都用空行，既不写请求的 KV，也不消耗正式生成的随机数。
        with torch.random.fork_rng(devices=[device]):
            for bs in sorted(sizes, reverse=True):
                for sample in (True, False):
                    graph = torch.cuda.CUDAGraph()
                    forward(bs, sample)
                    with torch.cuda.graph(graph, pool):
                        forward(bs, sample)
                    if pool is None:
                        pool = graph.pool()
                    self.graphs[bs, sample] = graph
                    torch.cuda.synchronize(device)
                    reset_context()

    def prepare_inputs(self, seqs, counts, offsets, block_tables, temperatures):
        # 长草稿放前面，每一步只重放仍要写 KV 的那部分。offset 保留原请求顺序。
        order = sorted(range(len(seqs)), key=counts.__getitem__, reverse=True)
        state = torch.tensor([(seqs[i].last_token, len(seqs[i]) - 1, counts[i], offsets[i]) for i in order],
                             dtype=torch.long, device="cpu", pin_memory=block_tables.is_cuda)
        active = len(seqs)
        if order != list(range(active)):
            indices = torch.tensor(order, dtype=torch.long, device=block_tables.device)
            block_tables = block_tables.index_select(0, indices)
            if temperatures is not None:
                temperatures = temperatures.index_select(0, indices)
        if not self.graphs:
            # 提前停会原地压紧活跃行，保留调用方的块表和温度。
            return order, state.to(block_tables.device, non_blocking=True), block_tables.clone(), (
                temperatures.clone() if temperatures is not None else None)
        self.graph_state[:active].copy_(state, non_blocking=True)
        self.graph_state[active:self.graph_size(active), 2].fill_(-1)
        tables = self.graph_tables[:, :block_tables.size(1)]
        tables[:active].copy_(block_tables)
        if temperatures is not None:
            self.graph_temperatures[:active].copy_(temperatures)
            temperatures = self.graph_temperatures
        return order, self.graph_state, tables, temperatures

    def gen_graph(self, seqs, block_tables, temperatures):
        counts = [seq.num_scheduled_tokens - 1 for seq in seqs]
        offsets = list(accumulate(counts, initial=0))
        order, _, _, _ = self.prepare_inputs(seqs, counts, offsets, block_tables, temperatures)
        active = len(seqs)
        draft_probs = None
        if temperatures is not None and offsets[-1]:
            draft_probs = self.graph_probs.new_empty((offsets[-1], self.graph_probs.size(1)))
        tokens = self.graph_state.new_empty((offsets[-1],))

        for step in range(counts[order[0]]):
            self.graphs[self.graph_size(active), True].replay()
            while counts[order[active - 1]] == step:
                active -= 1
            rows = self.graph_state[:active, 3] + step
            tokens.index_copy_(0, rows, self.graph_state[:active, 0])
            if draft_probs is not None:
                draft_probs.index_copy_(0, rows, self.graph_probs[:active])
        # 最后一次只补 KV，全批草稿为 0 时也会写入已确认的末 token。
        self.graphs[self.graph_size(active), False].replay()
        if offsets[-1]:
            sampled = tokens.tolist()
            for i, seq in enumerate(seqs):
                seq.draft_tokens = sampled[offsets[i]:offsets[i + 1]]
        return draft_probs

    def gen_early(self, seqs, block_tables, temperatures, cost):
        limits = [seq.num_scheduled_tokens - 1 for seq in seqs]
        offsets = list(accumulate(limits, initial=0))
        active, state, tables, temps = self.prepare_inputs(seqs, limits, offsets, block_tables, temperatures)
        tokens = state.new_empty((offsets[-1],))
        draft_probs = None
        rows_by_seq = [[] for _ in seqs]
        survival = [1.] * len(seqs)
        rows = len(seqs)
        expected = float(rows)
        offset = 0
        for step in range(max(limits) + 1):
            n = len(active)
            eligible = sum(limits[i] > step for i in active)
            if self.graphs:
                size = self.graph_size(n)
                if size > n:  # 被裁掉的请求即使落在图的填充区，也不能再读写它的 KV。
                    state[n:size, 2].fill_(-1)
                self.graphs[size, bool(eligible)].replay()
                probs = self.graph_probs[:eligible] if temps is not None and eligible else None
            else:
                probs = self.forward_step(state[:n], tables[:n], temps[:n] if temps is not None else None,
                                          bool(eligible))
            if not eligible:
                break  # 这一遍已经把最后一枚保留的草稿写入 KV。

            if probs is not None:
                # 当前输入的 KV 已经写好。只看采样前的分布，决定是否保留下一枚草稿。
                confidence = probs[:eligible].amax(-1).tolist()
                scores = [survival[i] * c for i, c in zip(active, confidence)]
                flags = self.verify_lengths([1] * eligible, scores, cost, rows, expected)
            if dist.get_world_size() > 1:
                mask = (torch.tensor(flags, dtype=torch.bool, device=state.device) if probs is not None
                        else torch.empty(eligible, dtype=torch.bool, device=state.device))
                dist.broadcast(mask, 0)
                if probs is None:
                    flags = mask.tolist()
            keep = [row for row, flag in enumerate(flags) if flag]
            if not keep:
                break
            if probs is not None:
                if draft_probs is None:
                    draft_probs = probs.new_empty((offsets[-1], probs.size(1)))
                draft_probs[offset:offset + eligible].copy_(probs[:eligible])
                for row in keep:
                    survival[active[row]] = scores[row]
                    expected += scores[row]
            rows += len(keep)
            tokens[offset:offset + eligible].copy_(state[:eligible, 0])
            for row in keep:
                rows_by_seq[active[row]].append(offset + row)
            offset += eligible
            active = [active[row] for row in keep]
            if keep[-1] + 1 != len(keep):
                # 继续的请求收拢到前面，下一步只按剩下的条数选图。
                idx = torch.tensor(keep, device=state.device)
                state[:len(keep)].copy_(state.index_select(0, idx))
                tables[:len(keep)].copy_(tables.index_select(0, idx))
                if temps is not None:
                    temps[:len(keep)].copy_(temps.index_select(0, idx))

        # q 按生成步存放，验证按请求读。只重排小的 token ID，词表概率用行号索引。
        draft_rows = [row for request in rows_by_seq for row in request]
        sampled = tokens[draft_rows].tolist() if draft_rows else []
        start = 0
        for seq, request in zip(seqs, rows_by_seq):
            end = start + len(request)
            seq.draft_tokens = sampled[start:end]
            seq.num_scheduled_tokens = 1 + len(request)
            start = end
        if draft_rows and draft_rows == list(range(len(draft_rows))):
            draft_rows = None
        return draft_probs, draft_rows

    @staticmethod
    def verify_lengths(counts, confidence, cost, rows=None, expected=None):
        offsets = list(accumulate(counts, initial=0))
        lengths = [0] * len(counts)
        # 每次只看各请求的下一枚，不能用后面的概率反过来决定前一枚送不送。
        candidates = [(-confidence[offsets[i]], i) for i, count in enumerate(counts) if count]
        heapify(candidates)
        if rows is None:  # 从头选时每条请求先占一行，否则接着上一草稿步的负载算。
            rows = len(counts)
            expected = float(rows)
        base_ms, row_ms = cost
        while candidates:
            negative, i = heappop(candidates)
            survival = -negative
            # 多验一行后，预计提交的 token / 验证耗时要比当前高。
            if survival * (base_ms + rows * row_ms) <= expected * row_ms:
                break
            lengths[i] += 1
            rows += 1
            expected += survival
            if lengths[i] < counts[i]:
                next_score = survival * confidence[offsets[i] + lengths[i]]
                heappush(candidates, (-next_score, i))
        return lengths

    def select_for_verify(self, seqs, draft_probs, cost, device):
        counts = [len(seq.draft_tokens) for seq in seqs]
        if not any(counts):
            return None
        if draft_probs is not None:  # 只有 rank 0 有完整的草稿词表。
            # 用采样前分布的最大概率估计置信度，避免偏向本次恰好抽中的 token。
            confidence = draft_probs.amax(dim=-1).tolist()
            lengths = self.verify_lengths(counts, confidence, cost)
        if dist.get_world_size() > 1:
            if dist.get_rank() == 0:
                sizes = torch.tensor(lengths, dtype=torch.int32, device=device)
            else:
                sizes = torch.empty(len(seqs), dtype=torch.int32, device=device)
            dist.broadcast(sizes, 0)
            lengths = sizes.tolist()
        if lengths == counts:
            return None
        # q 仍按原草稿顺序存放，只记留下的行号，避免整张词表重新拷贝。
        draft_rows = []
        offset = 0
        for seq, count, length in zip(seqs, counts, lengths):
            draft_rows.extend(range(offset, offset + length))
            offset += count
            del seq.draft_tokens[length:]
            seq.num_scheduled_tokens = 1 + length
        return draft_rows

    @staticmethod
    def verify(drafts: list[list[int]], target_probs: torch.Tensor, draft_probs: torch.Tensor, draft_rows=None):
        offsets = list(accumulate(map(len, drafts), initial=0))
        token_ids = [token for draft in drafts for token in draft]
        # p 每个请求比 q 多一行，最后一行用于全接受时再采一个 token。
        target_rows = [i + row for i, draft in enumerate(drafts)
                       for row in range(offsets[i], offsets[i + 1])]
        p = target_probs[target_rows, token_ids]
        q = draft_probs[range(len(token_ids)) if draft_rows is None else draft_rows, token_ids]
        accepts = (torch.rand_like(q) * q < p).tolist()

        accepted = []
        next_rows = []
        rejected_seqs = []
        rejected_rows = []
        for i, draft in enumerate(drafts):
            flags = accepts[offsets[i]:offsets[i + 1]]
            count = flags.index(False) if False in flags else len(draft)
            accepted.append(draft[:count])
            next_rows.append(i + offsets[i] + count)
            if count < len(draft):
                rejected_seqs.append(i)
                rejected_rows.append(offsets[i] + count)

        weights = target_probs[next_rows]
        if rejected_seqs:
            # 第一次拒绝从 (p-q) 的正数部分补采，后面的草稿全部丢弃。
            p = weights[rejected_seqs]
            if draft_rows is not None:
                rejected_rows = [draft_rows[row] for row in rejected_rows]
            residual = (p - draft_probs[rejected_rows]).clamp_min_(0)
            weights[rejected_seqs] = torch.where(residual.amax(1, keepdim=True) > 0, residual, p)
        for result, token in zip(accepted, Sampler.sample(weights).tolist()):
            result.append(token)
        return accepted

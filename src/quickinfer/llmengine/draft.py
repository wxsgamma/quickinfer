from dataclasses import dataclass
from itertools import accumulate

import torch
import torch.distributed as dist

from quickinfer.layers.sampler import Sampler
from quickinfer.utils.context import set_context

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

    def gen(self, seqs, block_tables: torch.Tensor, temperatures: torch.Tensor | None):
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

    @staticmethod
    def verify(drafts: list[list[int]], target_probs: torch.Tensor, draft_probs: torch.Tensor):
        offsets = list(accumulate(map(len, drafts), initial=0))
        token_ids = [token for draft in drafts for token in draft]
        # p 每个请求比 q 多一行，最后一行用于全接受时再采一个 token。
        target_rows = [i + row for i, draft in enumerate(drafts)
                       for row in range(offsets[i], offsets[i + 1])]
        p = target_probs[target_rows, token_ids]
        q = draft_probs[range(len(token_ids)), token_ids]
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
            residual = (p - draft_probs[rejected_rows]).clamp_min_(0)
            weights[rejected_seqs] = torch.where(residual.amax(1, keepdim=True) > 0, residual, p)
        for result, token in zip(accepted, Sampler.sample(weights).tolist()):
            result.append(token)
        return accepted

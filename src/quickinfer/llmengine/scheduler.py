from collections import deque

from quickinfer.llmengine.modelconfig import ModelConfig as Config
from quickinfer.llmengine.sequence import Sequence, SequenceStatus
from quickinfer.llmengine.block_manager import BlockManager
from quickinfer.llmengine.draft import N_gram


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.draft = None
        if getattr(config, "enabled_draft", False) and config.num_draft > 0:
            self.draft = N_gram(method=config.draft_method, k=config.num_draft)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.is_prefill = False
                seq.draft_tokens.clear()
                if self.draft is not None:
                    # 末 token 先占一行，剩下的位置才给草稿。
                    room = min(remaining, seq.max_tokens - seq.num_completion_tokens,
                               self.block_manager.append_capacity(seq)) - 1
                    if room > 0:
                        # token_ids 包含 prompt 和已经确认的输出，分块 prefill 也共用这份历史。
                        seq.draft_tokens = self.draft.gen(seq.token_ids)[:room]
                seq.num_scheduled_tokens = 1 + len(seq.draft_tokens)
                # 按本轮实际草稿数补齐块，验证时可以沿 block_table 跨块写 KV。
                self.block_manager.may_append(seq, seq.num_scheduled_tokens)
                num_batched_tokens += seq.num_scheduled_tokens
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int] | list[list[int]], is_prefill: bool) -> int:
        num_emitted = 0
        for seq, result in zip(seqs, token_ids, strict=True):
            new_tokens = [result] if isinstance(result, int) else result
            if is_prefill and seq.num_cached_tokens + seq.num_scheduled_tokens < len(seq):
                new_tokens = []
            new_tokens = new_tokens[:seq.max_tokens - seq.num_completion_tokens]
            hit_eos = not seq.ignore_eos and self.eos in new_tokens
            if hit_eos:
                new_tokens = new_tokens[:new_tokens.index(self.eos) + 1]
            seq.append_token(new_tokens)
            num_emitted += len(new_tokens)

            if not is_prefill:
                # 本轮写入有效 KV 的是旧末 token 和通过的草稿，最后一个新输出留到下轮。
                seq.num_scheduled_tokens = len(new_tokens)
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            seq.draft_tokens.clear()
            if hit_eos or seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
            elif not is_prefill:
                self.block_manager.trim_blocks(seq)
        return num_emitted

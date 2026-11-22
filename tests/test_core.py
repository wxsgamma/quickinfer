import math
import pickle
from types import SimpleNamespace
import unittest

import torch
from quickinfer import SamplingParams
from quickinfer.layers.sampler import Sampler
from quickinfer.llmengine.draft import Model_Draft
from quickinfer.llmengine.block_manager import BlockManager
from quickinfer.llmengine.scheduler import Scheduler
from quickinfer.llmengine.sequence import Sequence


class CoreTest(unittest.TestCase):
    def test_sampling_rejects_nonfinite_and_invalid_counts(self):
        for temperature in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                SamplingParams(temperature=temperature)
        for count in (0, -1, 0.5, True):
            with self.assertRaises(ValueError):
                SamplingParams(max_tokens=count)

    def test_prefix_cache_reuses_blocks_and_releases_references(self):
        manager = BlockManager(8, 256)
        a = Sequence(list(range(600)))
        manager.allocate(a, manager.can_allocate(a))
        a.num_scheduled_tokens = 600
        manager.hash_blocks(a)
        b = Sequence(list(range(512)) + [9] * 50)
        self.assertEqual(manager.can_allocate(b), 2)
        manager.allocate(b, 2)
        self.assertEqual(a.block_table[:2], b.block_table[:2])
        self.assertEqual(b.num_cached_tokens, 512)
        self.assertEqual(manager.blocks[b.block_table[0]].ref_count, 2)
        manager.deallocate(a)
        self.assertEqual(manager.blocks[b.block_table[0]].ref_count, 1)
        manager.deallocate(b)
        self.assertEqual(len(manager.free_block_ids), 8)

    def test_chunked_prefill_only_emits_after_full_prompt(self):
        scheduler = Scheduler(SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=128,
            eos=999, kvcache_block_size=256, num_kvcache_blocks=8))
        seq = Sequence(list(range(600)), SamplingParams(max_tokens=2))
        scheduler.add(seq)
        for step in range(5):
            scheduled, prefill = scheduler.schedule()
            self.assertTrue(prefill)
            self.assertLessEqual(sum(s.num_scheduled_tokens for s in scheduled), 128)
            scheduler.postprocess(scheduled, [42], prefill)
            self.assertEqual(seq.num_completion_tokens, int(step == 4))
        scheduled, prefill = scheduler.schedule()
        self.assertFalse(prefill)
        scheduler.postprocess(scheduled, [43], prefill)
        self.assertEqual(seq.completion_token_ids, [42, 43])
        self.assertTrue(scheduler.is_finished())
        self.assertEqual(len(scheduler.block_manager.free_block_ids), 8)

    def test_sequence_serialization_preserves_tp_inputs(self):
        seq = Sequence([3, 4, 5])
        seq.num_scheduled_tokens = 2
        seq.block_table = [7]
        restored = pickle.loads(pickle.dumps(seq))
        self.assertEqual(restored.token_ids, [3, 4, 5])
        self.assertEqual(restored.num_scheduled_tokens, 2)
        seq.is_prefill = False
        seq.append_token(6)
        seq.draft_tokens = [7, 8]
        restored = pickle.loads(pickle.dumps(seq))
        self.assertEqual(restored.last_token, 6)
        self.assertEqual(len(restored), 4)
        self.assertEqual(restored.block_table, [7])
        self.assertEqual(restored.draft_tokens, [7, 8])

    def test_model_draft_mixed_rejection_bonus_and_empty(self):
        drafts = [[1, 2], [0, 2], [], [2]]
        q = torch.eye(3)[[1, 2, 0, 2, 2]]
        p = torch.eye(3)[[1, 0, 2, 0, 2, 1, 2, 1, 0]]
        self.assertEqual(Model_Draft.verify(drafts, p, q), [[1, 0], [0, 2, 1], [2], [1]])

    def test_model_draft_reserves_and_rolls_back_across_blocks(self):
        scheduler = Scheduler(SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=512, eos=99,
            kvcache_block_size=256, num_kvcache_blocks=3, enabled_draft=True, draft_method="modeldraft", num_draft=4))
        first = Sequence([7] * 255, SamplingParams(max_tokens=12))
        second = Sequence([8], SamplingParams(max_tokens=2))
        scheduler.add(first)
        scheduler.add(second)
        seqs, prefill = scheduler.schedule()
        scheduler.postprocess(seqs, [7, 8], prefill)
        seqs, prefill = scheduler.schedule()
        self.assertEqual([s.num_scheduled_tokens for s in seqs], [5, 1])
        self.assertEqual(first.draft_tokens, [])
        kept, extra = first.block_table
        scheduler.postprocess(seqs, [[42], [8]], prefill)
        self.assertEqual(first.num_cached_tokens, 256)
        self.assertEqual(first.block_table, [kept])
        self.assertIn(extra, scheduler.block_manager.free_block_ids)
        self.assertTrue(second.is_finished)

    def test_model_draft_preserves_target_distribution(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(71)
            n = 6000
            q = torch.tensor([0.75, 0.25, 0.0]).expand(n, -1)
            p = torch.tensor([[0.25, 0.25, 0.5], [0, 1, 0]]).repeat(n, 1)
            drafts = Sampler.sample(q).unsqueeze(1).tolist()
            results = Model_Draft.verify(drafts, p, q)
        first = torch.tensor([row[0] for row in results])
        torch.testing.assert_close(first.bincount(minlength=3) / n, p[0], rtol=0, atol=0.025)
        self.assertTrue(all(row == [2] for row in results if len(row) == 1))
        self.assertTrue(all(row == draft + [1] for draft, row in zip(drafts, results) if len(row) == 2))

    def test_draft_history_and_commit(self):
        config = SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=32, eos=99,
            kvcache_block_size=256, num_kvcache_blocks=4, enabled_draft=True, draft_method="ngram", num_draft=3)
        for result, expected in [([3, 17], [3, 17]), ([3, 99, 17], [3, 99]), ([3, 1, 2, 4], [3, 1, 2, 4])]:
            with self.subTest(result=result):
                scheduler = Scheduler(config)
                seq = Sequence([1, 2, 3, 1], SamplingParams(max_tokens=5))
                scheduler.add(seq)
                seqs, prefill = scheduler.schedule()
                scheduler.postprocess(seqs, [2], prefill)
                seqs, prefill = scheduler.schedule()
                self.assertEqual(seq.draft_tokens, [3, 1, 2])
                self.assertEqual(seq.num_scheduled_tokens, 4)
                self.assertEqual(scheduler.postprocess(seqs, [result], prefill), len(expected))
                self.assertEqual(seq.completion_token_ids, [2] + expected)
                self.assertEqual(seq.is_finished, 99 in expected or len(expected) == 4)
                self.assertEqual(seq.num_cached_tokens, 0 if seq.is_finished else len(seq) - 1)

    def test_cross_block_draft_commit_and_reuse(self):
        config = SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=2048, eos=99,
            kvcache_block_size=256, num_kvcache_blocks=7, enabled_draft=True, draft_method="ngram", num_draft=520)
        for accepted in (0, 2, 256, 520):
            with self.subTest(accepted=accepted):
                scheduler = Scheduler(config)
                manager = scheduler.block_manager
                seq = Sequence([7] * 767, SamplingParams(max_tokens=600))
                scheduler.add(seq)
                seqs, prefill = scheduler.schedule()
                scheduler.postprocess(seqs, [7], prefill)
                # 另一个请求共享前两块，自己占一块，让接下来分配的物理块不连续。
                peer = Sequence([7] * 512 + [9])
                manager.allocate(peer, manager.can_allocate(peer))
                seqs, prefill = scheduler.schedule()
                reserved = seq.block_table.copy()
                self.assertEqual(seq.num_scheduled_tokens, 521)
                self.assertEqual(len(reserved), 6)
                self.assertNotIn(peer.block_table[-1], reserved)
                self.assertFalse(manager.free_block_ids)

                scheduler.postprocess(seqs, [[7] * accepted + [8]], prefill)
                valid = 768 + accepted
                keep = (valid + 255) // 256
                released = reserved[keep:]
                self.assertEqual(seq.num_cached_tokens, valid)
                self.assertEqual(seq.block_table, reserved[:keep])
                self.assertEqual(set(manager.free_block_ids), set(released))
                self.assertEqual(set(manager.hash_to_block_id.values()), set(reserved[:valid // 256]))
                self.assertEqual(manager.blocks[peer.block_table[0]].ref_count, 2)
                if released:
                    other = Sequence([42])
                    manager.allocate(other, manager.can_allocate(other))
                    self.assertIn(other.block_table[0], released)
                    manager.deallocate(other)
                manager.deallocate(peer)
                manager.deallocate(seq)
                self.assertCountEqual(manager.free_block_ids, range(7))

    def test_short_draft_at_block_edge_stops_at_eos(self):
        config = SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=1024, eos=1,
            kvcache_block_size=256, num_kvcache_blocks=5, enabled_draft=True, draft_method="ngram", num_draft=520)
        scheduler = Scheduler(config)
        seq = Sequence([9] * 251 + [1, 2, 3, 1], SamplingParams(max_tokens=600))
        scheduler.add(seq)
        seqs, prefill = scheduler.schedule()
        scheduler.postprocess(seqs, [2], prefill)
        seqs, prefill = scheduler.schedule()
        self.assertEqual(seq.draft_tokens, [3, 1, 2])
        self.assertEqual(len(seq.block_table), 2)
        first_block = seq.block_table[0]
        scheduler.postprocess(seqs, [[3, 1, 2, 8]], prefill)
        self.assertEqual(seq.completion_token_ids, [2, 3, 1])
        self.assertTrue(seq.is_finished)
        self.assertCountEqual(scheduler.block_manager.free_block_ids, range(5))
        self.assertEqual(list(scheduler.block_manager.hash_to_block_id.values()), [first_block])

    def test_shortens_drafts_when_other_requests_own_the_blocks(self):
        config = SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=1024, eos=99,
            kvcache_block_size=256, num_kvcache_blocks=2, enabled_draft=True, draft_method="ngram", num_draft=4)
        scheduler = Scheduler(config)
        for token in (7, 8):
            scheduler.add(Sequence([token] * 254))
        seqs, prefill = scheduler.schedule()
        scheduler.postprocess(seqs, [7, 8], prefill)
        seqs, prefill = scheduler.schedule()
        self.assertFalse(prefill)
        self.assertEqual([s.draft_tokens for s in seqs], [[7], [8]])
        self.assertEqual([s.num_scheduled_tokens for s in seqs], [2, 2])
        self.assertFalse(scheduler.block_manager.free_block_ids)


if __name__ == "__main__":
    unittest.main()

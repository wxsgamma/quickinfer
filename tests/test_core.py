import math
import pickle
from types import SimpleNamespace
import unittest

from quickinfer import SamplingParams
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
        restored = pickle.loads(pickle.dumps(seq))
        self.assertEqual(restored.last_token, 6)
        self.assertEqual(len(restored), 4)
        self.assertEqual(restored.block_table, [7])


if __name__ == "__main__":
    unittest.main()

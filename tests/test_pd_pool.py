import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import torch

from quickinfer.llmengine.pd_pool import PDPool, Stage, handoff_spans, check_argmax, reference_rows
from quickinfer.llmengine.block_manager import BlockManager
from quickinfer.llmengine.sequence import Sequence
from quickinfer import SamplingParams


class PDPoolTest(unittest.TestCase):
    def test_reference_rows_stop_after_the_first_wrong_draft(self):
        seqs = [Sequence([1]), Sequence([2]), Sequence([3])]
        seqs[0].draft_tokens = [4, 9, 6]
        seqs[2].draft_tokens = [7]
        refs = [[4, 5, 6, 8], [], [7, 8]]
        self.assertEqual(reference_rows(seqs, refs, False), [4, 5, None, None, None, 7, 8])
        self.assertEqual(reference_rows(seqs, refs, True), [4, None, 7])

    def test_decode_commits_chunks_and_releases_rejected_kv(self):
        pool = PDPool.__new__(PDPool)
        pool.config = SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=512, max_model_len=1024,
            kvcache_block_size=256, eos=99, enabled_draft=True, draft_method="modeldraft", num_draft=4,
            hf_config=SimpleNamespace(vocab_size=100))
        pool.size, pool.prefill, pool.decode, pool.checking = 256, [0], [1], False
        p, d = (Stage(g, Mock(), None) for g in (0, 1))
        pool.stages = {0:p, 1:d}
        for st in (p,d):
            st.blocks, st.bytes_per_token = 4, 128
        pool.graph_counts = Mock(return_value={})
        pool.reset()
        seq = pool.add_request([7]*255, SamplingParams(max_tokens=8))
        pool.schedule_compute()
        pool.finish_batch(p, [7])
        pool.schedule_transfers()
        pool.finish_transfer(p, (1., None))
        self.assertEqual(seq.num_completion_tokens, 0)
        pool.finish_transfer(d, (1., None))
        self.assertEqual((seq.num_completion_tokens, d.reserved), (1, 2))
        pool.schedule_compute()
        self.assertEqual((seq.num_scheduled_tokens, len(seq.block_table)), (5, 2))
        pool.finish_batch(d, [[17]], [2])  # 只送验两枚，首枚就被拒，跨出去的块当轮收回。
        self.assertEqual((seq.num_cached_tokens, len(seq.block_table), d.reserved), (256, 1, 2))
        pool.schedule_compute()
        pool.finish_batch(d, [[9, 99, 5]], [2])
        self.assertEqual(seq.completion_token_ids, [7, 17, 9, 99])
        self.assertEqual((d.stats["draft_tokens_verified"], d.stats["draft_tokens_accepted"], d.stats["decode_tokens"]), (4, 2, 3))
        self.assertEqual((p.stats["verify_steps"], d.stats["verify_steps"], pool.live, d.reserved), (0, 2, 0, 0))
        self.assertTrue(all(len(st.scheduler.block_manager.free_block_ids)==4 for st in (p,d)))

    def test_reference_token_must_attain_the_maximum(self):
        scores = torch.tensor([[3., 3., 2.], [1., 4., 2.]], dtype=torch.bfloat16)
        ties = []
        self.assertEqual(check_argmax(scores, [1, None], ties).tolist(), [1, 1])
        self.assertEqual(ties, [(0, 1, 3.)])
        with self.assertRaises(ValueError):
            check_argmax(scores, [2, None], ties)

    def test_join_only_ranges_contiguous_on_both_devices(self):
        self.assertEqual(handoff_spans([7, 8, 2, 3], [5, 1, 8, 9], 780, 256),
                         [[1792, 1280, 256], [2048, 256, 256], [512, 2048, 268]])
        self.assertEqual(handoff_spans([2, 3], [5, 6], 257, 256), [[512, 1280, 257]])
        for src, dst, n in (([1], [1], 257), ([1, 2], [2], 257), ([1], [1], 0)):
            with self.assertRaises(ValueError):
                handoff_spans(src, dst, n, 256)

    def test_release_sender_leaves_shared_prefix_valid(self):
        manager = BlockManager(8, 256)
        first = Sequence([7] * 257)
        manager.allocate(first, 0)
        first.num_scheduled_tokens = 257
        manager.hash_blocks(first, True)
        second = Sequence([7] * 256 + [9])
        manager.allocate(second, manager.can_allocate(second))
        shared = first.block_table[0]
        held, first.block_table = first.block_table, []
        manager.release_blocks(held)
        self.assertEqual(manager.blocks[shared].ref_count, 1)
        self.assertEqual(second.block_table[0], shared)
        manager.deallocate(second)
        self.assertFalse(manager.used_block_ids)


if __name__ == "__main__":
    unittest.main()

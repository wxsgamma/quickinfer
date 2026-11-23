import unittest

from benchmark.pd_metrics import report, summarize


def record(key, arrival, events, kind="chat"):
    return {"id": key, "class": kind, "arrival_us": arrival, "admit_us": arrival + 1000,
            "dispatch_us": arrival + 2000, "input_tokens": 10, "output_tokens": events[-1][1], "emits": events}


def trial(records, repeat=0, rate=4.):
    return {"workload": "arena", "mode": "pd", "rate": rate, "repeat": repeat, "requests": records,
            "gpu_count": 2, "seconds": max(r["emits"][-1][0] for r in records) / 1e6,
            "lanes": [], "gpu": {}, "nvlink_bytes": {}, "samples": 0}


class PDMetricsTest(unittest.TestCase):
    def test_one_token_and_speculative_chunk(self):
        a = record("a", 0, [[100000, 1], [220000, 3]])
        b = record("b", 50000, [[60000, 1]])
        metrics = summarize([trial([a, b])], 100, 50)
        self.assertEqual(metrics["ttft_ms"]["p50"], 55)
        self.assertEqual(metrics["tpot_ms"]["n"], 1)
        self.assertEqual(metrics["tpot_ms"]["p50"], 60)
        self.assertEqual(metrics["inter_emit_ms"]["p99"], 120)
        self.assertEqual(metrics["slo_hits"], 1)
        self.assertAlmostEqual(metrics["goodput_req_s_per_gpu"], 1 / .22 / 2)
        self.assertAlmostEqual(metrics["output_tok_s"], 4 / .22)

    def test_capacity_requires_all_repeats_and_joint_slo(self):
        fast = record("a", 0, [[10000, 1], [20000, 2]])
        slow_ttft = record("b", 0, [[200000, 1], [205000, 2]])
        slow_tpot = record("c", 0, [[10000, 1], [90000, 2]])
        runs = [trial([fast]), trial([fast], rate=12.), trial([slow_ttft, slow_tpot], repeat=1, rate=12.)]
        result = report(runs, 100, 50, .9)
        self.assertEqual(result["highest_tested_rps_meeting_slo"]["arena"]["pd"], 4.)
        self.assertEqual(result["cells"][1]["slo_hits"], 1)
        self.assertEqual(result["cells"][1]["ttft_ms"]["n"], 3)

    def test_streaming_slo_uses_actual_pause_between_commits(self):
        smooth = record("a", 0, [[10000, 1], [100000, 3], [190000, 5]])
        stalled = record("b", 0, [[10000, 1], [160000, 4], [190000, 5]])
        single = record("c", 0, [[10000, 1]])
        runs = [trial([smooth, stalled, single])]
        self.assertEqual(report(runs, 500, 50, .9)["cells"][0]["slo_hits"], 3)
        result = report(runs, 500, 50, .9, max_gap_ms=100)
        self.assertEqual(result["cells"][0]["slo_hits"], 2)
        self.assertEqual(result["slo"]["max_gap_ms"], 100)
        self.assertIsNone(result["highest_tested_rps_meeting_slo"]["arena"]["pd"])

    def test_probe_stall_includes_gap_spanning_long_prefill(self):
        probe = record("p", 0, [[10000, 1], [30000, 2], [180000, 3], [200000, 4]], "decode-probe")
        late = record("q", 0, [[140000, 1], [150000, 2]], "decode-probe")
        long = record("l", 50000, [[140000, 1], [150000, 2]], "long-burst")
        gaps = summarize([trial([probe, late, long])], 500, 50)["active_probe_max_gap_during_burst_ms"]
        self.assertEqual((gaps["n"], gaps["p95"]), (1, 150))

    def test_rejects_non_monotonic_emissions(self):
        bad = record("x", 0, [[10000, 1], [20000, 3], [18000, 2]])
        with self.assertRaises(ValueError):
            summarize([trial([bad])], 500, 50)


if __name__ == "__main__":
    unittest.main()

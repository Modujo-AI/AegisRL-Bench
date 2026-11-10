import unittest

from aegisrl_bench.resilience import pressure_intervals


class PressureEvidenceTests(unittest.TestCase):
    def test_clips_both_boundaries(self):
        result = pressure_intervals(
            [
                {"time": 5, "phase": "full"},
                {"time": 15, "phase": "released"},
                {"time": 25, "phase": "full"},
                {"time": 40, "phase": "released"},
            ],
            10,
            30,
        )
        self.assertEqual(result["full_seconds"], 10)
        self.assertEqual(result["full_fraction"], 0.5)
        self.assertTrue(result["intervals"][0]["left_censored"])
        self.assertTrue(result["intervals"][1]["right_censored"])

    def test_unreleased_failure_is_censored_not_recovered(self):
        result = pressure_intervals([{"time": 20, "phase": "full"}], 10, 30)
        self.assertEqual(result["full_seconds"], 10)
        self.assertEqual(result["right_censored_intervals"], 1)

    def test_outside_window_has_no_exposure(self):
        events = [
            {"time": 1, "phase": "full"},
            {"time": 5, "phase": "released"},
            {"time": 40, "phase": "full"},
        ]
        self.assertEqual(pressure_intervals(events, 10, 30)["full_seconds"], 0)

    def test_rejects_incomplete_or_invalid_event_stream(self):
        bad = [
            [{"time": 12, "phase": "released"}],
            [{"time": 12, "phase": "full"}, {"time": 13, "phase": "full"}],
            [{"time": 12, "phase": "full"}, {"time": 11, "phase": "released"}],
            [{"time": float("nan"), "phase": "full"}],
            [{"time": 12, "phase": "unknown"}],
        ]
        for events in bad:
            with self.subTest(events=events), self.assertRaises(ValueError):
                pressure_intervals(events, 10, 30)
        with self.assertRaises(ValueError):
            pressure_intervals([], 10, 10)

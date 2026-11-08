import json
import tempfile
import unittest
from pathlib import Path
from aegisrl_bench.validation import eligible, prediction_counts, validate_submission


class ValidationTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / ".state/tests"
        root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=root)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reject_external_symlink(self):
        work = self.root / "workspace"
        model = work / "output/v1"
        model.mkdir(parents=True)
        external = self.root / "private"
        external.mkdir()
        (external / "model.bin").write_bytes(b"private")
        (model / "model.bin").symlink_to(external / "model.bin")
        with self.assertRaises(ValueError):
            validate_submission(model, work)

    def test_reject_base_model_submission_path(self):
        work = self.root / "workspace"
        base = work / "models/base"
        base.mkdir(parents=True)
        with self.assertRaises(ValueError):
            validate_submission(base, work)

    def test_accept_project_model(self):
        work = self.root / "workspace"
        model = work / "output/v1"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        self.assertEqual(validate_submission(model, work), model)

    def test_deadline_and_failure_rules(self):
        row = {"score": 65.5, "timestamp": "1970-01-01T00:01:40"}
        self.assertTrue(eligible(row, 0, 100))
        self.assertFalse(eligible(row, 0, 99))
        self.assertFalse(eligible({**row, "error": "incomplete predictions"}, 0, 100))
        self.assertFalse(eligible({**row, "score": float("nan")}, 0, 100))
        self.assertFalse(eligible({**row, "score": True}, 0, 100))

    def test_prediction_validation_uses_latest_run(self):
        for folder, count in [("20260101_000000", 5), ("20260101_000001", 1319)]:
            p = (
                self.root
                / "benchmark_results"
                / folder
                / "predictions/model/gsm8k.json"
            )
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps({str(i): {} for i in range(count)}))
        self.assertEqual(prediction_counts(self.root), [1319])

    def test_missing_predictions(self):
        self.assertEqual(prediction_counts(self.root), [])


if __name__ == "__main__":
    unittest.main()

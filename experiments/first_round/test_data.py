"""Scientific bookkeeping regressions; runnable with the Python stdlib."""
import json
import tempfile
import unittest
from pathlib import Path

from experiments.first_round.data import load_samples, numeric_answer, prosqa_answer, score, summarize
from experiments.first_round.run import load_baseline, load_gate


class DataTests(unittest.TestCase):
    def test_number_and_empty_continuation(self):
        self.assertEqual(numeric_answer("### -1,234.50"), "-1234.5")
        self.assertEqual(numeric_answer("4 * 2 = 8\n### 8"), "8")
        self.assertIsNone(numeric_answer(""))
        self.assertIsNone(numeric_answer("No answer."))

    def test_prosqa_ambiguity_and_word_boundaries(self):
        options = ["hilpus", "sterpus"]
        self.assertEqual(prosqa_answer("Sally is a sterpus.", options), "sterpus")
        self.assertIsNone(prosqa_answer("hilpus or sterpus", options))
        self.assertIsNone(prosqa_answer("notsterpus", options))

    def test_real_datasets_have_stable_ids_and_parsable_gold(self):
        root = Path(__file__).resolve().parents[3] / "data"
        if not all((root / name / filename).is_file() for name, filename in (("gsm8k", "test.jsonl"), ("prosqa", "test.json"))):
            self.skipTest("Real datasets are downloaded separately; adapter fixtures are checked below")
        for name, suffix, count in (("gsm8k", "jsonl", 1319), ("prosqa", "json", 500)):
            path = root / name / f"test.{suffix}"
            rows = load_samples(path, name, "test")
            self.assertEqual(len(rows), count)
            self.assertEqual(len({row["sample_id"] for row in rows}), count)
            self.assertEqual(rows[0]["sample_id"], load_samples(path, name, "test")[0]["sample_id"])
            self.assertFalse(score("", rows[0], name)["correct"])

    def test_adapters_without_downloaded_datasets(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)
            gsm = path / "gsm.jsonl"
            gsm.write_text(json.dumps({"question": "How much?", "answer": "Calculation\n#### 1,200.00"}) + "\n")
            rows = load_samples(gsm, "gsm8k", "test")
            self.assertEqual(rows[0]["gold_answer"], "1200")
            self.assertTrue(score("### 1200", rows[0], "gsm8k")["correct"])
            prosqa = path / "prosqa.json"
            prosqa.write_text(json.dumps([{"question": "Is Sally a hilpus or sterpus?", "answer": "Sally is a sterpus.", "steps": ["Sally is a sterpus."]}]))
            rows = load_samples(prosqa, "prosqa", "test")
            self.assertEqual(rows[0]["reference_steps"], ["Sally is a sterpus."])
            self.assertFalse(score("Sally is a hilpus.", rows[0], "prosqa")["correct"])

    def test_parse_failure_and_execution_failure_have_distinct_denominators(self):
        rows = [{"status": "ok", "correct": True, "parse_status": "ok"},
                {"status": "ok", "correct": False, "parse_status": "failed"},
                {"status": "error"}]
        result = summarize(rows)
        self.assertEqual(result["accuracy"]["denominator"], 2)
        self.assertEqual(result["accuracy"]["value"], 0.5)
        self.assertEqual(result["execution_failed"], 1)
        self.assertEqual(result["parse_failed"], 1)

    def test_gate_rejects_empty_or_failed_validation(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)
            manifest = {"stage": "resume", "status": "completed", "signature": "sig", "baseline_run_id": "base"}
            (path / "manifest.json").write_text(json.dumps(manifest))
            (path / "summary.json").write_text(json.dumps({"checks_total": 0, "checks_passed": 0}))
            with self.assertRaises(ValueError):
                load_gate(name, "sig", "base")
            (path / "summary.json").write_text(json.dumps({"checks_total": 6, "checks_passed": 5}))
            with self.assertRaises(ValueError):
                load_gate(name, "sig", "base")
            (path / "summary.json").write_text(json.dumps({"checks_total": 6, "checks_passed": 6}))
            self.assertEqual(load_gate(name, "sig", "base")["stage"], "resume")

    def test_parse_failure_is_not_treated_as_located_reasoning_error(self):
        row = {"status": "ok", "correct": True, "parse_status": "ok", "baseline_correct": False,
               "baseline_parse_status": "failed", "predicted_answer": "5", "baseline_answer": None,
               "continuation": "5", "baseline_continuation": "no answer"}
        result = summarize([row])
        self.assertEqual(result["wrong_to_correct"]["denominator"], 0)
        self.assertEqual(result["paired_baseline_parse_failed"], 1)

    def test_baseline_signature_and_coverage(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)
            (path / "manifest.json").write_text(json.dumps({"stage": "baseline", "status": "completed", "signature": "sig"}))
            (path / "predictions.jsonl").write_text(json.dumps({"sample_id": "one", "status": "ok"}) + "\n")
            with self.assertRaises(ValueError):
                load_baseline(name, "different", [{"sample_id": "one"}])
            with self.assertRaises(ValueError):
                load_baseline(name, "sig", [{"sample_id": "missing"}])


if __name__ == "__main__":
    unittest.main()

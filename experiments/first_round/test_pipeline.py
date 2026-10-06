"""End-to-end CLI test on a synthetic sample and locally constructed tiny model."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from common.models.coconut_model import CoconutWrapper

ROOT = Path(__file__).resolve().parents[2]


class PipelineTest(unittest.TestCase):
    def test_stages_gate_export_and_offline_rescoring(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base_path = root / "models/gpt2"
            base_path.mkdir(parents=True)
            vocab = {char: index for index, char in enumerate(bytes_to_unicode().values())}
            vocab["<|endoftext|>"] = 256
            (base_path / "vocab.json").write_text(json.dumps(vocab))
            (base_path / "merges.txt").write_text("#version: 0.2\n")
            GPT2Tokenizer(str(base_path / "vocab.json"), str(base_path / "merges.txt")).save_pretrained(base_path)
            torch.manual_seed(321)
            GPT2LMHeadModel(GPT2Config(vocab_size=257, n_positions=128, n_embd=16, n_layer=2, n_head=2,
                                     eos_token_id=256, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
                                     _attn_implementation="eager")).save_pretrained(base_path)
            wrapper = CoconutWrapper()
            wrapper.load_from_config({"base_model_name_or_path": str(base_path), "device": "cpu", "num_latent_placeholders": 3})
            checkpoint = root / "tiny-checkpoint"
            torch.save(wrapper.coconut_model.state_dict(), checkpoint)
            dataset = root / "synthetic.jsonl"
            dataset.write_text(json.dumps({"question": "2+3?", "answer": "#### 5"}) + "\n")
            config = {"dataset": "gsm8k", "split": "synthetic", "dataset_path": str(dataset),
                      "checkpoint_path": str(checkpoint), "base_model_path": str(base_path),
                      "prompt_template": "{question}\n", "num_latent_steps": 3, "max_new_tokens": 3,
                      "seed": 0, "max_samples": 1, "atol": 1e-5, "rtol": 1e-5,
                      "random_strengths": [0.5], "noise_seeds": [0], "save_trace_samples": 1}
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            machine = root / "machine.json"
            machine.write_text(json.dumps({"OUTPUT_DIR": str(root / "outputs"), "device": "cpu"}))
            environment = dict(os.environ)
            environment["HF_HOME"] = str(root / "hf-cache")
            environment["OMP_NUM_THREADS"] = "1"

            def execute(script, *extra, expected_code=0):
                result = subprocess.run([sys.executable, str(ROOT / script), "--config", str(config_path),
                                         "--machine-config", str(machine), *extra], cwd=ROOT, env=environment,
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, expected_code, result.stdout + result.stderr)
                return result

            execute("run_baseline.py", "--run-id", "baseline")
            baseline = root / "outputs/baseline"
            # Interventions cannot start without a passed resume check.
            denied = execute("run_interventions.py", "--baseline-run", str(baseline), expected_code=2)
            self.assertIn("--resume-check is required", denied.stderr)
            execute("check_resume.py", "--baseline-run", str(baseline), "--run-id", "resume")
            resume = root / "outputs/resume"
            summary = json.loads((resume / "summary.json").read_text())
            self.assertEqual(summary["checks_total"], 3)
            self.assertEqual(summary["checks_passed"], 3)
            execute("run_interventions.py", "--baseline-run", str(baseline), "--resume-check", str(resume), "--run-id", "interventions")
            output = root / "outputs/interventions"
            rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 9)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "completed")
            for row in rows:
                if row["operation"] == "random":
                    self.assertAlmostEqual(row["edit_relative_norm"], 0.5, places=5)
            analysis = subprocess.run([sys.executable, str(ROOT / "analyze_first_round.py"), "--run-dir", str(output),
                                       "--output-dir", str(root / "analysis")], cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(analysis.returncode, 0, analysis.stdout + analysis.stderr)
            self.assertEqual(len(json.loads((root / "analysis/by_condition.json").read_text())), 9)
            # A damaged returned result must not silently produce statistics.
            with (output / "predictions.jsonl").open("a") as handle:
                handle.write("{}\n")
            damaged = subprocess.run([sys.executable, str(ROOT / "analyze_first_round.py"), "--run-dir", str(output)], cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(damaged.returncode, 0)
            self.assertIn("modified result file", damaged.stderr)


if __name__ == "__main__":
    unittest.main()

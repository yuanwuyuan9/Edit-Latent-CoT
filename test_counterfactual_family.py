"""Check frozen edits, natural continuation, and calibration/variant accounting."""
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

from experiments.first_round.run import digest
from experiments.first_round.test_tiny_model import tiny_model
from run_counterfactual_family import delta_hash, edit_directions, evaluate_question, family_rows

ROOT = Path(__file__).resolve().parent


class CounterfactualTests(unittest.TestCase):
    def test_only_flock_size_changes_and_invalid_protocol_is_rejected(self):
        sample = {"sample_id": "original", "question": "3 cups each, 15 then 25; flock is 20 chickens.", "gold_answer": "20"}
        protocol = {"counts": [20, 16, 30], "calibration_count": 20}
        rows = family_rows(sample, protocol)
        self.assertEqual([r["answer"].split("#### ")[-1] for r in rows], ["20", "8", "50"])
        self.assertEqual([r["is_calibration"] for r in rows], [True, False, False])
        self.assertEqual(rows[1]["question"], sample["question"].replace("20 chickens", "16 chickens"))
        for counts in ([16, 20], [20, 20], [20, 1]):
            with self.assertRaises(ValueError):
                family_rows(sample, {**protocol, "counts": counts})
        with self.assertRaises(ValueError):
            family_rows({**sample, "question": "20 chickens and 20 chickens"}, protocol)

    @torch.no_grad()
    def test_equal_absolute_norm_and_natural_feedback_with_original_prefix(self):
        model = tiny_model()
        delta = torch.linspace(-1, 1, 16).view(1, 16)
        protocol = {"scale": 1.0, "random_seeds": [100, 101]}
        edits = edit_directions(delta, protocol)
        for _, _, change in edits[1:]:
            torch.testing.assert_close(change.norm(), delta.norm())
        torch.testing.assert_close(edits[2][2], -delta)
        torch.testing.assert_close(edits[3][2], edit_directions(delta, protocol)[3][2])
        cfg = {"prompt_template": "{question}\n", "num_latent_steps": 3, "max_new_tokens": 5, "atol": 1e-5, "rtol": 1e-5}
        for question in ("Hi", "Longer question"):
            outputs, base = evaluate_question(model, {"question": question}, cfg, 2, edits)
            actual = next(trace for op, _, _, trace, _ in outputs if op == "frozen_delta")
            torch.testing.assert_close(actual[:, :1], base[:, :1])
            torch.testing.assert_close(actual[:, 1] - base[:, 1], delta)
            self.assertGreater(float((actual[:, 2:] - base[:, 2:]).norm()), 0)
        with self.assertRaises(ValueError):
            edit_directions(torch.zeros_like(delta), protocol)

    def test_cli_preserves_provenance_and_detects_constant_answer_on_variants(self):
        # Deliberately constant numeric decoder: validates reporting, not repair efficacy.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base"
            base.mkdir()
            vocab = {c: i for i, c in enumerate(bytes_to_unicode().values())}
            vocab.update({"<|endoftext|>": 256, "20": 257})
            (base / "vocab.json").write_text(json.dumps(vocab))
            (base / "merges.txt").write_text("#version: 0.2\n2 0\n")
            GPT2Tokenizer(str(base / "vocab.json"), str(base / "merges.txt")).save_pretrained(base)
            torch.manual_seed(321)
            network = GPT2LMHeadModel(GPT2Config(vocab_size=258, n_positions=128, n_embd=16, n_layer=2,
                n_head=2, eos_token_id=256, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, _attn_implementation="eager"))
            with torch.no_grad():
                network.transformer.ln_f.weight.fill_(0.1)
                network.transformer.ln_f.bias.zero_()
                network.transformer.ln_f.bias[0] = 10
                network.transformer.wte.weight[257].zero_()
                network.transformer.wte.weight[257, 0] = 1
            network.save_pretrained(base)
            from common.models.coconut_model import CoconutWrapper
            wrapper = CoconutWrapper()
            wrapper.load_from_config({"base_model_name_or_path": str(base), "device": "cpu", "num_latent_placeholders": 3})
            checkpoint = root / "checkpoint"
            torch.save(wrapper.coconut_model.state_dict(), checkpoint)
            dataset = root / "data.jsonl"
            dataset.write_text(json.dumps({"question": "flock is 20 chickens", "answer": "#### 20"}) + "\n")
            config = root / "config.json"
            config.write_text(json.dumps({"dataset": "gsm8k", "split": "synthetic", "dataset_path": str(dataset),
                "checkpoint_path": str(checkpoint), "base_model_path": str(base), "prompt_template": "{question}\n",
                "num_latent_steps": 3, "max_new_tokens": 1, "seed": 0, "max_samples": 1, "atol": 1e-5,
                "rtol": 1e-5, "random_strengths": [0.5], "noise_seeds": [0], "save_trace_samples": 1}))
            machine = root / "machine.json"
            machine.write_text(json.dumps({"OUTPUT_DIR": str(root / "outputs"), "device": "cpu"}))
            env = {**os.environ, "OMP_NUM_THREADS": "1", "HF_HOME": str(root / "hf-cache")}
            def execute(script, *args):
                result = subprocess.run([sys.executable, str(ROOT / script), "--config", str(config),
                    "--machine-config", str(machine), *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            baseline, gate, source, transplant = (root / "outputs" / n for n in ("baseline", "gate", "source", "transplant"))
            execute("run_baseline.py", "--run-id", "baseline")
            execute("check_resume.py", "--run-id", "gate", "--baseline-run", str(baseline))
            execute("run_interventions.py", "--run-id", "source", "--baseline-run", str(baseline), "--resume-check", str(gate))
            execute("run_path_controls.py", "--mode", "transplant", "--steps", "1", "--sample-index", "0",
                    "--input-run", str(source), "--run-id", "transplant")
            sources = [json.loads(line) for line in (source / "predictions.jsonl").read_text().splitlines()]
            source_row = next(r for r in sources if r["operation"] == "random" and r["step"] == 1)
            transplanted = [json.loads(line) for line in (transplant / "predictions.jsonl").read_text().splitlines()]
            target_row = next(r for r in transplanted if r["step"] == 2)
            saved = torch.load(source / source_row["trace_path"], weights_only=True)
            delta = saved["modified_latents"][:, 1] - saved["baseline_latents"][:, 1]
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({"origin_sample_index": 0, "calibration_count": 20, "counts": [20, 18, 22],
                "source_step": 1, "source_strength": 0.5, "source_noise_seed": 0, "target_step": 2, "scale": 1.0,
                "random_seeds": [100, 101], "source_trace_sha256": digest(source / source_row["trace_path"]),
                "transplant_trace_sha256": digest(transplant / target_row["trace_path"]), "delta_values_sha256": delta_hash(delta)}))
            execute("run_counterfactual_family.py", "--protocol", str(protocol), "--source-run", str(source),
                    "--transplant-run", str(transplant), "--run-id", "family")
            output = root / "outputs/family"
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual((summary["status"], summary["record_count"]), ("completed", 18))
            positive = summary["by_condition"]["frozen_delta/seed=None"]
            self.assertEqual(positive["calibration"]["accuracy"]["numerator"], 1)
            self.assertEqual(positive["held_out_variants"]["accuracy"], {"numerator": 0, "denominator": 2, "value": 0.0})
            self.assertEqual(positive["held_out_variants"]["answers_equal_20"], 2)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertNotEqual(manifest["signature"], manifest["source_signature"])
            self.assertIn("run_counterfactual_family.py", manifest["identity"]["source_hashes"])
            result = subprocess.run([sys.executable, str(ROOT / "analyze_first_round.py"), "--run-dir", str(output),
                "--output-dir", str(root / "analysis")], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

"""Verify trajectory clamping and decoding against independent tiny-model rollout."""
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

from experiments.first_round.test_tiny_model import tiny_model
from run_path_controls import crossed_latents, decode_fixed_latents

ROOT = Path(__file__).resolve().parent


class PathControlTests(unittest.TestCase):
    def test_crossing_changes_exactly_the_declared_slots_without_mutation(self):
        base = torch.arange(24, dtype=torch.float32).view(1, 3, 8)
        edited = base + 100
        before = base.clone(), edited.clone()
        for step in (1, 2, 3):
            result = crossed_latents(base, edited, step)
            torch.testing.assert_close(result["baseline_replay"], base)
            torch.testing.assert_close(result["edit_only"][:, step - 1], edited[:, step - 1])
            torch.testing.assert_close(result["edit_only"][:, step:], base[:, step:])
            torch.testing.assert_close(result["suffix_only"][:, :step], base[:, :step])
            torch.testing.assert_close(result["suffix_only"][:, step:], edited[:, step:])
            torch.testing.assert_close(result["edited_replay"][:, step - 1:], edited[:, step - 1:])
            result["edit_only"].zero_()
            torch.testing.assert_close(result["baseline_replay"], base)
        torch.testing.assert_close(base, before[0])
        torch.testing.assert_close(edited, before[1])

    @torch.no_grad()
    def test_full_context_replay_matches_official_and_edited_trajectories(self):
        model = tiny_model()
        for prompt in ("Hi\n", "Longer question\n"):
            tokens = model._prepare_inputs(prompt)
            positions = (tokens["input_ids"][0] == model.latent_token_id).nonzero().flatten()
            reference = model.coconut_model(**tokens, labels=tokens["input_ids"].clone())
            base = reference.inputs_embeds[:, positions].clone()
            baseline = model.run_baseline(prompt)
            result = decode_fixed_latents(model, prompt, base)
            self.assertEqual(result["generated_token_ids"], baseline["generated_token_ids"][0])
            torch.testing.assert_close(result["first_logits"], reference.logits[:, -1, :])
            for step in (1, 2, 3):
                h, state = model.forward_until_step(prompt, step)
                edited = model.rollout_from_step(torch.zeros_like(h), state)
                trajectory = edited["inputs_embeds"][:, positions].clone()
                replay = decode_fixed_latents(model, prompt, trajectory)
                self.assertEqual(replay["generated_token_ids"], edited["generated_token_ids"][0])
                torch.testing.assert_close(replay["first_logits"], edited["logits"][:, -1, :])
                for name, latents in crossed_latents(base, trajectory, step).items():
                    output = decode_fixed_latents(model, prompt, latents)
                    torch.testing.assert_close(output["inputs_embeds"][:, positions], latents)

    def test_invalid_latents_are_rejected(self):
        model = tiny_model()
        for latents in (torch.zeros(1, 2, 16), torch.full((1, 3, 16), float("nan"))):
            with self.assertRaises(ValueError):
                decode_fixed_latents(model, "Hi\n", latents)
        with self.assertRaises(ValueError):
            crossed_latents(torch.zeros(1, 3, 16), torch.zeros(1, 3, 16), 0)

    def test_cli_export_and_rescoring_with_real_saved_tiny_trajectories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base"
            base.mkdir()
            vocab = {c: i for i, c in enumerate(bytes_to_unicode().values())}
            vocab["<|endoftext|>"] = 256
            (base / "vocab.json").write_text(json.dumps(vocab))
            (base / "merges.txt").write_text("#version: 0.2\n")
            GPT2Tokenizer(str(base / "vocab.json"), str(base / "merges.txt")).save_pretrained(base)
            torch.manual_seed(321)
            GPT2LMHeadModel(GPT2Config(vocab_size=257, n_positions=128, n_embd=16, n_layer=2, n_head=2,
                                     eos_token_id=256, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
                                     _attn_implementation="eager")).save_pretrained(base)
            from common.models.coconut_model import CoconutWrapper
            wrapper = CoconutWrapper()
            wrapper.load_from_config({"base_model_name_or_path": str(base), "device": "cpu", "num_latent_placeholders": 3})
            checkpoint = root / "checkpoint"
            torch.save(wrapper.coconut_model.state_dict(), checkpoint)
            dataset = root / "data.jsonl"
            dataset.write_text(json.dumps({"question": "2+3?", "answer": "#### 5"}) + "\n")
            config = root / "config.json"
            config.write_text(json.dumps({"dataset": "gsm8k", "split": "synthetic", "dataset_path": str(dataset),
                "checkpoint_path": str(checkpoint), "base_model_path": str(base), "prompt_template": "{question}\n",
                "num_latent_steps": 3, "max_new_tokens": 3, "seed": 0, "max_samples": 1, "atol": 1e-5,
                "rtol": 1e-5, "random_strengths": [0.5], "noise_seeds": [0], "save_trace_samples": 1}))
            machine = root / "machine.json"
            machine.write_text(json.dumps({"OUTPUT_DIR": str(root / "outputs"), "device": "cpu"}))
            env = {**os.environ, "OMP_NUM_THREADS": "1", "HF_HOME": str(root / "hf-cache")}

            def execute(script, *args):
                command = [sys.executable, str(ROOT / script), "--config", str(config), "--machine-config", str(machine), *args]
                result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            baseline, gate, source = (root / "outputs" / name for name in ("baseline", "gate", "source"))
            execute("run_baseline.py", "--run-id", "baseline")
            execute("check_resume.py", "--run-id", "gate", "--baseline-run", str(baseline))
            execute("run_interventions.py", "--run-id", "source", "--baseline-run", str(baseline), "--resume-check", str(gate))
            execute("run_path_controls.py", "--run-id", "paths", "--input-run", str(source), "--sample-index", "0")
            output = root / "outputs/paths"
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["record_count"], 12)
            self.assertEqual(summary["source_conditions"], 3)
            result = subprocess.run([sys.executable, str(ROOT / "analyze_first_round.py"), "--run-dir", str(output),
                                     "--output-dir", str(root / "analysis")], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            original = [json.loads(s) for s in (source / "predictions.jsonl").read_text().splitlines()]
            actual = [json.loads(s) for s in (output / "predictions.jsonl").read_text().splitlines()]
            for row in actual:
                if row["operation"] == "edited_replay":
                    reference = next(s for s in original if s["operation"] == "random" and s["step"] == row["step"])
                    self.assertEqual(row["generated_token_ids"], reference["generated_token_ids"])


if __name__ == "__main__":
    unittest.main()

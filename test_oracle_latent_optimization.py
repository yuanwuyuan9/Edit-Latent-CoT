"""Check full feedback gradients, projection, and independent free-generation CLI."""
import importlib.metadata
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

from experiments.first_round.data import load_samples, score
from experiments.first_round.run import digest, provenance, stable_hash, write_json
from experiments.first_round.test_tiny_model import tiny_model
from run_counterfactual_family import delta_hash
from run_same_question_donors import runtime_info
from run_oracle_latent_optimization import projected_search, teacher_forward, validate_protocol

ROOT = Path(__file__).resolve().parent


class OracleOptimizationTests(unittest.TestCase):
    def test_feedback_and_teacher_logits_match_cached_inference(self):
        model = tiny_model()
        model.base_model.requires_grad_(False)
        ids = torch.tensor([[3, 7, 36]])
        for step in (1, 2, 3):
            h, state = model.forward_until_step('Hi\n', step)
            for changed in (h, h * 0.5):
                _, logits, embeds = teacher_forward(model, state, changed, ids)
                output = model.rollout_from_step(changed, state)
                cached = model.compute_logits(changed, output, ids)
                torch.testing.assert_close(logits, cached)
                torch.testing.assert_close(embeds, output['inputs_embeds'])
        _, _, first = teacher_forward(model, state, h, ids)
        _, _, second = teacher_forward(model, state, h, torch.tensor([[9, 2, 36]]))
        torch.testing.assert_close(first, second)

    def test_finite_difference_and_gradient_through_future_feedback(self):
        model = tiny_model()
        model.base_model.requires_grad_(False)
        h, state = model.forward_until_step('Hi\n', 2)
        x = h.clone().requires_grad_(True)
        ids = torch.tensor([[3, 7, 36]])
        loss, _, embeds = teacher_forward(model, state, x, ids)
        future = embeds[:, state['latent_lists'][0][2]]
        feedback_gradient, = torch.autograd.grad(future.square().sum(), x, retain_graph=True)
        self.assertGreater(float(feedback_gradient.norm()), 0)
        gradient, = torch.autograd.grad(loss, x)
        direction = gradient / gradient.norm()
        epsilon = 0.01
        with torch.no_grad():
            plus = teacher_forward(model, state, h + epsilon * direction, ids)[0]
            minus = teacher_forward(model, state, h - epsilon * direction, ids)[0]
        numerical = (plus - minus) / (2 * epsilon)
        analytical = (gradient * direction).sum()
        torch.testing.assert_close(numerical, analytical, atol=2e-5, rtol=0.03)
        self.assertTrue(all(p.grad is None for p in model.base_model.parameters()))

    def test_projection_improves_loss_without_mutating_prefix_or_weights(self):
        model = tiny_model()
        model.base_model.requires_grad_(False)
        h, state = model.forward_until_step('Hi\n', 2)
        before = state['inputs_embeds'].clone()
        weights = {name: p.detach().clone() for name, p in model.base_model.named_parameters()}
        protocol = {'updates': 20, 'step_fraction_of_radius': 0.05, 'initial_fraction_of_radius': 0.1}
        for seed in (0, 1):
            delta, log = projected_search(model, state, h, torch.tensor([[3, 7, 36]]), 0.5, seed, protocol)
            self.assertLess(log['best_teacher_nll'], log['initial_teacher_nll'])
            self.assertLessEqual(float(delta.norm() / h.norm()), 0.500001)
            self.assertEqual(len(log['history']), 21)
            self.assertTrue(all(r['relative_norm'] <= 0.500001 for r in log['history']))
        self.assertTrue(torch.equal(before, state['inputs_embeds']))
        for name, p in model.base_model.named_parameters():
            self.assertTrue(torch.equal(p, weights[name]))
            self.assertIsNone(p.grad)
        full = json.loads((ROOT / 'configs/first_round/gsm8k_oracle_latent_optimization.json').read_text())
        validate_protocol(full, 6)
        for update in ({'updates': 0}, {'relative_radii': [0]}, {'restart_seeds': [1]}, {'append_eos': False}):
            with self.assertRaises(ValueError):
                validate_protocol({**full, **update}, 6)

    def test_cli_complete_search_and_free_generation_separate_from_teacher_loss(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / 'base'
            base.mkdir()
            vocab = {c: i for i, c in enumerate(bytes_to_unicode().values())}
            vocab.update({'<|endoftext|>': 256, '20': 257})
            (base / 'vocab.json').write_text(json.dumps(vocab))
            (base / 'merges.txt').write_text('#version: 0.2\n2 0\n')
            GPT2Tokenizer(str(base / 'vocab.json'), str(base / 'merges.txt')).save_pretrained(base)
            torch.manual_seed(321)
            network = GPT2LMHeadModel(GPT2Config(vocab_size=258, n_positions=128, n_embd=16,
                n_layer=2, n_head=2, eos_token_id=256, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
                _attn_implementation='eager'))
            with torch.no_grad():
                network.transformer.ln_f.weight.fill_(0.1)
                network.transformer.ln_f.bias.zero_()
                network.transformer.ln_f.bias[0] = 10
                network.transformer.wte.weight[257].zero_()
                network.transformer.wte.weight[257, 0] = 1
            network.save_pretrained(base)
            from common.models.coconut_model import CoconutWrapper
            model = CoconutWrapper()
            model.load_from_config({'base_model_name_or_path': str(base), 'device': 'cpu',
                'num_latent_placeholders': 3, 'generation_kwargs': {'max_new_tokens': 1}})
            checkpoint = root / 'checkpoint'
            torch.save(model.coconut_model.state_dict(), checkpoint)
            parent = root / 'outputs/parent'
            parent.mkdir(parents=True)
            family = [{'question': f'flock is {n} chickens', 'answer': f'#### {3*n-40}',
                       'chicken_count': n, 'is_calibration': n == 20} for n in (20, 18)]
            (parent / 'family.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in family))
            cfg = {'dataset': 'gsm8k', 'split': 'synthetic_flock_family', 'dataset_path': str(parent / 'family.jsonl'),
                   'checkpoint_path': str(checkpoint), 'base_model_path': str(base), 'prompt_template': '{question}\n',
                   'num_latent_steps': 3, 'max_new_tokens': 1, 'seed': 0, 'max_samples': 0, 'atol': 1e-5, 'rtol': 1e-5,
                   'random_strengths': [0.5], 'noise_seeds': [0], 'save_trace_samples': 2, 'device': 'cpu'}
            config, machine, protocol_path = (root / name for name in ('config.json', 'machine.json', 'protocol.json'))
            write_json(config, cfg)
            write_json(machine, {'device': 'cpu', 'OUTPUT_DIR': str(root / 'outputs')})
            protocol = json.loads((ROOT / 'configs/first_round/gsm8k_oracle_latent_optimization.json').read_text())
            protocol.update(target_step=2, relative_radii=[0.1], restart_seeds=[0, 1], updates=3, random_control_seeds=[3000, 3001])
            write_json(protocol_path, protocol)
            write_json(parent / 'protocol.json', {'target_step': 2})
            delta = torch.linspace(-0.1, 0.1, 16).view(1, 16)
            torch.save({'delta': delta}, parent / 'frozen_edit.pt')
            samples = load_samples(parent / 'family.jsonl', 'gsm8k', cfg['split'])
            predictions = []
            with torch.no_grad():
                for sample in samples:
                    prompt = sample['question'] + '\n'
                    baseline = model.run_baseline(prompt)
                    h, state = model.forward_until_step(prompt, 2)
                    edited = model.rollout_from_step(h + delta, state)
                    for op, output in (('baseline', baseline), ('frozen_delta', edited)):
                        predictions.append({'sample_id': sample['sample_id'], 'operation': op, 'status': 'ok',
                            'generated_token_ids': output['generated_token_ids'][0], 'continuation': output['continuation'][0],
                            **score(output['continuation'][0], sample, 'gsm8k')})
            (parent / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in predictions))
            packages = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'numpy', 'huggingface_hub')}
            runtime = runtime_info(torch)
            runtime['torch_threads'] = 1
            prov = provenance(cfg, packages, runtime)
            prov['identity']['delta_values_sha256'] = delta_hash(delta)
            prov['signature'] = stable_hash(prov['identity'])
            for name in prov['identity']['source_hashes']:
                target = parent / 'source' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / name).read_bytes())
            write_json(parent / 'manifest.json', {**prov, 'stage': 'counterfactual_family', 'status': 'completed', 'run_id': 'parent'})
            write_json(parent / 'checksums.json', {str(p.relative_to(parent)): digest(p) for p in parent.rglob('*') if p.is_file()})
            env = {**os.environ, 'OMP_NUM_THREADS': '1', 'HF_HOME': str(root / 'hf-cache'), 'TOKENIZERS_PARALLELISM': 'false'}
            result = subprocess.run([sys.executable, str(ROOT / 'run_oracle_latent_optimization.py'), '--config', str(config),
                '--machine-config', str(machine), '--family-run', str(parent), '--protocol', str(protocol_path), '--run-id', 'optimized'],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = root / 'outputs/optimized'
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual((manifest['status'], manifest['record_count']), ('completed', 22))
            self.assertTrue(manifest['weights_unchanged'])
            records = [json.loads(line) for line in (output / 'predictions.jsonl').read_text().splitlines()]
            for row in records:
                self.assertAlmostEqual(sum(row['teacher_token_nll']) / len(row['teacher_token_nll']), row['teacher_nll'], places=5)
                if row['operation'] == 'optimized':
                    log = json.loads((output / row['optimization_log']).read_text())
                    self.assertEqual(len(log['history']), 4)
                    self.assertLessEqual(row['relative_edit_norm'], 0.10001)
                    self.assertEqual(row['predicted_answer'], '20')
                    if row['chicken_count'] == 18:
                        self.assertFalse(row['correct'])
            result = subprocess.run([sys.executable, str(ROOT / 'analyze_first_round.py'), '--run-dir', str(output),
                '--output-dir', str(root / 'analysis')], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()

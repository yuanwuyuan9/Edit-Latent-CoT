"""Verify donor construction, natural continuation and the complete export CLI."""
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
from experiments.first_round.run import digest, provenance, random_edit, stable_hash, write_json
from experiments.first_round.test_tiny_model import tiny_model
from run_counterfactual_family import delta_hash
from run_same_question_donors import (
    check_state, check_trace, donor_controls, identity_state, runtime_info, search_conditions,
)

ROOT = Path(__file__).resolve().parent


class SameQuestionDonorTests(unittest.TestCase):
    def test_fixed_budget_and_invalid_search(self):
        protocol = json.loads((ROOT / 'configs/first_round/gsm8k_same_question_donors.json').read_text())
        self.assertEqual(len(search_conditions(protocol, 6)), 90)
        self.assertIn((1, 1.0, 9), search_conditions(protocol, 6))
        for update in ({'source_steps': [1, 1]}, {'source_steps': [5]}, {'strengths': [0]},
                       {'noise_seeds': [-1]}, {'control_seed_start': 0}, {'target_step': 7}):
            with self.assertRaises(ValueError):
                search_conditions({**protocol, **update}, 6)

    @torch.no_grad()
    def test_original_prefix_natural_feedback_and_equal_control_norms(self):
        model = tiny_model()
        prompt = 'Hi\n'
        tokens = model._prepare_inputs(prompt)
        positions = (tokens['input_ids'][0] == model.latent_token_id).nonzero().flatten()
        reference = model.coconut_model(**tokens, labels=tokens['input_ids'])
        base = reference.inputs_embeds[:, positions].clone()
        logits = reference.logits[:, -1:, :].clone()
        ids = model.run_baseline(prompt)['generated_token_ids'][0]
        cfg = {'atol': 1e-5, 'rtol': 1e-5}
        h1, state1, _, snapshot1 = identity_state(model, prompt, 1, positions, base, logits, ids, cfg)
        source = model.rollout_from_step(torch.zeros_like(h1), state1)
        source_latents = source['inputs_embeds'][:, positions]
        h2, state2, _, snapshot2 = identity_state(model, prompt, 2, positions, base, logits, ids, cfg)
        donor = source_latents[:, 1].clone()
        branches = donor_controls(h2, donor, 1000)
        for kind, inserted in branches.items():
            torch.testing.assert_close((inserted - h2).norm(), (donor - h2).norm())
            output = model.rollout_from_step(inserted, state2)
            actual = check_trace(torch, output, positions, base, 2, inserted, cfg)
            if kind == 'donor':
                torch.testing.assert_close(actual[:, 1], donor)
                self.assertGreater(float((actual[:, 2:] - source_latents[:, 2:]).norm()), 0)
        torch.testing.assert_close(branches['reverse'] - h2, -(donor - h2))
        check_state(model, state1, snapshot1)
        check_state(model, state2, snapshot2)
        state2['inputs_embeds'].add_(1)
        with self.assertRaises(ValueError):
            check_state(model, state2, snapshot2)

    def test_cli_all_candidates_including_failed_sources_and_offline_scoring(self):
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
            (parent / 'traces').mkdir(parents=True)
            family = [{'question': f'flock is {n} chickens', 'answer': f'#### {3*n-40}',
                       'chicken_count': n, 'is_calibration': n == 20} for n in (20, 18)]
            (parent / 'family.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in family))
            cfg = {'dataset': 'gsm8k', 'split': 'synthetic_flock_family', 'dataset_path': str(parent / 'family.jsonl'),
                   'checkpoint_path': str(checkpoint), 'base_model_path': str(base), 'prompt_template': '{question}\n',
                   'num_latent_steps': 3, 'max_new_tokens': 1, 'seed': 0, 'max_samples': 0, 'atol': 1e-5, 'rtol': 1e-5,
                   'random_strengths': [0.5], 'noise_seeds': [0], 'save_trace_samples': 2, 'device': 'cpu'}
            config, machine, protocol = (root / n for n in ('config.json', 'machine.json', 'search.json'))
            write_json(config, cfg)
            write_json(machine, {'device': 'cpu', 'OUTPUT_DIR': str(root / 'outputs')})
            write_json(protocol, {'target_step': 2, 'source_steps': [1], 'strengths': [0.5], 'noise_seeds': [0, 1], 'control_seed_start': 1000})
            write_json(parent / 'protocol.json', {'target_step': 2, 'source_step': 1, 'source_strength': 0.5,
                                                  'source_noise_seed': 0, 'scale': 1.0})
            samples = load_samples(parent / 'family.jsonl', 'gsm8k', cfg['split'])
            with torch.no_grad():
                h1, state1 = model.forward_until_step(samples[0]['question'] + '\n', 1)
                source = model.rollout_from_step(random_edit(torch, h1, 0.5, 0), state1)
                positions = (model._prepare_inputs(samples[0]['question'] + '\n')['input_ids'][0] == model.latent_token_id).nonzero().flatten()
                h2, _ = model.forward_until_step(samples[0]['question'] + '\n', 2)
                delta = source['inputs_embeds'][:, positions][:, 1] - h2
                torch.save({'delta': delta.cpu()}, parent / 'frozen_edit.pt')
                predictions = []
                for sample in samples:
                    prompt = sample['question'] + '\n'
                    baseline = model.run_baseline(prompt)
                    h, state = model.forward_until_step(prompt, 2)
                    edited = model.rollout_from_step(h + delta, state)
                    for op, output in (('baseline', baseline), ('frozen_delta', edited)):
                        predictions.append({'sample_id': sample['sample_id'], 'operation': op,
                                            'generated_token_ids': output['generated_token_ids'][0], 'status': 'ok',
                                            'continuation': output['continuation'][0], **score(output['continuation'][0], sample, 'gsm8k')})
            (parent / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in predictions))
            # Match the child subprocess's CPU threading environment.
            packages = {'torch': torch.__version__, 'transformers': '4.44.2', 'numpy': '1.26.4', 'huggingface_hub': '0.36.2'}
            import importlib.metadata
            packages = {key: importlib.metadata.version(key) for key in packages}
            runtime = runtime_info(torch)
            runtime['torch_threads'] = 1
            prov = provenance(cfg, packages, runtime)
            prov['identity']['delta_values_sha256'] = delta_hash(delta)
            prov['signature'] = stable_hash(prov['identity'])
            for name, checksum in prov['identity']['source_hashes'].items():
                target = parent / 'source' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / name).read_bytes())
            write_json(parent / 'manifest.json', {**prov, 'stage': 'counterfactual_family', 'status': 'completed', 'run_id': 'parent'})
            write_json(parent / 'checksums.json', {str(p.relative_to(parent)): digest(p) for p in parent.rglob('*') if p.is_file()})
            env = {**os.environ, 'OMP_NUM_THREADS': '1', 'HF_HOME': str(root / 'hf-cache')}
            command = [sys.executable, str(ROOT / 'run_same_question_donors.py'), '--config', str(config),
                       '--machine-config', str(machine), '--family-run', str(parent), '--protocol', str(protocol), '--run-id', 'donors']
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = root / 'outputs/donors'
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual((summary['status'], summary['record_count'], summary['expected_records']), ('completed', 24, 24))
            records = [json.loads(line) for line in (output / 'predictions.jsonl').read_text().splitlines()]
            failed_sources = [r for r in records if r.get('branch_kind') == 'source' and not r['correct']]
            self.assertEqual(len(failed_sources), 2)
            self.assertEqual(len([r for r in records if r.get('branch_kind') == 'donor' and not r['source_correct']]), 2)
            self.assertFalse(summary['per_question'][1]['any_donor_correct'])
            for donor in (r for r in records if r.get('branch_kind') == 'donor'):
                src = torch.load(output / donor['source_trace_path'], weights_only=True)['latent_inputs']
                saved = torch.load(output / donor['trace_path'], weights_only=True)
                self.assertTrue(torch.equal(saved['latent_inputs'][:, 1], src[:, 1]))
                self.assertTrue(torch.equal(saved['latent_inputs'][:, :1], saved['baseline_latents'][:, :1]))
            result = subprocess.run([sys.executable, str(ROOT / 'analyze_first_round.py'), '--run-dir', str(output),
                                     '--output-dir', str(root / 'analysis')], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()

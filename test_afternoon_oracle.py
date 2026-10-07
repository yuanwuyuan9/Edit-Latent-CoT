"""Check absolute budget matching and a real tiny-model artifact pipeline."""
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

from common.models.coconut_model import CoconutWrapper
from experiments.first_round.data import load_samples, score
from experiments.first_round.run import digest, provenance, stable_hash, write_json
from experiments.first_round.test_tiny_model import tiny_model
from run_afternoon_oracle import matched_budget, paired_summary, validate_protocol
from run_counterfactual_family import checked_run, delta_hash, evaluate_question
from run_oracle_latent_optimization import projected_search
from run_same_question_donors import runtime_info

ROOT = Path(__file__).resolve().parent


class AfternoonOracleTests(unittest.TestCase):
    def test_same_absolute_cap_for_different_latent_norms(self):
        model = tiny_model()
        model.base_model.requires_grad_(False)
        h, state = model.forward_until_step('Hi\n', 2)
        fixed = torch.linspace(-0.1, 0.1, h.numel()).reshape_as(h)
        protocol = json.loads((ROOT / 'configs/first_round/gsm8k_afternoon_oracle.json').read_text())
        validate_protocol(protocol, 6)
        protocol['updates'] = 3
        caps, radii = [], []
        for scale in (0.5, 2):
            original = h*scale
            cap, radius = matched_budget(original, fixed)
            delta, history = projected_search(model, state, original, torch.tensor([[3, 7, 36]]), radius, 0, protocol)
            caps.append(cap)
            radii.append(radius)
            self.assertLessEqual(float(delta.norm()), cap+1e-6)
            self.assertLessEqual(float(((original+delta)-original).norm()), cap+1e-6)
            self.assertEqual(history['history'][0]['relative_norm'], 0)
            self.assertTrue(all(r['relative_norm']*float(original.norm()) <= cap+1e-6 for r in history['history']))
        self.assertEqual(caps[0], caps[1])
        self.assertAlmostEqual(radii[0]/radii[1], 4)
        for h_bad, fixed_bad in ((h*0, fixed), (h, fixed*0), (h, fixed*float('nan'))):
            with self.assertRaises(ValueError):
                matched_budget(h_bad, fixed_bad)
        for change in ({'restart_seed': 1}, {'restart_seed': False}, {'updates': 0},
                       {'target_step': 6}, {'append_eos': False}):
            with self.assertRaises(ValueError):
                validate_protocol({**protocol, **change}, 6)

    def test_cli_four_searches_replays_and_free_generation_scoring(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root, base = Path(temp), Path(temp) / 'base'
            base.mkdir()
            vocab = {c: i for i, c in enumerate(bytes_to_unicode().values())}
            vocab.update({'<|endoftext|>': 256, '20': 257})
            (base / 'vocab.json').write_text(json.dumps(vocab))
            (base / 'merges.txt').write_text('#version: 0.2\n2 0\n')
            GPT2Tokenizer(str(base / 'vocab.json'), str(base / 'merges.txt')).save_pretrained(base)
            torch.manual_seed(321)
            network = GPT2LMHeadModel(GPT2Config(vocab_size=258, n_positions=128, n_embd=16, n_layer=2, n_head=2,
                eos_token_id=256, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, _attn_implementation='eager'))
            with torch.no_grad():
                network.transformer.ln_f.weight.fill_(0.1)
                network.transformer.ln_f.bias.zero_()
                network.transformer.ln_f.bias[0] = 10
                network.transformer.wte.weight[257].zero_()
                network.transformer.wte.weight[257, 0] = 1
            network.save_pretrained(base)
            model = CoconutWrapper()
            model.load_from_config({'base_model_name_or_path': str(base), 'device': 'cpu', 'num_latent_placeholders': 3,
                                    'generation_kwargs': {'max_new_tokens': 1}})
            checkpoint = root / 'checkpoint'
            torch.save(model.coconut_model.state_dict(), checkpoint)
            parent = root / 'outputs/frozen'
            (parent / 'traces').mkdir(parents=True)
            cups = [25, 17, 21, 29, 33]
            family = [{'question': f'another {a} cups of feed for 20 chickens', 'answer': f'#### {45-a}',
                'chicken_count': 20, 'afternoon_cups': a, 'afternoon_change': a-25, 'is_calibration': a == 25} for a in cups]
            (parent / 'family.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in family))
            cfg = {'dataset': 'gsm8k', 'split': 'synthetic_afternoon_family', 'dataset_path': str(parent / 'family.jsonl'),
                'checkpoint_path': str(checkpoint), 'base_model_path': str(base), 'prompt_template': '{question}\n',
                'num_latent_steps': 3, 'max_new_tokens': 1, 'seed': 0, 'max_samples': 0, 'atol': 1e-5, 'rtol': 1e-5,
                'random_strengths': [0.5], 'noise_seeds': [0], 'save_trace_samples': 1, 'device': 'cpu'}
            config, machine, protocol_path = root / 'config.json', root / 'machine.json', root / 'protocol.json'
            write_json(config, cfg)
            write_json(machine, {'device': 'cpu', 'OUTPUT_DIR': str(root / 'outputs')})
            samples = load_samples(parent / 'family.jsonl', 'gsm8k', cfg['split'])
            fixed = torch.linspace(-0.1, 0.1, 16).view(1, 16)
            with torch.no_grad():
                h, state = model.forward_until_step(samples[0]['question']+'\n', 2)
                fixed = (h+fixed)-h
            predictions, source_base, source_latents = [], None, None
            for sample, raw in zip(samples, family):
                sample.update({k: raw[k] for k in ('chicken_count', 'afternoon_cups', 'afternoon_change', 'is_calibration')})
                outputs, original = evaluate_question(model, sample, cfg, 2, [('identity', None, fixed*0), ('frozen_gold', None, fixed)])
                for op, seed, output, actual, norm in outputs:
                    path = f'traces/{len(predictions):03d}_{op}.pt'
                    torch.save({'sample_id': sample['sample_id'], 'baseline_latents': original,
                        'latent_inputs': actual, 'delta': actual[:, 1]-original[:, 1]}, parent / path)
                    predictions.append({'sample_id': sample['sample_id'], 'operation': op, 'trace_path': path,
                        'generated_token_ids': output['generated_token_ids'][0], 'continuation': output['continuation'][0],
                        **score(output['continuation'][0], sample, 'gsm8k')})
                    if sample['is_calibration'] and op == 'frozen_gold':
                        source_base, source_latents = original, actual
            torch.save({'source_baseline_latents': source_base, 'edits': {'gold': {'delta': fixed, 'source_latents': source_latents}}}, parent / 'frozen_edits.pt')
            (parent / 'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
            write_json(parent / 'protocol.json', {'target_step': 2, 'afternoon_cups': cups})
            packages = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'numpy', 'huggingface_hub')}
            prov = provenance(cfg, packages, runtime_info(torch))
            prov['identity']['frozen_delta_hashes'] = {'gold': delta_hash(fixed)}
            prov['signature'] = stable_hash(prov['identity'])
            for name in prov['identity']['source_hashes']:
                path = parent / 'source' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT / name).read_bytes())
            write_json(parent / 'manifest.json', {**prov, 'stage': 'frozen_consequences', 'status': 'completed',
                'run_id': 'frozen', 'weights_unchanged': True})
            write_json(parent / 'checksums.json', {str(p.relative_to(parent)): digest(p) for p in parent.rglob('*') if p.is_file()})
            protocol = json.loads((ROOT / 'configs/first_round/gsm8k_afternoon_oracle.json').read_text())
            protocol.update(parent_signature=prov['signature'], frozen_delta_sha256=delta_hash(fixed), target_step=2, updates=3)
            write_json(protocol_path, protocol)
            env = {**os.environ, 'OMP_NUM_THREADS': '1', 'HF_HOME': str(root / 'hf-cache'), 'TOKENIZERS_PARALLELISM': 'false'}
            command = [sys.executable, str(ROOT / 'run_afternoon_oracle.py'), '--config', str(config), '--machine-config', str(machine),
                '--frozen-run', str(parent), '--protocol', str(protocol_path), '--run-id', 'oracle']
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            output = root / 'outputs/oracle'
            manifest, rows = checked_run(output, 'afternoon_oracle')
            self.assertEqual((manifest['record_count'], manifest['expected_records'], manifest['completed_searches']), (19, 19, 4))
            self.assertTrue(manifest['weights_unchanged'])
            self.assertEqual(paired_summary(rows), {k: json.loads((output / 'summary.json').read_text())[k]
                for k in ('per_question', 'new_variant_oracle', 'new_variant_frozen')})
            searched = [r for r in rows if r['operation'] == 'optimized_gold']
            self.assertEqual(len(searched), 4)
            self.assertEqual({r['afternoon_cups'] for r in searched}, {17, 21, 29, 33})
            self.assertEqual(len({r['absolute_norm_cap'] for r in rows}), 1)
            self.assertTrue(all(not r['correct'] for r in searched))
            for row in rows:
                self.assertEqual(row['feedback_max_abs_error'], 0)
                self.assertEqual(row['teacher_logits_max_abs_error'], 0)
                self.assertLessEqual(row['edit_norm'], row['absolute_norm_cap']+1e-5)
                self.assertAlmostEqual(row['absolute_norm_cap'], row['relative_radius']*float(torch.load(
                    output / row['trace_path'], weights_only=True)['original_h'].norm()), places=6)
                if row['operation'] in ('baseline', 'frozen_gold'):
                    self.assertTrue(row['parent_replay']['close'] and row['parent_replay']['tokens_equal'])
                if row['operation'] == 'optimized_gold':
                    log = json.loads((output / row['optimization_log']).read_text())
                    self.assertEqual(len(log['history']), 4)
                    self.assertEqual(log['history'][0]['relative_norm'], 0)
                    self.assertAlmostEqual(row['teacher_nll'], log['best_teacher_nll'], places=4)
            # An existing run must remain intact rather than being overwritten.
            before = digest(output / 'checksums.json')
            second = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(digest(output / 'checksums.json'), before)
            result = subprocess.run([sys.executable, str(ROOT / 'analyze_first_round.py'), '--run-dir', str(output),
                '--output-dir', str(root / 'analysis')], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()

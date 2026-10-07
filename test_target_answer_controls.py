"""Check target/gold separation, inherited budgets, and the complete tiny CLI."""
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
from run_oracle_latent_optimization import projected_search, teacher_forward
from run_same_question_donors import runtime_info
from run_target_answer_controls import compare_gold, compare_search_history, target_cases, target_summary

ROOT = Path(__file__).resolve().parent


class TargetAnswerControlTests(unittest.TestCase):
    def test_fixed_targets_and_invalid_or_trivial_controls(self):
        protocol = {'target_offsets': [0, -4, 4]}
        cases = target_cases({'gold_answer': '20'}, '10', protocol)
        self.assertEqual([c['target_answer'] for c in cases], ['20', '16', '24'])
        for sample, baseline, changed in [({'gold_answer': '4'}, '10', protocol),
                ({'gold_answer': '20'}, '16', protocol), ({'gold_answer': '20.0'}, '10', protocol),
                ({'gold_answer': '20'}, '10', {'target_offsets': [0, -6, 6]})]:
            with self.assertRaises(ValueError):
                target_cases(sample, baseline, changed)

    def test_success_means_target_hit_separate_from_true_gold(self):
        base = {'sample_id': 'q', 'chicken_count': 20, 'is_calibration': True,
                'gold_answer': '20', 'predicted_answer': '10', 'operation': 'baseline'}
        wrong = {**base, 'operation': 'optimized_minus4', 'target_kind': 'minus4',
                 'target_answer': '16', 'predicted_answer': '16', 'target_hit': True,
                 'correct': False, 'strength': 0.25, 'parse_status': 'ok'}
        result = target_summary([base, wrong])[0]['targets'][1]
        self.assertEqual((result['target_hits'], result['gold_correct']), (1, 0))
        self.assertEqual(result['smallest_tested_success_radius'], 0.25)

    def test_fixed_input_replay_and_independent_search_comparisons_are_distinct(self):
        trace = torch.zeros(1, 3, 2)
        positions = torch.tensor([0, 1, 2])
        old = {'generated_token_ids': [7], 'teacher_nll': 0.01}
        metrics = {'teacher_nll': 0.01}
        replay = {'inputs_embeds': trace.clone(), 'generated_token_ids': [[7]]}
        cfg = {'atol': 1e-5, 'rtol': 1e-5}
        self.assertTrue(compare_gold(replay, metrics, old, trace, positions, cfg)['matches'])
        searched = {'inputs_embeds': trace.clone(), 'generated_token_ids': [[7]]}
        searched['inputs_embeds'][0, 2, 0] = 1e-4
        result = compare_gold(searched, metrics, old, trace, positions, cfg)
        self.assertFalse(result['matches'])
        self.assertTrue(result['tokens_equal'] and result['teacher_nll']['close'])
        self.assertEqual(result['latents']['failed_elements'], 1)
        self.assertFalse(compare_gold(replay, {'teacher_nll': 0.1}, old, trace, positions, cfg)['matches'])
        self.assertFalse(compare_gold({**replay, 'generated_token_ids': [[8]]}, metrics, old, trace, positions, cfg)['matches'])
        previous = {'best_iteration': 0, 'history': [{'iteration': 0, 'teacher_nll': 0.01, 'relative_norm': 0.1}]}
        current = {'best_iteration': 0, 'history': [{'iteration': 0, 'teacher_nll': 0.01000001, 'relative_norm': 0.1}]}
        self.assertEqual(compare_search_history(current, previous)['first_exact_difference']['iteration'], 0)

    def test_cli_reproduces_gold_and_reuses_search_budget_for_wrong_targets(self):
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
            model.base_model.requires_grad_(False)
            parent = root / 'outputs/oracle'
            (parent / 'traces').mkdir(parents=True)
            (parent / 'optimization').mkdir()
            family = [{'question': f'flock is {n} chickens', 'answer': f'#### {3*n-40}',
                       'chicken_count': n, 'is_calibration': n == 20} for n in (20, 18)]
            (parent / 'family.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in family))
            cfg = {'dataset': 'gsm8k', 'split': 'synthetic_flock_family', 'dataset_path': str(parent / 'family.jsonl'),
                'checkpoint_path': str(checkpoint), 'base_model_path': str(base), 'prompt_template': '{question}\n',
                'num_latent_steps': 3, 'max_new_tokens': 1, 'seed': 0, 'max_samples': 0, 'atol': 1e-5, 'rtol': 1e-5,
                'random_strengths': [0.5], 'noise_seeds': [0], 'save_trace_samples': 2, 'device': 'cpu'}
            config, machine = root / 'config.json', root / 'machine.json'
            write_json(config, cfg)
            write_json(machine, {'device': 'cpu', 'OUTPUT_DIR': str(root / 'outputs')})
            search = json.loads((ROOT / 'configs/first_round/gsm8k_oracle_latent_optimization.json').read_text())
            search.update(target_step=2, relative_radii=[0.1], restart_seeds=[0, 1], updates=3)
            write_json(parent / 'protocol.json', search)
            samples = load_samples(parent / 'family.jsonl', 'gsm8k', cfg['split'])
            prior, gold_histories = [], {}
            for sample in samples:
                prompt = sample['question'] + '\n'
                with torch.no_grad():
                    baseline = model.run_baseline(prompt)
                    h, state = model.forward_until_step(prompt, 2)
                prior.append({'sample_id': sample['sample_id'], 'operation': 'baseline',
                    'generated_token_ids': baseline['generated_token_ids'][0], 'predicted_answer': '20'})
                ids = torch.tensor([model.tokenizer('### ' + sample['gold_answer'])['input_ids'] + [model.eos_token_id]])
                for seed in search['restart_seeds']:
                    delta, log = projected_search(model, state, h, ids, 0.1, seed, search)
                    gold_histories[(sample['sample_id'], seed)] = log['history']
                    log_path = f"optimization/{sample['index']}_{seed}.json"
                    write_json(parent / log_path, log)
                    with torch.no_grad():
                        output = model.rollout_from_step(h + delta, state)
                        loss = teacher_forward(model, state, h + delta, ids)[0]
                    positions = state['latent_lists'][0]
                    path = f"traces/{sample['index']}_{seed}.pt"
                    torch.save({'latent_inputs': output['inputs_embeds'][:, positions]}, parent / path)
                    prior.append({'sample_id': sample['sample_id'], 'operation': 'optimized', 'strength': 0.1,
                        'restart_seed': seed, 'teacher_nll': float(loss), 'trace_path': path,
                        'optimization_log': log_path,
                        'generated_token_ids': output['generated_token_ids'][0]})
            (parent / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in prior))
            packages = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'numpy', 'huggingface_hub')}
            prov = provenance(cfg, packages, runtime_info(torch))
            for name in ('run_oracle_latent_optimization.py', 'test_oracle_latent_optimization.py'):
                prov['identity']['source_hashes'][name] = digest(ROOT / name)
            prov['signature'] = stable_hash(prov['identity'])
            for name in prov['identity']['source_hashes']:
                path = parent / 'source' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT / name).read_bytes())
            write_json(parent / 'manifest.json', {**prov, 'stage': 'oracle_latent_optimization',
                'status': 'completed', 'run_id': 'oracle', 'weights_unchanged': True})
            write_json(parent / 'checksums.json', {str(p.relative_to(parent)): digest(p) for p in parent.rglob('*') if p.is_file()})
            env = {**os.environ, 'OMP_NUM_THREADS': '1', 'HF_HOME': str(root / 'hf-cache'), 'TOKENIZERS_PARALLELISM': 'false'}
            result = subprocess.run([sys.executable, str(ROOT / 'run_target_answer_controls.py'), '--config', str(config),
                '--machine-config', str(machine), '--oracle-run', str(parent), '--run-id', 'controls'],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = root / 'outputs/controls'
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual((manifest['status'], manifest['record_count'], manifest['gold_parent_replay_passed']), ('completed', 16, 4))
            self.assertEqual(manifest['gold_search_matches_parent'], 4)
            self.assertEqual(manifest['gold_search_differences'], 0)
            self.assertTrue(manifest['weights_unchanged'])
            self.assertEqual(json.loads((output / 'search_protocol.json').read_text()), search)
            rows = [json.loads(line) for line in (output / 'predictions.jsonl').read_text().splitlines()]
            for row in rows:
                self.assertEqual(row['feedback_max_abs_error'], 0)
                self.assertEqual(row['teacher_logits_max_abs_error'], 0)
                sample = next(s for s in samples if s['sample_id'] == row['sample_id'])
                self.assertEqual(row['correct'], score(row['continuation'], sample, 'gsm8k')['correct'])
                if 'target_kind' not in row:
                    continue
                self.assertEqual(row['target_hit'], row['predicted_answer'] == row['target_answer'])
                log = json.loads((output / row['optimization_log']).read_text())
                self.assertEqual(len(log['history']), 4)
                self.assertEqual(log['target_answer'], row['target_answer'])
                self.assertLessEqual(row['relative_edit_norm'], 0.10001)
                self.assertEqual(len(row['target_token_ids']), row['target_token_count'])
                if row['target_kind'] == 'gold':
                    self.assertTrue(row['gold_parent_replay_passed'] and row['gold_search_matches_parent'])
                    diagnostic = json.loads((output / row['gold_parent_comparison_path']).read_text())
                    self.assertTrue(diagnostic['fixed_input_replay']['matches'])
                    self.assertTrue(diagnostic['independent_search']['matches'])
                    self.assertIsNone(diagnostic['search_history']['first_exact_difference'])
                    self.assertEqual(log['history'], gold_histories[(row['sample_id'], row['restart_seed'])])
                    self.assertEqual(row['target_hit'], row['correct'])
                elif row['sample_id'] == samples[0]['sample_id']:
                    self.assertTrue(row['correct'])
                    self.assertFalse(row['target_hit'])
            result = subprocess.run([sys.executable, str(ROOT / 'analyze_first_round.py'), '--run-dir', str(output),
                '--output-dir', str(root / 'analysis')], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for name, checksum in json.loads((output / 'checksums.json').read_text()).items():
                self.assertEqual(digest(output / name), checksum)
            # A different valid parent input must replay strictly, while the fresh
            # search remains an independently scored candidate, without filtering.
            changed_row = next(r for r in prior if r['operation'] == 'optimized')
            original_row = changed_row.copy()
            trace_file = parent / changed_row['trace_path']
            original_bytes = trace_file.read_bytes()
            sample = samples[0]
            with torch.no_grad():
                h, state = model.forward_until_step(sample['question'] + '\n', 2)
                old_trace = torch.load(trace_file, weights_only=True)['latent_inputs']
                modified = h + 0.8 * (old_trace[:, 1] - h)
                replay = model.rollout_from_step(modified, state)
                ids = torch.tensor([model.tokenizer('### ' + sample['gold_answer'])['input_ids'] + [model.eos_token_id]])
                changed_row['teacher_nll'] = float(teacher_forward(model, state, modified, ids)[0])
                changed_row['generated_token_ids'] = replay['generated_token_ids'][0]
                torch.save({'latent_inputs': replay['inputs_embeds'][:, state['latent_lists'][0]]}, trace_file)
            (parent / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in prior))
            checksums = json.loads((parent / 'checksums.json').read_text())
            for name in ('predictions.jsonl', changed_row['trace_path']):
                checksums[name] = digest(parent / name)
            write_json(parent / 'checksums.json', checksums)
            result = subprocess.run([sys.executable, str(ROOT / 'run_target_answer_controls.py'), '--config', str(config),
                '--machine-config', str(machine), '--oracle-run', str(parent), '--run-id', 'search-drift'],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            drift = root / 'outputs/search-drift'
            drift_manifest = json.loads((drift / 'manifest.json').read_text())
            self.assertEqual((drift_manifest['record_count'], drift_manifest['gold_parent_replay_passed'],
                              drift_manifest['gold_search_differences']), (16, 4, 1))
            drift_rows = [json.loads(line) for line in (drift / 'predictions.jsonl').read_text().splitlines()]
            compared = next(r for r in drift_rows if r.get('gold_search_matches_parent') is False)
            self.assertTrue(compared['gold_parent_replay_passed'])
            self.assertEqual(compared['predicted_answer'], '20')
            self.assertTrue((drift / compared['gold_parent_comparison_path'].replace('.json', '.pt')).is_file())
            changed_row.update(original_row)
            trace_file.write_bytes(original_bytes)
            # Corrupt a parent scalar and update its checksum: a genuine replay
            # discrepancy must still stop, rather than being classified as search drift.
            next(r for r in prior if r['operation'] == 'optimized')['teacher_nll'] += 0.1
            (parent / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in prior))
            checksums = json.loads((parent / 'checksums.json').read_text())
            checksums['predictions.jsonl'] = digest(parent / 'predictions.jsonl')
            checksums[changed_row['trace_path']] = digest(trace_file)
            write_json(parent / 'checksums.json', checksums)
            result = subprocess.run([sys.executable, str(ROOT / 'run_target_answer_controls.py'), '--config', str(config),
                '--machine-config', str(machine), '--oracle-run', str(parent), '--run-id', 'bad-replay'],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            failed = root / 'outputs/bad-replay'
            diagnostic = json.loads((failed / 'logs/gold_replay_mismatch.json').read_text())
            self.assertTrue(diagnostic['fixed_input_replay']['tokens_equal'])
            self.assertFalse(diagnostic['fixed_input_replay']['teacher_nll']['close'])
            self.assertTrue((failed / 'logs/gold_replay_mismatch.pt').is_file())
            self.assertEqual(json.loads((failed / 'manifest.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()

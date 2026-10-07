"""Check frozen inputs, conditional scoring, and the complete artifact pipeline.

The CLI fixture uses a controlled numeric decoder over real tiny GPT-2 feedback
and caches. It tests reporting/calibration, not real model repair performance.
"""
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
from experiments.first_round.data import load_samples
from experiments.first_round.run import digest, provenance, stable_hash, write_json
from experiments.first_round.test_tiny_model import tiny_model
from run_counterfactual_family import evaluate_question
from run_frozen_consequences import KINDS, afternoon_family, consequence_summary, frozen_directions, frozen_sources, validate_protocol
from run_same_question_donors import runtime_info

ROOT = Path(__file__).resolve().parent


class ControlledNumericDecoder(CoconutWrapper):
    """Test-only outputs distinguish source directions and stay constant across variants."""
    def forward_until_step(self, *args, **kwargs):
        h, state = super().forward_until_step(*args, **kwargs)
        state['_test_original_h'] = h.clone()
        return h, state

    def rollout_from_step(self, h, state, *args, **kwargs):
        output = super().rollout_from_step(h, state, *args, **kwargs)
        delta = h - state['_test_original_h']
        answer = '16' if delta[0, 0] < -0.05 else '24' if delta[0, 1] > 0.05 else '20'
        output['generated_token_ids'] = [self.tokenizer(answer, add_special_tokens=False)['input_ids']]
        output['continuation'] = [answer]
        return output


class FrozenConsequenceTests(unittest.TestCase):
    def test_only_afternoon_changes_and_all_three_predictions_are_distinct(self):
        sample = {'sample_id': 'original', 'question': 'another 25 cups of feed for 20 chickens', 'gold_answer': '20'}
        protocol = {'afternoon_cups': [25, 17, 21, 29, 33]}
        rows = afternoon_family(sample, protocol)
        self.assertEqual([r['answer'].split('#### ')[-1] for r in rows], ['20', '28', '24', '16', '12'])
        for row in rows:
            self.assertEqual(row['question'], sample['question'].replace('another 25 cups of feed', f"another {row['afternoon_cups']} cups of feed"))
            for source in (20, 16, 24):
                expected = source-row['afternoon_change']
                self.assertEqual(expected == source, row['is_calibration'])
        with self.assertRaises(ValueError):
            afternoon_family({**sample, 'question': 'missing the required condition'}, protocol)
        full = json.loads((ROOT / 'configs/first_round/gsm8k_frozen_consequences.json').read_text())
        validate_protocol(full, 6)
        for changed in ({'afternoon_cups': [17, 25]}, {'afternoon_cups': [25, 25]},
                        {'afternoon_cups': [25, 41]}, {'target_step': 6}, {'random_seeds': []}):
            with self.assertRaises(ValueError):
                validate_protocol({**full, **changed}, 6)

    @torch.no_grad()
    def test_three_fixed_norms_and_natural_feedback_on_new_prefixes(self):
        sources = {kind: {'delta': torch.linspace(-0.1, 0.1, 16).view(1, 16)*(i+1)} for i, kind in enumerate(KINDS)}
        directions = frozen_directions(sources, {'random_seeds': [4000, 4001]})
        self.assertEqual(len(directions), 13)
        self.assertEqual(sum(op == 'identity' for op, _, _ in directions), 1)
        for op, _, delta in directions[1:]:
            kind = op.split('_', 1)[1]
            torch.testing.assert_close(delta.norm(), sources[kind]['delta'].norm())
        model = tiny_model()
        cfg = {'prompt_template': '{question}\n', 'num_latent_steps': 3, 'max_new_tokens': 5, 'atol': 1e-5, 'rtol': 1e-5}
        for question in ('Hi', 'Different condition'):
            outputs, base = evaluate_question(model, {'question': question}, cfg, 2, directions)
            for op, _, _, actual, _ in outputs:
                if op.startswith('frozen_'):
                    kind = op.split('_', 1)[1]
                    torch.testing.assert_close(actual[:, 1]-base[:, 1], sources[kind]['delta'])
                    self.assertTrue(torch.equal(actual[:, :1], base[:, :1]))
                    self.assertFalse(torch.equal(actual[:, 2:], base[:, 2:]))

    def test_cli_reports_constant_source_outputs_separately_from_adaptation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root, base = Path(temp), Path(temp) / 'base'
            base.mkdir()
            vocab = {c: i for i, c in enumerate(bytes_to_unicode().values())}
            vocab.update({'<|endoftext|>': 256, '20': 257, '16': 258, '24': 259})
            (base / 'vocab.json').write_text(json.dumps(vocab))
            (base / 'merges.txt').write_text('#version: 0.2\n2 0\n1 6\n2 4\n')
            GPT2Tokenizer(str(base / 'vocab.json'), str(base / 'merges.txt')).save_pretrained(base)
            torch.manual_seed(321)
            network = GPT2LMHeadModel(GPT2Config(vocab_size=260, n_positions=128, n_embd=16, n_layer=2, n_head=2,
                eos_token_id=256, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, _attn_implementation='eager'))
            with torch.no_grad():
                network.transformer.ln_f.weight.fill_(0.1)
                network.transformer.ln_f.bias.zero_()
                network.transformer.ln_f.bias[0] = 10
                network.transformer.wte.weight[257].zero_()
                network.transformer.wte.weight[257, 0] = 1
            network.save_pretrained(base)
            model = ControlledNumericDecoder()
            model.load_from_config({'base_model_name_or_path': str(base), 'device': 'cpu', 'num_latent_placeholders': 3,
                                    'generation_kwargs': {'max_new_tokens': 1}})
            checkpoint = root / 'checkpoint'
            torch.save(model.coconut_model.state_dict(), checkpoint)
            parent = root / 'outputs/targets'
            (parent / 'traces').mkdir(parents=True)
            family = [{'question': 'another 25 cups of feed for 20 chickens', 'answer': '#### 20',
                       'chicken_count': 20, 'is_calibration': True}]
            (parent / 'family.jsonl').write_text(json.dumps(family[0]) + '\n')
            cfg = {'dataset': 'gsm8k', 'split': 'synthetic_flock_family', 'dataset_path': str(parent / 'family.jsonl'),
                'checkpoint_path': str(checkpoint), 'base_model_path': str(base), 'prompt_template': '{question}\n',
                'num_latent_steps': 3, 'max_new_tokens': 1, 'seed': 0, 'max_samples': 0, 'atol': 1e-5, 'rtol': 1e-5,
                'random_strengths': [0.5], 'noise_seeds': [0], 'save_trace_samples': 1, 'device': 'cpu'}
            config, machine, protocol_path = root / 'config.json', root / 'machine.json', root / 'protocol.json'
            write_json(config, cfg)
            write_json(machine, {'device': 'cpu', 'OUTPUT_DIR': str(root / 'outputs')})
            sample = load_samples(parent / 'family.jsonl', 'gsm8k', cfg['split'])[0]
            with torch.no_grad():
                baseline = model.run_baseline(sample['question'] + '\n')
                h, state = model.forward_until_step(sample['question'] + '\n', 2)
                positions = state['latent_lists'][0]
                original = model.rollout_from_step(h, state)['inputs_embeds'][:, positions]
            baseline_path = 'traces/baseline.pt'
            torch.save({'latent_inputs': original}, parent / baseline_path)
            rows = [{'sample_id': sample['sample_id'], 'operation': 'baseline', 'trace_path': baseline_path,
                     'generated_token_ids': baseline['generated_token_ids'][0]}]
            hashes = {}
            for kind, answer, axis, value in [('gold', '20', 0, 0.1), ('minus4', '16', 0, -0.1), ('plus4', '24', 1, 0.1)]:
                delta = torch.zeros_like(h)
                delta[0, axis] = value
                with torch.no_grad():
                    output = model.rollout_from_step(h+delta, state)
                    actual = output['inputs_embeds'][:, positions]
                self.assertEqual(output['continuation'][0], answer)
                path = f'traces/{kind}.pt'
                torch.save({'sample_id': sample['sample_id'], 'baseline_latents': original, 'latent_inputs': actual,
                            'delta': actual[:, 1]-original[:, 1]}, parent / path)
                hashes[kind] = digest(parent / path)
                rows.append({'sample_id': sample['sample_id'], 'operation': 'optimized_' + kind, 'strength': 0.25,
                    'restart_seed': 0, 'step': 2, 'target_hit': True, 'target_answer': answer, 'predicted_answer': answer,
                    'trace_path': path, 'generated_token_ids': output['generated_token_ids'][0]})
            (parent / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            write_json(parent / 'samples.json', [{**sample, 'chicken_count': 20, 'is_calibration': True}])
            packages = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'numpy', 'huggingface_hub')}
            prov = provenance(cfg, packages, runtime_info(torch))
            prov['signature'] = stable_hash(prov['identity'])
            for name in prov['identity']['source_hashes']:
                path = parent / 'source' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT / name).read_bytes())
            write_json(parent / 'manifest.json', {**prov, 'stage': 'target_answer_controls', 'status': 'completed',
                                                'run_id': 'targets', 'weights_unchanged': True})
            write_json(parent / 'checksums.json', {str(p.relative_to(parent)): digest(p) for p in parent.rglob('*') if p.is_file()})
            protocol = {'source_chicken_count': 20, 'source_radius': 0.25, 'source_restart_seed': 0, 'target_step': 2,
                'afternoon_cups': [25, 21, 29], 'random_seeds': [4000, 4001], 'source_signature': prov['signature'], 'source_trace_hashes': hashes}
            write_json(protocol_path, protocol)
            frozen_sources(parent, rows, sample, protocol)
            with self.assertRaises(ValueError):
                frozen_sources(parent, rows, sample, {**protocol, 'source_trace_hashes': {**hashes, 'gold': 'bad'}})
            env = {**os.environ, 'OMP_NUM_THREADS': '1', 'HF_HOME': str(root / 'hf-cache'), 'TOKENIZERS_PARALLELISM': 'false'}
            controlled_cli = ('import common.models.coconut_model as cm; '
                'from test_frozen_consequences import ControlledNumericDecoder; '
                'cm.CoconutWrapper=ControlledNumericDecoder; import run_frozen_consequences; run_frozen_consequences.main()')
            result = subprocess.run([sys.executable, '-c', controlled_cli, '--config', str(config), '--machine-config', str(machine),
                '--target-run', str(parent), '--protocol', str(protocol_path), '--run-id', 'frozen'],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            output = root / 'outputs/frozen'
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual((manifest['status'], manifest['record_count'], manifest['calibration_edits_reproduced']), ('completed', 42, 3))
            self.assertTrue(manifest['weights_unchanged'])
            records = [json.loads(line) for line in (output / 'predictions.jsonl').read_text().splitlines()]
            self.assertEqual(consequence_summary(records), json.loads((output / 'summary.json').read_text())['by_condition'])
            for row in records:
                if row.get('branch_kind') == 'frozen' and not row['is_calibration']:
                    self.assertTrue(row['source_answer_repeated'])
                    self.assertFalse(row['condition_adapted_hit'])
                    self.assertEqual(row['condition_adapted_answer'], str(int(row['source_answer'])-row['afternoon_change']))
            self.assertTrue(any(r['correct'] and not r.get('condition_adapted_hit', True) for r in records if r.get('branch_kind') == 'frozen'))
            result = subprocess.run([sys.executable, str(ROOT / 'analyze_first_round.py'), '--run-dir', str(output),
                '--output-dir', str(root / 'analysis')], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            for name, checksum in json.loads((output / 'checksums.json').read_text()).items():
                self.assertEqual(digest(output / name), checksum)


if __name__ == '__main__':
    unittest.main()

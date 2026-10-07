"""Bounded oracle feasibility search using each question's own donor vectors.

Perturb an early feedback input, take one later donor input, restore the original
prefix, and continue naturally. Gold labels only score outcomes. All sources,
donors, reversed displacements and norm-matched random controls are retained.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

from experiments.first_round.data import load_samples, score, summarize
from experiments.first_round.run import (
    ROOT, digest, get_tokens, load_config, preflight, provenance, random_edit,
    reference_forward, source_files, stable_hash, write_json,
)
from run_counterfactual_family import checked_run, delta_hash


def runtime_info(torch):
    runtime = {"python": sys.version, "platform": platform.platform(), "torch_cuda": torch.version.cuda,
               "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
               "dtype": "float32", "batch_size": 1, "tf32": False, "torch_threads": torch.get_num_threads(),
               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True) if shutil.which("nvidia-smi") else None
    runtime["driver"] = driver.stdout.strip() if driver and driver.returncode == 0 else None
    return runtime


def search_conditions(protocol, steps):
    target = protocol['target_step']
    for key in ('source_steps', 'strengths', 'noise_seeds'):
        values = protocol[key]
        if not values or len(set(values)) != len(values):
            raise ValueError(f'Expected unique, nonempty {key}')
    if (type(target) is not int or not 1 <= target <= steps
            or any(type(s) is not int or not 1 <= s < target for s in protocol['source_steps'])
            or any(not math.isfinite(a) or a <= 0 for a in protocol['strengths'])
            or any(type(s) is not int or s < 0 for s in protocol['noise_seeds'])
            or type(protocol['control_seed_start']) is not int or protocol['control_seed_start'] < 0):
        raise ValueError('Invalid source position, target position, strength or seed')
    conditions = list(product(protocol['source_steps'], protocol['strengths'], protocol['noise_seeds']))
    controls = set(range(protocol['control_seed_start'], protocol['control_seed_start'] + len(conditions)))
    if controls.intersection(protocol['noise_seeds']):
        raise ValueError('Random control seeds must be distinct from source seeds')
    return conditions


def donor_controls(h, donor, seed):
    import torch
    if h.shape != donor.shape or not torch.isfinite(donor).all():
        raise ValueError('Invalid donor vector')
    delta = donor - h
    noise = torch.randn(h.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32).to(h)
    return {'donor': donor.clone(), 'reverse': h - delta,
            'matched_random': h + noise / noise.norm() * delta.norm()}


def check_trace(torch, output, positions, base, step, inserted, cfg):
    actual = output['inputs_embeds'][:, positions].detach().clone()
    if (actual.shape != base.shape or not torch.isfinite(actual).all()
            or not torch.allclose(actual[:, :step - 1], base[:, :step - 1], atol=cfg['atol'], rtol=cfg['rtol'])
            or not torch.equal(actual[:, step - 1], inserted)):
        raise ValueError('Non-finite trajectory, changed prefix or incorrect vector insertion')
    return actual


def identity_state(model, prompt, step, positions, base, logits, baseline_ids, cfg):
    import torch
    h, state = model.forward_until_step(prompt, step)
    output = model.rollout_from_step(h.clone(), state)
    trace = check_trace(torch, output, positions, base, step, h, cfg)
    if (output['generated_token_ids'][0] != baseline_ids
            or not torch.allclose(trace, base, atol=cfg['atol'], rtol=cfg['rtol'])
            or not torch.allclose(output['logits'][:, -1:, :], logits, atol=cfg['atol'], rtol=cfg['rtol'])):
        raise ValueError(f'Identity recovery failed at step {step}')
    snapshot = (state['inputs_embeds'].clone(), [t.clone() for t in state['logits']],
                [(k.clone(), v.clone()) for k, v in model._kv_cache_to_legacy_pairs(state['past_key_values'])])
    return h, state, output, snapshot


def check_state(model, state, snapshot):
    import torch
    embeds, logits, cache = snapshot
    after = model._kv_cache_to_legacy_pairs(state['past_key_values'])
    if (not torch.equal(state['inputs_embeds'], embeds) or len(state['logits']) != len(logits)
            or not all(torch.equal(a, b) for a, b in zip(state['logits'], logits))
            or len(after) != len(cache)
            or not all(torch.equal(a, c) and torch.equal(b, d) for (a, b), (c, d) in zip(cache, after))):
        raise ValueError('Branch evaluation mutated the shared original prefix')


def per_question_summary(records):
    result = []
    for sample_id in dict.fromkeys(r['sample_id'] for r in records):
        rows = [r for r in records if r['sample_id'] == sample_id]
        baseline = next(r for r in rows if r['operation'] == 'baseline')
        fixed = [r for r in rows if r['operation'] == 'frozen_delta']
        by_kind = {kind: summarize([r for r in rows if r.get('branch_kind') == kind])
                   for kind in ('source', 'donor', 'reverse', 'matched_random')}
        result.append({'sample_id': sample_id, 'chicken_count': baseline['chicken_count'],
                       'is_calibration': baseline['is_calibration'], 'baseline_answer': baseline['predicted_answer'],
                       'baseline_correct': baseline['correct'], 'frozen_delta_answer': fixed[0]['predicted_answer'] if fixed else None,
                       'any_donor_correct': any(r['correct'] for r in rows if r.get('branch_kind') == 'donor'),
                       'by_kind': by_kind,
                       'successful_donor_conditions': [{k: r[k] for k in ('source_step', 'strength', 'noise_seed', 'source_correct', 'edit_norm', 'trace_path')}
                                                      for r in rows if r.get('branch_kind') == 'donor' and r['correct']]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--machine-config', required=True)
    parser.add_argument('--family-run', required=True)
    parser.add_argument('--protocol', default=str(ROOT / 'configs/first_round/gsm8k_same_question_donors.json'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    cfg = load_config(args)
    parent_dir = Path(args.family_run)
    parent, parent_rows = checked_run(parent_dir, 'counterfactual_family')
    if stable_hash(parent['identity']) != parent['signature']:
        raise ValueError('Invalid parent signature')
    cfg.update(dataset_path=str(parent_dir / 'family.jsonl'), split=parent['identity']['split'], max_samples=0)
    report, samples = preflight(cfg, False)
    if not report['ok'] or cfg['dataset'] != 'gsm8k':
        raise ValueError(report['errors'])
    raw = [json.loads(line) for line in (parent_dir / 'family.jsonl').read_text().splitlines()]
    for sample, row in zip(samples, raw):
        sample.update(chicken_count=row['chicken_count'], is_calibration=row['is_calibration'])
    if sum(s['is_calibration'] for s in samples) != 1:
        raise ValueError('Expected exactly one calibration question')
    parent_protocol = json.loads((parent_dir / 'protocol.json').read_text())
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    conditions = search_conditions(protocol, cfg['num_latent_steps'])
    if (protocol['target_step'] != parent_protocol['target_step'] or parent_protocol['scale'] != 1.0
            or (parent_protocol['source_step'], parent_protocol['source_strength'], parent_protocol['source_noise_seed']) not in conditions):
        raise ValueError('Search must include the fixed calibration route and the same target step')
    import torch
    from common.models.coconut_model import CoconutWrapper
    torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = runtime_info(torch)
    prov = provenance(cfg, report['packages'], runtime)
    for key, value in prov['identity'].items():
        if key != 'source_hashes' and parent['identity'][key] != value:
            raise ValueError(f'Family input/current model environment mismatch: {key}')
    for name, checksum in parent['identity']['source_hashes'].items():
        if digest(ROOT / name) != checksum or digest(parent_dir / 'source' / name) != checksum:
            raise ValueError(f'Parent source changed: {name}')
    fixed = torch.load(parent_dir / 'frozen_edit.pt', weights_only=True, map_location='cpu')['delta']
    if delta_hash(fixed) != parent['identity']['delta_values_sha256'] or not torch.isfinite(fixed).all():
        raise ValueError('Frozen comparison edit mismatch')
    parent_baselines = {r['sample_id']: r for r in parent_rows if r['operation'] == 'baseline'}
    parent_fixed = {r['sample_id']: r for r in parent_rows if r['operation'] == 'frozen_delta'}
    if len(parent_baselines) != len(samples) or len(parent_fixed) != len(samples):
        raise ValueError('Parent does not cover all question baselines and fixed edits')
    if Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        raise ValueError('run-id must be a directory name')
    folder = Path(cfg['output_root']) / args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'traces').mkdir()
    (folder / 'logs').mkdir()
    shutil.copy2(parent_dir / 'family.jsonl', folder / 'family.jsonl')
    shutil.copy2(parent_dir / 'frozen_edit.pt', folder / 'frozen_edit.pt')
    cfg['dataset_path'] = str(folder / 'family.jsonl')
    extra = [ROOT / 'run_counterfactual_family.py', ROOT / 'test_counterfactual_family.py', Path(__file__), ROOT / 'test_same_question_donors.py']
    prov['identity']['source_hashes'].update({p.name: digest(p) for p in extra})
    prov['identity'].update(search_protocol_sha256=digest(protocol_path), parent_signature=parent['signature'],
                            parent_checksums_sha256=digest(parent_dir / 'checksums.json'))
    prov['signature'] = stable_hash(prov['identity'])
    for path in [*source_files(), *extra]:
        target = folder / 'source' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write_json(folder / 'config.resolved.json', cfg)
    write_json(folder / 'protocol.json', protocol)
    write_json(folder / 'samples.json', samples)
    write_json(folder / 'environment.json', {**runtime, 'packages': report['packages']})
    expected = len(samples) * (2 + len(protocol['source_steps']) + 1 + 4 * len(conditions))
    manifest = {**prov, 'run_id': args.run_id, 'stage': 'same_question_donors', 'status': 'running',
                'family_run_id': parent['run_id'], 'expected_records': expected, 'record_count': 0,
                'command': sys.argv, 'started_utc': datetime.now(timezone.utc).isoformat(),
                'interpretation': 'Bounded oracle feasibility at one target step. Every candidate and matched-budget control is reported; not automatic repair or error-origin localization.'}
    write_json(folder / 'manifest.json', manifest)
    records, error = [], None
    target_step = protocol['target_step']
    print(f'RUN_DIR={folder}', flush=True)
    try:
        model = CoconutWrapper()
        model.load_from_config({'base_model_name_or_path': cfg['base_model_path'], 'tokenizer_name_or_path': cfg['base_model_path'],
                               'checkpoint_path': cfg['checkpoint_path'], 'strict_checkpoint': True,
                               'device': cfg['device'], 'num_latent_placeholders': cfg['num_latent_steps'],
                               'use_coconut_question_only': False, 'generation_kwargs': {'max_new_tokens': cfg['max_new_tokens']}})
        manifest['checkpoint_load_report'] = model.checkpoint_load_report
        with (folder / 'predictions.jsonl').open('w') as handle, torch.no_grad():
            for sample in samples:
                prompt = cfg['prompt_template'].format(question=sample['question'])
                tokens = get_tokens(model, prompt, cfg)
                positions = (tokens['input_ids'][0] == model.latent_token_id).nonzero().flatten()
                reference = reference_forward(model, tokens)
                base = reference.inputs_embeds[:, positions].detach().clone()
                logits = reference.logits[:, -1:, :].detach().clone()
                del reference
                baseline = model.run_baseline(prompt)
                prior = parent_baselines[sample['sample_id']]
                if baseline['generated_token_ids'][0] != prior['generated_token_ids'] or not torch.allclose(
                        baseline['inputs_embeds'][:, positions], base, atol=cfg['atol'], rtol=cfg['rtol']):
                    raise ValueError('Baseline differs from parent/independent reference')
                baseline_score = score(baseline['continuation'][0], sample, 'gsm8k')
                def emit(operation, output, actual, step, strength=0.0, seed=None, **fields):
                    trace_path = f"traces/{len(records):05d}_{operation}.pt"
                    torch.save({'sample_id': sample['sample_id'], 'baseline_latents': base.cpu(), 'latent_inputs': actual.cpu()}, folder / trace_path)
                    row = {'sample_id': sample['sample_id'], 'sample_index': sample['index'], 'status': 'ok',
                           'chicken_count': sample['chicken_count'], 'is_calibration': sample['is_calibration'],
                           'gold_answer': sample['gold_answer'], 'operation': operation, 'step': step,
                           'strength': strength, 'noise_seed': seed, 'continuation': output['continuation'][0],
                           'generated_token_ids': output['generated_token_ids'][0], **score(output['continuation'][0], sample, 'gsm8k'),
                           'trace_path': trace_path, 'latent_delta_norms': (actual - base).norm(dim=-1)[0].tolist(), **fields}
                    if operation != 'baseline':
                        row.update(baseline_correct=baseline_score['correct'], baseline_answer=baseline_score['predicted_answer'],
                                   baseline_parse_status=baseline_score['parse_status'], baseline_continuation=baseline['continuation'][0])
                    records.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    handle.flush()
                    return row
                emit('baseline', baseline, base, None)
                states = {}
                for step in [*protocol['source_steps'], target_step]:
                    h, state, identity, snapshot = identity_state(model, prompt, step, positions, base, logits, prior['generated_token_ids'], cfg)
                    states[step] = (h, state, snapshot)
                    emit('identity', identity, identity['inputs_embeds'][:, positions], step, passed=True)
                h_target, target_state, _ = states[target_step]
                fixed_output = model.rollout_from_step(h_target + fixed.to(h_target), target_state)
                fixed_trace = check_trace(torch, fixed_output, positions, base, target_step, h_target + fixed.to(h_target), cfg)
                if fixed_output['generated_token_ids'][0] != parent_fixed[sample['sample_id']]['generated_token_ids']:
                    raise ValueError('Frozen comparison edit differs from previous family run')
                emit('frozen_delta', fixed_output, fixed_trace, target_step, 1.0, edit_norm=float(fixed.norm()))
                for case, (source_step, strength, seed) in enumerate(conditions):
                    h_source, source_state, _ = states[source_step]
                    changed = random_edit(torch, h_source, strength, seed)
                    source = model.rollout_from_step(changed, source_state)
                    source_trace = check_trace(torch, source, positions, base, source_step, changed, cfg)
                    source_row = emit(f'source_random_s{source_step}', source, source_trace, source_step, strength, seed,
                                      branch_kind='source', source_step=source_step, edit_norm=float((changed - h_source).norm()))
                    donor = source_trace[:, target_step - 1].clone()
                    control_seed = protocol['control_seed_start'] + case
                    branches = donor_controls(h_target, donor, control_seed)
                    for kind, inserted in branches.items():
                        output = model.rollout_from_step(inserted, target_state)
                        actual = check_trace(torch, output, positions, base, target_step, inserted, cfg)
                        if (sample['is_calibration'] and kind == 'donor'
                                and (source_step, strength, seed) == (parent_protocol['source_step'], parent_protocol['source_strength'], parent_protocol['source_noise_seed'])):
                            if output['generated_token_ids'][0] != parent_fixed[sample['sample_id']]['generated_token_ids'] or not torch.allclose(
                                    donor - h_target, fixed.to(h_target), atol=cfg['atol'], rtol=cfg['rtol']):
                                raise ValueError('Known calibration donor route failed to reproduce')
                        emit(f'{kind}_from_s{source_step}', output, actual, target_step, strength, seed,
                             branch_kind=kind, source_step=source_step, source_trace_path=source_row['trace_path'],
                             source_correct=source_row['correct'], source_answer=source_row['predicted_answer'],
                             control_noise_seed=control_seed if kind == 'matched_random' else None,
                             edit_norm=float((inserted - h_target).norm()), donor_edit_norm=float((donor - h_target).norm()))
                    if (case + 1) % 10 == 0:
                        print(f"count={sample['chicken_count']} candidates={case + 1}/{len(conditions)} records={len(records)}/{expected}", flush=True)
                for step, (_, state, snapshot) in states.items():
                    check_state(model, state, snapshot)
                    _, _, replay, _ = identity_state(model, prompt, step, positions, base, logits, prior['generated_token_ids'], cfg)
                    # Also repeat on the reused state, whose tensors were checked above.
                    repeat = model.rollout_from_step(states[step][0].clone(), state)
                    if repeat['generated_token_ids'] != replay['generated_token_ids'] or not torch.allclose(
                            repeat['inputs_embeds'][:, positions], base, atol=cfg['atol'], rtol=cfg['rtol']):
                        raise ValueError('Identity drift after candidate sweep')
        if len(records) != expected:
            raise ValueError('Incomplete condition coverage')
        manifest['status'] = 'completed'
    except BaseException:
        error = traceback.format_exc()
        (folder / 'logs/error.log').write_text(error)
        manifest['status'] = 'failed'
        raise
    finally:
        manifest.update(record_count=len(records), ended_utc=datetime.now(timezone.utc).isoformat())
        write_json(folder / 'manifest.json', manifest)
        summary = {'status': manifest['status'], 'record_count': len(records), 'expected_records': expected,
                   'uncompleted_records': expected - len(records), 'run_error': error,
                   'candidates_per_question': len(conditions), 'per_question': per_question_summary(records),
                   'note': 'Any-success is oracle feasibility under a fixed search budget; compare donor/reverse/random budgets equally. Calibration separate from seven related variants.'}
        write_json(folder / 'summary.json', summary)
        write_json(folder / 'checksums.json', {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob('*'))
                                              if p.is_file() and p.name != 'checksums.json'})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

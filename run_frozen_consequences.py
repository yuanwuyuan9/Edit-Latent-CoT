"""Freeze three source edits; vary afternoon feed without further optimization."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from experiments.first_round.data import load_samples, score, summarize
from experiments.first_round.run import ROOT, digest, load_config, preflight, provenance, stable_hash, write_json
from run_counterfactual_family import checked_run, delta_hash, edit_directions, evaluate_question
from run_same_question_donors import runtime_info

KINDS = ('gold', 'minus4', 'plus4')


def validate_protocol(protocol, steps):
    cups, seeds = protocol['afternoon_cups'], protocol['random_seeds']
    if (protocol['source_chicken_count'] != 20 or type(protocol['target_step']) is not int
            or not 1 <= protocol['target_step'] < steps
            or not math.isfinite(protocol['source_radius']) or protocol['source_radius'] <= 0
            or type(protocol['source_restart_seed']) is not int or protocol['source_restart_seed'] < 0
            or not cups or cups[0] != 25 or len(set(cups)) != len(cups)
            or any(type(n) is not int or n < 0 or 41-n <= 0 for n in cups)
            or not seeds or len(set(seeds)) != len(seeds) or any(type(n) is not int or n < 0 for n in seeds)
            or set(protocol['source_trace_hashes']) != set(KINDS)):
        raise ValueError('Invalid frozen source or afternoon family protocol')


def afternoon_family(sample, protocol):
    marker = 'another 25 cups of feed'
    if (sample['gold_answer'] != '20' or sample['question'].count(marker) != 1
            or sample['question'].count('20 chickens') != 1):
        raise ValueError('Expected the original 20-chicken, 25-cup afternoon question')
    return [{'question': sample['question'].replace(marker, f'another {cups} cups of feed'),
             'answer': f'20 * 3 - 15 - {cups} = {45-cups}\n#### {45-cups}',
             'origin_sample_id': sample['sample_id'], 'chicken_count': 20, 'afternoon_cups': cups,
             'afternoon_change': cups-25, 'is_calibration': cups == 25}
            for cups in protocol['afternoon_cups']]


def frozen_sources(folder, rows, original, protocol):
    """Read only the three prespecified candidates; verify their saved values."""
    import torch
    baseline_rows = [r for r in rows if r['sample_id'] == original['sample_id'] and r['operation'] == 'baseline']
    if len(baseline_rows) != 1:
        raise ValueError('Expected one source baseline')
    baseline_row = baseline_rows[0]
    base = torch.load(folder / baseline_row['trace_path'], map_location='cpu', weights_only=True)['latent_inputs']
    result = {}
    for kind, offset in zip(KINDS, (0, -4, 4)):
        candidates = [r for r in rows if r['sample_id'] == original['sample_id']
            and r['operation'] == 'optimized_' + kind and r['strength'] == protocol['source_radius']
            and r['restart_seed'] == protocol['source_restart_seed']]
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one fixed {kind} candidate')
        row = candidates[0]
        if (row['step'] != protocol['target_step'] or not row['target_hit']
                or row['target_answer'] != str(20+offset) or row['predicted_answer'] != row['target_answer']
                or digest(folder / row['trace_path']) != protocol['source_trace_hashes'][kind]):
            raise ValueError(f'Frozen source metadata/hash mismatch: {kind}')
        saved = torch.load(folder / row['trace_path'], map_location='cpu', weights_only=True)
        actual = saved['latent_inputs']
        step = protocol['target_step']
        if (saved['sample_id'] != original['sample_id'] or actual.shape != base.shape
                or not torch.isfinite(actual).all() or not torch.equal(saved['baseline_latents'], base)
                or not torch.equal(actual[:, :step-1], base[:, :step-1])):
            raise ValueError(f'Invalid frozen trace: {kind}')
        delta = actual[:, step-1] - base[:, step-1]
        if not torch.equal(delta, saved['delta']) or delta.norm() == 0:
            raise ValueError(f'Invalid saved displacement: {kind}')
        result[kind] = {'row': row, 'delta': delta.float(), 'source_latents': actual,
                        'delta_sha256': delta_hash(delta)}
    return baseline_row, base, result


def frozen_directions(sources, protocol):
    directions = []
    for kind in KINDS:
        for op, seed, delta in edit_directions(sources[kind]['delta'], {'scale': 1.0, 'random_seeds': protocol['random_seeds']}):
            if op == 'identity':
                if not directions:
                    directions.append((op, seed, delta))
                continue
            label = {'frozen_delta': 'frozen', 'negative_delta': 'reverse', 'norm_matched_random': 'random'}[op]
            directions.append((label + '_' + kind, seed, delta))
    return directions


def consequence_summary(records):
    groups = []
    for op, seed in sorted({(r['operation'], r['noise_seed']) for r in records}, key=str):
        subset = [r for r in records if (r['operation'], r['noise_seed']) == (op, seed)]
        result = {'operation': op, 'noise_seed': seed}
        for label, calibration in (('calibration', True), ('new_variants', False)):
            rows = [r for r in subset if r['is_calibration'] == calibration]
            result[label] = {**summarize(rows), 'condition_adapted_hits': sum(r.get('condition_adapted_hit', False) for r in rows),
                'source_answer_repeated': sum(r.get('source_answer_repeated', False) for r in rows),
                'answers': [{'afternoon_cups': r['afternoon_cups'], 'answer': r['predicted_answer']} for r in rows]}
        groups.append(result)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--machine-config', required=True)
    parser.add_argument('--target-run', required=True)
    parser.add_argument('--protocol', default=str(ROOT / 'configs/first_round/gsm8k_frozen_consequences.json'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    cfg = load_config(args)
    parent_dir = Path(args.target_run)
    parent, prior = checked_run(parent_dir, 'target_answer_controls')
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol, cfg['num_latent_steps'])
    if (parent['signature'] != protocol['source_signature'] or stable_hash(parent['identity']) != parent['signature']
            or not parent.get('weights_unchanged')):
        raise ValueError('Frozen source provenance mismatch')
    cfg.update(dataset_path=str(parent_dir / 'family.jsonl'), split=parent['identity']['split'], max_samples=0)
    report, original_samples = preflight(cfg, False)
    if not report['ok'] or cfg['dataset'] != 'gsm8k':
        raise ValueError(report['errors'])
    metadata = json.loads((parent_dir / 'samples.json').read_text())
    original_ids = [s['sample_id'] for s in metadata if s['chicken_count'] == protocol['source_chicken_count'] and s['is_calibration']]
    if len(original_ids) != 1:
        raise ValueError('Expected one source calibration question')
    original = next(s for s in original_samples if s['sample_id'] == original_ids[0])
    raw_family = afternoon_family(original, protocol)
    import torch
    from common.models.coconut_model import CoconutWrapper
    torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = runtime_info(torch)
    original_prov = provenance(cfg, report['packages'], runtime)
    for key, value in original_prov['identity'].items():
        if key != 'source_hashes' and parent['identity'][key] != value:
            raise ValueError(f'Parent inference identity mismatch: {key}')
    for name, sha in parent['identity']['source_hashes'].items():
        if digest(ROOT / name) != sha or digest(parent_dir / 'source' / name) != sha:
            raise ValueError(f'Parent source changed: {name}')
    source_baseline, source_base, sources = frozen_sources(parent_dir, prior, original, protocol)
    directions = frozen_directions(sources, protocol)
    if Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        raise ValueError('run-id must be a directory name')
    folder = Path(cfg['output_root']) / args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    for name in ('logs', 'traces'):
        (folder / name).mkdir()
    family_path = folder / 'family.jsonl'
    family_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in raw_family))
    cfg.update(dataset_path=str(family_path), split='synthetic_afternoon_family', max_samples=0)
    samples = load_samples(family_path, 'gsm8k', cfg['split'])
    for sample, raw in zip(samples, raw_family):
        sample.update(chicken_count=20, afternoon_cups=raw['afternoon_cups'], afternoon_change=raw['afternoon_change'], is_calibration=raw['is_calibration'])
    prov = provenance(cfg, report['packages'], runtime)
    source_hashes = {**parent['identity']['source_hashes'],
        **{name: digest(ROOT / name) for name in ('run_frozen_consequences.py', 'test_frozen_consequences.py')}}
    prov['identity'].update(source_hashes=source_hashes, protocol_sha256=digest(protocol_path), parent_signature=parent['signature'],
        parent_checksums_sha256=digest(parent_dir / 'checksums.json'), frozen_delta_hashes={kind: sources[kind]['delta_sha256'] for kind in KINDS})
    prov['signature'] = stable_hash(prov['identity'])
    for name in source_hashes:
        path = folder / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    torch.save({'source_baseline_latents': source_base, 'edits': {kind: {
        'delta': sources[kind]['delta'], 'source_latents': sources[kind]['source_latents']} for kind in KINDS}}, folder / 'frozen_edits.pt')
    write_json(folder / 'frozen_sources.json', {kind: {'source_trace_path': sources[kind]['row']['trace_path'],
        'source_answer': sources[kind]['row']['target_answer'], 'delta_sha256': sources[kind]['delta_sha256'],
        'delta_norm': float(sources[kind]['delta'].norm())} for kind in KINDS})
    write_json(folder / 'protocol.json', protocol)
    write_json(folder / 'config.resolved.json', cfg)
    write_json(folder / 'samples.json', samples)
    write_json(folder / 'environment.json', {**runtime, 'packages': report['packages']})
    expected = len(samples) * (1 + len(directions))
    manifest = {**prov, 'run_id': args.run_id, 'stage': 'frozen_consequences', 'status': 'running', 'record_count': 0,
        'expected_records': expected, 'target_run_id': parent['run_id'], 'command': sys.argv,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'interpretation': 'Frozen source edits, no new-target optimization; conditional adaptation and constant source answers are distinct outcomes.'}
    write_json(folder / 'manifest.json', manifest)
    records, error = [], None
    print(f'RUN_DIR={folder}', flush=True)
    try:
        model = CoconutWrapper()
        model.load_from_config({'base_model_name_or_path': cfg['base_model_path'], 'tokenizer_name_or_path': cfg['base_model_path'],
            'checkpoint_path': cfg['checkpoint_path'], 'strict_checkpoint': True, 'device': cfg['device'],
            'num_latent_placeholders': cfg['num_latent_steps'], 'use_coconut_question_only': False,
            'generation_kwargs': {'max_new_tokens': cfg['max_new_tokens']}})
        model.base_model.requires_grad_(False)
        model.base_model.eval()
        weights = {name: delta_hash(p) for name, p in model.base_model.named_parameters()}
        manifest['checkpoint_load_report'] = model.checkpoint_load_report
        with (folder / 'predictions.jsonl').open('w') as handle:
            for sample in samples:
                outputs, base = evaluate_question(model, sample, cfg, protocol['target_step'], directions)
                baseline = outputs[0][2]
                baseline_score = score(baseline['continuation'][0], sample, 'gsm8k')
                if sample['is_calibration']:
                    if (baseline['generated_token_ids'][0] != source_baseline['generated_token_ids']
                            or not torch.allclose(base.cpu(), source_base, atol=cfg['atol'], rtol=cfg['rtol'])):
                        raise ValueError('Calibration baseline differs from source')
                    for kind in KINDS:
                        _, _, output, actual, _ = next(x for x in outputs if x[0] == 'frozen_' + kind)
                        if (output['generated_token_ids'][0] != sources[kind]['row']['generated_token_ids']
                                or not torch.allclose(actual.cpu(), sources[kind]['source_latents'], atol=cfg['atol'], rtol=cfg['rtol'])
                                or score(output['continuation'][0], sample, 'gsm8k')['predicted_answer'] != sources[kind]['row']['target_answer']):
                            raise ValueError(f'Fixed source calibration failed: {kind}')
                    manifest['calibration_edits_reproduced'] = 3
                for op, seed, output, actual, norm in outputs:
                    path = f'traces/{len(records):04d}_{op}_{seed}.pt'
                    delta = actual[:, protocol['target_step']-1] - base[:, protocol['target_step']-1]
                    torch.save({'sample_id': sample['sample_id'], 'baseline_latents': base.cpu(),
                        'latent_inputs': actual.cpu(), 'delta': delta.cpu()}, folder / path)
                    row = {'sample_id': sample['sample_id'], 'sample_index': sample['index'], 'status': 'ok',
                        'chicken_count': 20, 'afternoon_cups': sample['afternoon_cups'], 'afternoon_change': sample['afternoon_change'],
                        'is_calibration': sample['is_calibration'], 'gold_answer': sample['gold_answer'],
                        'operation': op, 'step': protocol['target_step'], 'strength': 0.0 if op in ('baseline', 'identity') else 1.0,
                        'noise_seed': seed, 'generated_token_ids': output['generated_token_ids'][0], 'continuation': output['continuation'][0],
                        **score(output['continuation'][0], sample, 'gsm8k'), 'trace_path': path, 'applied_delta_norm': norm,
                        'edit_norm': float(delta.norm()), 'relative_edit_norm': float(delta.norm()/base[:, protocol['target_step']-1].norm()),
                        'latent_delta_norms': (actual-base).norm(dim=-1)[0].tolist()}
                    if op != 'baseline':
                        row.update(baseline_correct=baseline_score['correct'], baseline_answer=baseline_score['predicted_answer'],
                                   baseline_parse_status=baseline_score['parse_status'], baseline_continuation=baseline['continuation'][0])
                    if op == 'identity':
                        row['passed'] = True
                    if op not in ('baseline', 'identity'):
                        branch, kind = op.split('_', 1)
                        source_answer = sources[kind]['row']['target_answer']
                        adapted = str(int(source_answer)-sample['afternoon_change'])
                        row.update(branch_kind=branch, direction_kind=kind, source_answer=source_answer,
                            condition_adapted_answer=adapted, condition_adapted_hit=row['predicted_answer'] == adapted,
                            source_answer_repeated=row['predicted_answer'] == source_answer,
                            frozen_delta_sha256=sources[kind]['delta_sha256'])
                    records.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    handle.flush()
                print(f"afternoon={sample['afternoon_cups']} gold={sample['gold_answer']} records={len(records)}/{expected}", flush=True)
        if len(records) != expected or any(p.grad is not None or p.requires_grad or delta_hash(p) != weights[name]
                                          for name, p in model.base_model.named_parameters()):
            raise ValueError('Incomplete results or changed weights')
        manifest.update(status='completed', weights_unchanged=True)
    except BaseException:
        error = traceback.format_exc()
        (folder / 'logs/error.log').write_text(error)
        manifest['status'] = 'failed'
        raise
    finally:
        manifest.update(record_count=len(records), ended_utc=datetime.now(timezone.utc).isoformat())
        write_json(folder / 'manifest.json', manifest)
        write_json(folder / 'summary.json', {'status': manifest['status'], 'record_count': len(records), 'expected_records': expected,
            'uncompleted_records': expected-len(records), 'run_error': error, 'by_condition': consequence_summary(records),
            'note': 'Compare frozen branch adaptation and source-answer repetition on four new related variants; correct always means true gold.'})
        write_json(folder / 'checksums.json', {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob('*'))
                                             if p.is_file() and p.name != 'checksums.json'})
    print(f'COMPLETED={folder}')


if __name__ == '__main__':
    main()

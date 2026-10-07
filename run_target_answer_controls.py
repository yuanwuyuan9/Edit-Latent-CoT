"""Compare gold and two fixed wrong targets with identical latent search budgets.

Reuse the completed oracle run's optimizer and protocol unchanged. Each search
starts from the original prefix, changes one input, and recomputes the suffix.
Target hits and true-gold correctness are separate free-generation outcomes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from experiments.first_round.data import load_samples, score
from experiments.first_round.run import (
    ROOT, digest, get_tokens, load_config, preflight, provenance, reference_forward,
    stable_hash, write_json,
)
from run_counterfactual_family import checked_run, delta_hash
from run_oracle_latent_optimization import comparison_details, projected_search, teacher_forward, validate_protocol
from run_same_question_donors import check_state, check_trace, identity_state, runtime_info


def target_cases(sample, baseline_answer, protocol):
    if (protocol != {'target_offsets': [0, -4, 4]}
            or any(type(offset) is not int for offset in protocol['target_offsets'])):
        raise ValueError('Keep the prespecified gold, minus-four and plus-four targets')
    gold = sample['gold_answer']
    if not gold.isdigit() or str(int(gold)) != gold:
        raise ValueError('Expected canonical nonnegative integer gold answer')
    cases = []
    for offset, kind in zip(protocol['target_offsets'], ('gold', 'minus4', 'plus4')):
        answer = str(int(gold) + offset)
        if int(answer) <= 0 or (offset != 0 and answer == baseline_answer):
            raise ValueError('Wrong targets must be positive and differ from baseline')
        cases.append({'target_kind': kind, 'target_offset': offset, 'target_answer': answer})
    return cases


def target_summary(records):
    questions = []
    for sid in dict.fromkeys(r['sample_id'] for r in records):
        rows = [r for r in records if r['sample_id'] == sid]
        baseline = next(r for r in rows if r['operation'] == 'baseline')
        targets = []
        for kind in ('gold', 'minus4', 'plus4'):
            trials = [r for r in rows if r.get('target_kind') == kind]
            by_radius = []
            for radius in sorted({r['strength'] for r in trials}):
                subset = [r for r in trials if r['strength'] == radius]
                by_radius.append({'relative_radius': radius, 'attempts': len(subset),
                    'target_hits': sum(r['target_hit'] for r in subset),
                    'gold_correct': sum(r['correct'] for r in subset),
                    'parse_failed': sum(r['parse_status'] != 'ok' for r in subset),
                    'answers': [r['predicted_answer'] for r in subset]})
            hits = [r['strength'] for r in trials if r['target_hit']]
            targets.append({'target_kind': kind, 'target_answer': trials[0]['target_answer'] if trials else None,
                'attempts': len(trials), 'target_hits': sum(r['target_hit'] for r in trials),
                'gold_correct': sum(r['correct'] for r in trials),
                'smallest_tested_success_radius': min(hits) if hits else None, 'by_radius': by_radius})
        questions.append({'sample_id': sid, 'chicken_count': baseline['chicken_count'],
            'is_calibration': baseline['is_calibration'], 'gold_answer': baseline['gold_answer'],
            'baseline_answer': baseline['predicted_answer'], 'targets': targets})
    return questions


def compare_gold(output, metrics, old, old_trace, positions, cfg):
    """Compare fixed-input inference or independent searches without conflating them."""
    import math
    latents = comparison_details(output['inputs_embeds'][:, positions], old_trace, cfg)
    current, expected = metrics['teacher_nll'], old['teacher_nll']
    finite = math.isfinite(current) and math.isfinite(expected)
    nll = {'actual': current, 'expected': expected, 'atol': 1e-4, 'finite': finite,
           'abs_error': abs(current-expected) if finite else None,
           'close': finite and abs(current-expected) <= 1e-4}
    tokens_equal = output['generated_token_ids'][0] == old['generated_token_ids']
    return {'matches': bool(tokens_equal and latents['close'] and nll['close']),
            'tokens_equal': tokens_equal, 'actual_token_ids': output['generated_token_ids'][0],
            'expected_token_ids': old['generated_token_ids'], 'latents': latents, 'teacher_nll': nll}


def compare_search_history(current, previous):
    """Record exact numerical divergence; this does not select or reject candidates."""
    fields = ('teacher_nll', 'relative_norm', 'gradient_norm')
    first = None
    for now, old in zip(current['history'], previous['history']):
        if any(now.get(key) != old.get(key) for key in fields):
            first = {'iteration': now['iteration'], 'actual': now, 'expected': old}
            break
    return {'actual_best_iteration': current['best_iteration'], 'expected_best_iteration': previous['best_iteration'],
            'actual_evaluations': len(current['history']), 'expected_evaluations': len(previous['history']),
            'first_exact_difference': first}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--machine-config', required=True)
    parser.add_argument('--oracle-run', required=True)
    parser.add_argument('--protocol', default=str(ROOT / 'configs/first_round/gsm8k_target_answer_controls.json'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    cfg = load_config(args)
    parent_dir = Path(args.oracle_run)
    parent, prior_rows = checked_run(parent_dir, 'oracle_latent_optimization')
    if stable_hash(parent['identity']) != parent['signature'] or not parent.get('weights_unchanged'):
        raise ValueError('Invalid completed oracle provenance')
    cfg.update(dataset_path=str(parent_dir / 'family.jsonl'), split=parent['identity']['split'], max_samples=0)
    report, samples = preflight(cfg, False)
    if not report['ok'] or cfg['dataset'] != 'gsm8k':
        raise ValueError(report['errors'])
    raw = [json.loads(line) for line in (parent_dir / 'family.jsonl').read_text().splitlines()]
    if len(raw) != len(samples):
        raise ValueError('Incomplete family coverage')
    for sample, row in zip(samples, raw):
        sample.update(chicken_count=row['chicken_count'], is_calibration=row['is_calibration'])
    search = json.loads((parent_dir / 'protocol.json').read_text())
    validate_protocol(search, cfg['num_latent_steps'])
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    baselines = {r['sample_id']: r for r in prior_rows if r['operation'] == 'baseline'}
    old_gold = {(r['sample_id'], r['strength'], r['restart_seed']): r
                for r in prior_rows if r['operation'] == 'optimized'}
    expected_gold = {(s['sample_id'], radius, seed) for s in samples
                     for radius in search['relative_radii'] for seed in search['restart_seeds']}
    if (set(baselines) != {s['sample_id'] for s in samples} or set(old_gold) != expected_gold
            or sum(r['operation'] == 'optimized' for r in prior_rows) != len(expected_gold)
            or sum(r['operation'] == 'baseline' for r in prior_rows) != len(samples)):
        raise ValueError('Incomplete parent baseline or optimized coverage')
    for sample in samples:
        target_cases(sample, baselines[sample['sample_id']]['predicted_answer'], protocol)
    import torch
    from common.models.coconut_model import CoconutWrapper
    torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = runtime_info(torch)
    prov = provenance(cfg, report['packages'], runtime)
    for key, value in prov['identity'].items():
        if key != 'source_hashes' and parent['identity'][key] != value:
            raise ValueError(f'Parent inference identity mismatch: {key}')
    for name, checksum in parent['identity']['source_hashes'].items():
        if digest(ROOT / name) != checksum or digest(parent_dir / 'source' / name) != checksum:
            raise ValueError(f'Parent source changed: {name}')
    if Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        raise ValueError('run-id must be a directory name')
    folder = Path(cfg['output_root']) / args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    for name in ('traces', 'optimization', 'logs', 'logs/gold_reproduction'):
        (folder / name).mkdir()
    shutil.copy2(parent_dir / 'family.jsonl', folder / 'family.jsonl')
    cfg['dataset_path'] = str(folder / 'family.jsonl')
    sources = {**parent['identity']['source_hashes'],
        **{name: digest(ROOT / name) for name in ('run_target_answer_controls.py', 'test_target_answer_controls.py')}}
    prov['identity'].update(source_hashes=sources, protocol_sha256=digest(protocol_path),
        search_protocol_sha256=digest(parent_dir / 'protocol.json'), parent_signature=parent['signature'],
        parent_checksums_sha256=digest(parent_dir / 'checksums.json'))
    prov['signature'] = stable_hash(prov['identity'])
    for name in sources:
        path = folder / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    write_json(folder / 'config.resolved.json', cfg)
    write_json(folder / 'protocol.json', protocol)
    write_json(folder / 'search_protocol.json', search)
    write_json(folder / 'samples.json', samples)
    write_json(folder / 'environment.json', {**runtime, 'packages': report['packages']})
    expected = len(samples) * (2 + 3 * len(search['relative_radii']) * len(search['restart_seeds']))
    manifest = {**prov, 'run_id': args.run_id, 'stage': 'target_answer_controls', 'status': 'running',
        'record_count': 0, 'expected_records': expected, 'oracle_run_id': parent['run_id'], 'command': sys.argv,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'interpretation': 'Equal-budget target reachability; target_hit is distinct from true-gold correctness.',
        'reproduction_gate': 'Fixed parent inputs must reproduce inference at original tolerances; independent search differences are recorded, not filtered.',
        'branches': 'Baseline, identity, and optimized gold/minus4/plus4; no reverse or random branches.'}
    write_json(folder / 'manifest.json', manifest)
    records, error, replay_passed, search_matched = [], None, 0, 0
    print(f'RUN_DIR={folder}', flush=True)
    try:
        model = CoconutWrapper()
        model.load_from_config({'base_model_name_or_path': cfg['base_model_path'], 'tokenizer_name_or_path': cfg['base_model_path'],
            'checkpoint_path': cfg['checkpoint_path'], 'strict_checkpoint': True, 'device': cfg['device'],
            'num_latent_placeholders': cfg['num_latent_steps'], 'use_coconut_question_only': False,
            'generation_kwargs': {'max_new_tokens': cfg['max_new_tokens']}})
        model.base_model.requires_grad_(False)
        model.base_model.eval()
        weight_checksums = {name: delta_hash(p) for name, p in model.base_model.named_parameters()}
        manifest['checkpoint_load_report'] = model.checkpoint_load_report
        with (folder / 'predictions.jsonl').open('w') as handle:
            for sample in samples:
                sid = sample['sample_id']
                prompt = cfg['prompt_template'].format(question=sample['question'])
                tokens = get_tokens(model, prompt, cfg)
                positions = (tokens['input_ids'][0] == model.latent_token_id).nonzero().flatten()
                with torch.no_grad():
                    reference = reference_forward(model, tokens)
                    base = reference.inputs_embeds[:, positions].clone()
                    first_logits = reference.logits[:, -1:, :].clone()
                    del reference
                    baseline = model.run_baseline(prompt)
                    if baseline['generated_token_ids'][0] != baselines[sid]['generated_token_ids']:
                        raise ValueError('Baseline differs from oracle parent')
                    h, state, identity, snapshot = identity_state(model, prompt, search['target_step'], positions,
                        base, first_logits, baseline['generated_token_ids'][0], cfg)
                baseline_score = score(baseline['continuation'][0], sample, 'gsm8k')
                cases = target_cases(sample, baseline_score['predicted_answer'], protocol)
                def check_teacher(inserted, output, ids, **condition):
                    with torch.no_grad():
                        loss, logits, embeds = teacher_forward(model, state, inserted, ids)
                        cached_logits = model.compute_logits(inserted, output, ids, allow_grad=False)
                    feedback = comparison_details(embeds[:, positions], output['inputs_embeds'][:, positions], cfg)
                    teacher = comparison_details(logits, cached_logits, cfg)
                    if not torch.isfinite(loss) or not feedback['close'] or not teacher['close']:
                        diagnostic = {'sample_id': sid, 'condition': condition, 'feedback_check': feedback,
                            'logits_check': teacher, 'target_ids': ids[0].tolist()}
                        write_json(folder / 'logs/teacher_mismatch.json', diagnostic)
                        torch.save({'modified_latent': inserted.detach().cpu(), 'teacher_latents': embeds[:, positions].cpu(),
                            'cached_latents': output['inputs_embeds'][:, positions].detach().cpu(), 'teacher_logits': logits.cpu(),
                            'cached_teacher_logits': cached_logits.cpu(), 'target_ids': ids.cpu()}, folder / 'logs/teacher_mismatch.pt')
                        raise ValueError('Teacher/cache mismatch: ' + json.dumps(diagnostic, allow_nan=False))
                    token_nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids.reshape(-1), reduction='none').tolist()
                    return {'teacher_nll': float(loss), 'teacher_token_nll': token_nll,
                        'feedback_max_abs_error': feedback['max_abs_error'], 'teacher_logits_max_abs_error': teacher['max_abs_error']}
                def emit(op, output, inserted, radius=0.0, seed=None, case=None, **fields):
                    actual = check_trace(torch, output, positions, base, search['target_step'], inserted, cfg)
                    delta = inserted - h
                    trace_path = f'traces/{len(records):04d}_{op}.pt'
                    torch.save({'sample_id': sid, 'baseline_latents': base.cpu(), 'latent_inputs': actual.cpu(),
                                'delta': delta.detach().cpu()}, folder / trace_path)
                    row = {'sample_id': sid, 'sample_index': sample['index'], 'status': 'ok', 'operation': op,
                        'chicken_count': sample['chicken_count'], 'is_calibration': sample['is_calibration'],
                        'gold_answer': sample['gold_answer'], 'step': search['target_step'], 'strength': radius,
                        'noise_seed': seed, 'restart_seed': seed, 'generated_token_ids': output['generated_token_ids'][0],
                        'continuation': output['continuation'][0], **score(output['continuation'][0], sample, 'gsm8k'),
                        'trace_path': trace_path, 'edit_norm': float(delta.norm()), 'relative_edit_norm': float(delta.norm()/h.norm()),
                        'latent_delta_norms': (actual-base).norm(dim=-1)[0].tolist(), **fields}
                    if case is not None:
                        row.update(case, target_hit=row['predicted_answer'] == case['target_answer'])
                    if op != 'baseline':
                        row.update(baseline_correct=baseline_score['correct'], baseline_answer=baseline_score['predicted_answer'],
                            baseline_parse_status=baseline_score['parse_status'], baseline_continuation=baseline['continuation'][0])
                    records.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    handle.flush()
                target_metadata = []
                for case in cases:
                    text = search['target_template'].format(answer=case['target_answer'])
                    ids = torch.tensor([model.tokenizer(text, add_special_tokens=False)['input_ids'] + [model.eos_token_id]], device=model.device)
                    original_metrics = check_teacher(h, identity, ids, operation='identity', **case)
                    metadata = {**case, 'target_text': text, 'target_token_ids': ids[0].tolist(),
                        'target_token_count': ids.shape[1], 'eos_appended': True, 'original_metrics': original_metrics}
                    target_metadata.append(metadata)
                    write_json(folder / 'optimization' / f"{sample['index']:03d}_targets.json", target_metadata)
                    if case['target_offset'] == 0:
                        emit('baseline', baseline, h, **original_metrics)
                        emit('identity', identity, h, **original_metrics, passed=True)
                    for radius in search['relative_radii']:
                        for seed in search['restart_seeds']:
                            reproduction = None
                            if case['target_offset'] == 0:
                                old = old_gold[(sid, radius, seed)]
                                old_trace = torch.load(parent_dir / old['trace_path'], weights_only=True, map_location=model.device)['latent_inputs']
                                if old_trace.shape != base.shape or not torch.isfinite(old_trace).all():
                                    raise ValueError('Invalid parent latent trace')
                                old_input = old_trace[:, search['target_step']-1].clone()
                                with torch.no_grad():
                                    replay = model.rollout_from_step(old_input, state)
                                    check_trace(torch, replay, positions, base, search['target_step'], old_input, cfg)
                                    replay_metrics = check_teacher(old_input, replay, ids, operation='gold_parent_replay', radius=radius, restart_seed=seed)
                                replay_check = compare_gold(replay, replay_metrics, old, old_trace, positions, cfg)
                                reproduction_path = f"logs/gold_reproduction/{sample['index']:03d}_r{radius}_seed{seed}.json"
                                reproduction = {'sample_id': sid, 'radius': radius, 'restart_seed': seed,
                                                'parent_trace_path': old['trace_path'], 'fixed_input_replay': replay_check}
                                write_json(folder / reproduction_path, reproduction)
                                if not replay_check['matches']:
                                    write_json(folder / 'logs/gold_replay_mismatch.json', reproduction)
                                    torch.save({'parent_latents': old_trace.cpu(), 'replay_latents': replay['inputs_embeds'][:, positions].detach().cpu()},
                                               folder / 'logs/gold_replay_mismatch.pt')
                                    raise ValueError('Fixed gold input replay differs from parent: ' + json.dumps(reproduction, allow_nan=False))
                                replay_passed += 1
                                check_state(model, state, snapshot)
                            delta, log = projected_search(model, state, h, ids, radius, seed, search)
                            log_path = f"optimization/{sample['index']:03d}_{case['target_kind']}_r{radius}_seed{seed}.json"
                            write_json(folder / log_path, {**case, **log})
                            inserted = h + delta
                            with torch.no_grad():
                                output = model.rollout_from_step(inserted, state)
                                metrics = check_teacher(inserted, output, ids, radius=radius, restart_seed=seed, **case)
                            if abs(metrics['teacher_nll'] - log['best_teacher_nll']) > 1e-4:
                                raise ValueError('Selected optimization loss did not reproduce')
                            if float((inserted-h).norm()/h.norm()) > radius + 1e-5:
                                raise ValueError('Evaluated edit exceeded radius')
                            fields = {}
                            if case['target_offset'] == 0:
                                comparison = compare_gold(output, metrics, old, old_trace, positions, cfg)
                                reproduction['independent_search'] = comparison
                                previous_log = json.loads((parent_dir / old['optimization_log']).read_text())
                                reproduction['search_history'] = compare_search_history(log, previous_log)
                                write_json(folder / reproduction_path, reproduction)
                                if not comparison['matches']:
                                    tensor_path = reproduction_path.replace('.json', '.pt')
                                    torch.save({'parent_latents': old_trace.cpu(), 'searched_latents': output['inputs_embeds'][:, positions].detach().cpu(),
                                                'searched_delta': delta.detach().cpu()}, folder / tensor_path)
                                    print('GOLD_SEARCH_DIFFERENCE=' + reproduction_path, flush=True)
                                search_matched += comparison['matches']
                                fields.update(gold_parent_replay_passed=True, gold_search_matches_parent=comparison['matches'],
                                              gold_parent_comparison_path=reproduction_path)
                            emit('optimized_' + case['target_kind'], output, inserted, radius, seed, case,
                                **metrics, **fields, original_teacher_nll=original_metrics['teacher_nll'],
                                target_text=text, target_token_ids=ids[0].tolist(), target_token_count=ids.shape[1], optimization_log=log_path)
                            check_state(model, state, snapshot)
                            print(f"count={sample['chicken_count']} target={case['target_answer']} radius={radius} restart={seed} records={len(records)}/{expected}", flush=True)
                    with torch.no_grad():
                        replay = model.rollout_from_step(h, state)
                        check_teacher(h, replay, ids, operation='identity_after_target', **case)
                    check_state(model, state, snapshot)
                    if replay['generated_token_ids'][0] != baseline['generated_token_ids'][0]:
                        raise ValueError('Identity drift after target search')
        if len(records) != expected or replay_passed != len(expected_gold) or any(
                p.grad is not None or p.requires_grad or delta_hash(p) != weight_checksums[name]
                for name, p in model.base_model.named_parameters()):
            raise ValueError('Incomplete coverage, reproduction, or changed model weights')
        manifest.update(status='completed', weights_unchanged=True, gold_parent_replay_passed=replay_passed,
                        gold_search_matches_parent=search_matched, gold_search_differences=replay_passed-search_matched)
    except BaseException:
        error = traceback.format_exc()
        (folder / 'logs/error.log').write_text(error)
        manifest['status'] = 'failed'
        raise
    finally:
        manifest.update(record_count=len(records), ended_utc=datetime.now(timezone.utc).isoformat())
        write_json(folder / 'manifest.json', manifest)
        write_json(folder / 'summary.json', {'status': manifest['status'], 'record_count': len(records),
            'expected_records': expected, 'uncompleted_records': expected-len(records), 'run_error': error,
            'gold_parent_replay_passed': replay_passed, 'gold_search_matches_parent': search_matched,
            'gold_search_differences': sum(r.get('gold_search_matches_parent') is False for r in records),
            'per_question': target_summary(records),
            'note': 'Compare free-generation target hits under identical search budgets; correct always means true gold.'})
        write_json(folder / 'checksums.json', {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob('*'))
                                             if p.is_file() and p.name != 'checksums.json'})
    print(f'COMPLETED={folder}')


if __name__ == '__main__':
    main()

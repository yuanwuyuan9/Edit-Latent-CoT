"""Four own-question h5 searches at the frozen edit's absolute norm budget."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from experiments.first_round.data import score
from experiments.first_round.run import ROOT, digest, get_tokens, load_config, preflight, provenance, reference_forward, stable_hash, write_json
from run_counterfactual_family import checked_run, delta_hash
from run_oracle_latent_optimization import comparison_details, projected_search, teacher_forward
from run_same_question_donors import check_state, check_trace, identity_state, runtime_info


def validate_protocol(protocol, steps):
    if (type(protocol['target_step']) is not int or not 1 <= protocol['target_step'] < steps
            or type(protocol['restart_seed']) is not int or protocol['restart_seed'] != 0
            or type(protocol['updates']) is not int or protocol['updates'] < 1
            or not 0 < protocol['step_fraction_of_radius'] <= 1
            or not 0 <= protocol['initial_fraction_of_radius'] <= 1
            or protocol['target_template'] != '### {answer}' or protocol['append_eos'] is not True
            or len(protocol['parent_signature']) != 64 or len(protocol['frozen_delta_sha256']) != 64):
        raise ValueError('Invalid single-restart, matched-budget protocol')


def matched_budget(h, fixed):
    """Use the original interface's dtype/device for both norms; never rescale fixed."""
    import torch
    fixed = fixed.to(h)
    if fixed.shape != h.shape or not torch.isfinite(h).all() or not torch.isfinite(fixed).all():
        raise ValueError('Invalid original latent or frozen direction')
    cap, norm = float(fixed.norm()), float(h.norm())
    if not math.isfinite(cap) or not math.isfinite(norm) or min(cap, norm) <= 0:
        raise ValueError('Expected finite nonzero norms')
    return cap, cap / norm


def paired_summary(records):
    questions = []
    for sid in dict.fromkeys(r['sample_id'] for r in records):
        rows = {r['operation']: r for r in records if r['sample_id'] == sid}
        baseline = rows['baseline']
        questions.append({k: baseline[k] for k in ('sample_id', 'afternoon_cups', 'is_calibration', 'gold_answer')})
        questions[-1]['branches'] = {op: {k: r[k] for k in ('predicted_answer', 'correct', 'parse_status',
            'teacher_nll', 'edit_norm', 'relative_edit_norm', 'absolute_norm_cap', 'relative_radius')}
            for op, r in rows.items() if op != 'identity'}
    searched = [r for r in records if r['operation'] == 'optimized_gold']
    frozen = [r for r in records if r['operation'] == 'frozen_gold' and not r['is_calibration']]
    return {'per_question': questions,
        'new_variant_oracle': {'correct': sum(r['correct'] for r in searched), 'attempts': len(searched)},
        'new_variant_frozen': {'correct': sum(r['correct'] for r in frozen), 'attempts': len(frozen)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--machine-config', required=True)
    parser.add_argument('--frozen-run', required=True)
    parser.add_argument('--protocol', default=str(ROOT / 'configs/first_round/gsm8k_afternoon_oracle.json'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    cfg = load_config(args)
    parent_dir = Path(args.frozen_run)
    parent, prior = checked_run(parent_dir, 'frozen_consequences')
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol, cfg['num_latent_steps'])
    if (stable_hash(parent['identity']) != parent['signature'] or parent['signature'] != protocol['parent_signature']
            or not parent.get('weights_unchanged')):
        raise ValueError('Unexpected frozen parent provenance')
    cfg.update(dataset_path=str(parent_dir / 'family.jsonl'), split=parent['identity']['split'], max_samples=0)
    report, samples = preflight(cfg, False)
    if not report['ok'] or cfg['dataset'] != 'gsm8k':
        raise ValueError(report['errors'])
    raw = [json.loads(s) for s in (parent_dir / 'family.jsonl').read_text().splitlines()]
    if len(raw) != len(samples):
        raise ValueError('Incomplete afternoon family')
    for sample, row in zip(samples, raw):
        sample.update({k: row[k] for k in ('chicken_count', 'afternoon_cups', 'afternoon_change', 'is_calibration')})
        if sample['chicken_count'] != 20 or sample['gold_answer'] != str(45-sample['afternoon_cups']):
            raise ValueError('Afternoon family gold/condition mismatch')
    old_protocol = json.loads((parent_dir / 'protocol.json').read_text())
    if (protocol['target_step'] != old_protocol['target_step']
            or [s['afternoon_cups'] for s in samples] != old_protocol['afternoon_cups']
            or sum(s['is_calibration'] for s in samples) != 1
            or any(s['is_calibration'] != (s['afternoon_cups'] == 25) for s in samples)):
        raise ValueError('Keep the complete previous family and position')
    baselines = {r['sample_id']: r for r in prior if r['operation'] == 'baseline'}
    frozen_rows = {r['sample_id']: r for r in prior if r['operation'] == 'frozen_gold'}
    ids = {s['sample_id'] for s in samples}
    if (set(baselines) != ids or set(frozen_rows) != ids
            or sum(r['operation'] in ('baseline', 'frozen_gold') for r in prior) != 2*len(samples)):
        raise ValueError('Incomplete parent baseline/frozen coverage')
    import torch
    from common.models.coconut_model import CoconutWrapper
    saved = torch.load(parent_dir / 'frozen_edits.pt', map_location='cpu', weights_only=True)
    fixed = saved['edits']['gold']['delta']
    source = saved['edits']['gold']['source_latents']
    source_base = saved['source_baseline_latents']
    step = protocol['target_step']
    if (delta_hash(fixed) != protocol['frozen_delta_sha256']
            or delta_hash(fixed) != parent['identity']['frozen_delta_hashes']['gold']
            or source.shape != source_base.shape or not torch.isfinite(source).all()
            or not torch.equal(source[:, :step-1], source_base[:, :step-1])
            or not torch.equal(source[:, step-1]-source_base[:, step-1], fixed)):
        raise ValueError('Saved source delta mismatch')
    torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prov = provenance(cfg, report['packages'], runtime_info(torch))
    for key, value in prov['identity'].items():
        if key != 'source_hashes' and parent['identity'][key] != value:
            raise ValueError(f'Parent inference identity mismatch: {key}')
    for name, sha in parent['identity']['source_hashes'].items():
        if digest(ROOT / name) != sha or digest(parent_dir / 'source' / name) != sha:
            raise ValueError(f'Parent source changed: {name}')
    if Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        raise ValueError('run-id must be a directory name')
    folder = Path(cfg['output_root']) / args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    for name in ('traces', 'optimization', 'logs'):
        (folder / name).mkdir()
    for name in ('family.jsonl', 'frozen_edits.pt'):
        shutil.copy2(parent_dir / name, folder / name)
    cfg['dataset_path'] = str(folder / 'family.jsonl')
    hashes = {**parent['identity']['source_hashes'],
        **{name: digest(ROOT / name) for name in ('run_afternoon_oracle.py', 'test_afternoon_oracle.py')}}
    prov['identity'].update(source_hashes=hashes, protocol_sha256=digest(protocol_path),
        parent_signature=parent['signature'], parent_checksums_sha256=digest(parent_dir / 'checksums.json'))
    prov['signature'] = stable_hash(prov['identity'])
    for name in hashes:
        path = folder / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    write_json(folder / 'config.resolved.json', cfg)
    write_json(folder / 'protocol.json', protocol)
    write_json(folder / 'samples.json', samples)
    write_json(folder / 'environment.json', {**prov['identity']['runtime'], 'packages': report['packages']})
    searches = sum(not s['is_calibration'] for s in samples)
    expected = 3*len(samples)+searches
    manifest = {**prov, 'run_id': args.run_id, 'stage': 'afternoon_oracle', 'status': 'running',
        'record_count': 0, 'expected_records': expected, 'expected_searches': searches,
        'frozen_run_id': parent['run_id'], 'command': sys.argv, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'interpretation': 'Own-question gold-guided reachability at the frozen delta absolute norm cap; calibration has no search.'}
    write_json(folder / 'manifest.json', manifest)
    records, error, completed_searches = [], None, 0
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
                sid = sample['sample_id']
                prompt = cfg['prompt_template'].format(question=sample['question'])
                tokens = get_tokens(model, prompt, cfg)
                positions = (tokens['input_ids'][0] == model.latent_token_id).nonzero().flatten()
                with torch.no_grad():
                    reference = reference_forward(model, tokens)
                    base, logits = reference.inputs_embeds[:, positions].clone(), reference.logits[:, -1:, :].clone()
                    del reference
                    baseline = model.run_baseline(prompt)
                    h, state, identity, snapshot = identity_state(model, prompt, step, positions, base, logits,
                        baseline['generated_token_ids'][0], cfg)
                cap, radius = matched_budget(h, fixed)
                text = protocol['target_template'].format(answer=sample['gold_answer'])
                target_ids = torch.tensor([model.tokenizer(text, add_special_tokens=False)['input_ids']+[model.eos_token_id]], device=model.device)
                write_json(folder / 'optimization' / f"{sample['index']:03d}_target.json",
                    {'text': text, 'token_ids': target_ids[0].tolist(), 'eos_appended': True,
                     'absolute_norm_cap': cap, 'relative_radius': radius, 'search_enabled': not sample['is_calibration']})
                baseline_score = score(baseline['continuation'][0], sample, 'gsm8k')
                def metrics(inserted, output, op):
                    with torch.no_grad():
                        loss, teacher_logits, embeds = teacher_forward(model, state, inserted, target_ids)
                        cached = model.compute_logits(inserted, output, target_ids, allow_grad=False)
                    feedback = comparison_details(embeds[:, positions], output['inputs_embeds'][:, positions], cfg)
                    teacher = comparison_details(teacher_logits, cached, cfg)
                    if not torch.isfinite(loss) or not feedback['close'] or not teacher['close']:
                        write_json(folder / 'logs/teacher_mismatch.json', {'sample_id': sid, 'operation': op,
                            'feedback_check': feedback, 'logits_check': teacher})
                        torch.save({'modified_latent': inserted.detach().cpu(), 'teacher_latents': embeds[:, positions].cpu(),
                            'cached_latents': output['inputs_embeds'][:, positions].detach().cpu(), 'teacher_logits': teacher_logits.cpu(),
                            'cached_teacher_logits': cached.cpu(), 'target_ids': target_ids.cpu()}, folder / 'logs/teacher_mismatch.pt')
                        raise ValueError('Differentiable feedback/teacher logits differ from cached inference')
                    return {'teacher_nll': float(loss), 'teacher_token_nll': torch.nn.functional.cross_entropy(
                        teacher_logits.reshape(-1, teacher_logits.shape[-1]), target_ids.reshape(-1), reduction='none').tolist(),
                        'feedback_max_abs_error': feedback['max_abs_error'], 'teacher_logits_max_abs_error': teacher['max_abs_error']}
                def emit(op, output, inserted, **fields):
                    actual = check_trace(torch, output, positions, base, step, inserted, cfg)
                    displacement = (inserted-h).detach()
                    norm = float(displacement.norm())
                    if norm > cap+1e-5:
                        raise ValueError('Evaluated input exceeded the absolute norm cap')
                    replay = None
                    if op in ('baseline', 'frozen_gold'):
                        old = baselines[sid] if op == 'baseline' else frozen_rows[sid]
                        old_trace = torch.load(parent_dir / old['trace_path'], map_location=model.device, weights_only=True)['latent_inputs']
                        replay = comparison_details(actual, old_trace, cfg)
                        replay['tokens_equal'] = output['generated_token_ids'][0] == old['generated_token_ids']
                        replay['parent_trace_path'] = old['trace_path']
                        if not replay['close'] or not replay['tokens_equal']:
                            write_json(folder / 'logs/parent_replay_mismatch.json', {'sample_id': sid, 'operation': op, **replay})
                            torch.save({'parent_latents': old_trace.cpu(), 'replay_latents': actual.cpu()}, folder / 'logs/parent_replay_mismatch.pt')
                            raise ValueError('Fixed baseline/frozen replay differs from parent')
                    path = f'traces/{len(records):04d}_{op}.pt'
                    torch.save({'sample_id': sid, 'baseline_latents': base.cpu(), 'original_h': h.cpu(),
                                'latent_inputs': actual.cpu(), 'delta': displacement.cpu()}, folder / path)
                    row = {'sample_id': sid, 'sample_index': sample['index'], 'status': 'ok', 'operation': op,
                        **{k: sample[k] for k in ('chicken_count', 'afternoon_cups', 'afternoon_change', 'is_calibration', 'gold_answer')},
                        'step': step, 'strength': 0.0 if op in ('baseline', 'identity') else radius,
                        'noise_seed': 0 if op == 'optimized_gold' else None, 'restart_seed': 0 if op == 'optimized_gold' else None,
                        'generated_token_ids': output['generated_token_ids'][0], 'continuation': output['continuation'][0],
                        **score(output['continuation'][0], sample, 'gsm8k'), 'trace_path': path,
                        'absolute_norm_cap': cap, 'relative_radius': radius, 'edit_norm': norm,
                        'relative_edit_norm': norm/float(h.norm()), 'latent_delta_norms': (actual-base).norm(dim=-1)[0].tolist(),
                        # Official baseline generation does not expose a prompt
                        # cache. Its verified identity replay supplies teacher NLL,
                        # as in the existing oracle runners.
                        **metrics(inserted, identity if op == 'baseline' else output, op), **fields}
                    if replay is not None:
                        row['parent_replay'] = replay
                    if op != 'baseline':
                        row.update(baseline_correct=baseline_score['correct'], baseline_answer=baseline_score['predicted_answer'],
                            baseline_parse_status=baseline_score['parse_status'], baseline_continuation=baseline['continuation'][0])
                    records.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    handle.flush()
                emit('baseline', baseline, h)
                emit('identity', identity, h, passed=True)
                with torch.no_grad():
                    inserted = h+fixed.to(h)
                    frozen_output = model.rollout_from_step(inserted, state)
                emit('frozen_gold', frozen_output, inserted, frozen_delta_sha256=delta_hash(fixed))
                check_state(model, state, snapshot)
                if not sample['is_calibration']:
                    delta, log = projected_search(model, state, h, target_ids, radius, 0, protocol)
                    path = f"optimization/{sample['index']:03d}_gold_seed0.json"
                    write_json(folder / path, {**log, 'absolute_norm_cap': cap, 'relative_radius': radius})
                    inserted = h+delta
                    with torch.no_grad():
                        output = model.rollout_from_step(inserted, state)
                    if abs(metrics(inserted, output, 'optimized_gold')['teacher_nll']-log['best_teacher_nll']) > 1e-4:
                        raise ValueError('Selected minimum-NLL iterate did not reproduce its loss')
                    emit('optimized_gold', output, inserted, optimization_log=path, target_text=text,
                        target_token_ids=target_ids[0].tolist(), original_teacher_nll=log['initial_teacher_nll'])
                    completed_searches += 1
                check_state(model, state, snapshot)
                with torch.no_grad():
                    replay = model.rollout_from_step(h, state)
                replay_actual = check_trace(torch, replay, positions, base, step, h, cfg)
                if (replay['generated_token_ids'] != baseline['generated_token_ids']
                        or not torch.allclose(replay_actual, base, atol=cfg['atol'], rtol=cfg['rtol'])):
                    raise ValueError('Identity drift after search')
                check_state(model, state, snapshot)
                print(f"afternoon={sample['afternoon_cups']} cap={cap:.6f} radius={radius:.6f} records={len(records)}/{expected}", flush=True)
        if len(records) != expected or completed_searches != searches or any(
                p.grad is not None or p.requires_grad or delta_hash(p) != weights[name] for name, p in model.base_model.named_parameters()):
            raise ValueError('Incomplete results/searches or changed weights')
        manifest.update(status='completed', weights_unchanged=True)
    except BaseException:
        error = traceback.format_exc()
        (folder / 'logs/error.log').write_text(error)
        manifest['status'] = 'failed'
        raise
    finally:
        manifest.update(record_count=len(records), completed_searches=completed_searches, ended_utc=datetime.now(timezone.utc).isoformat())
        write_json(folder / 'manifest.json', manifest)
        write_json(folder / 'summary.json', {'status': manifest['status'], 'record_count': len(records), 'expected_records': expected,
            'uncompleted_records': expected-len(records), 'run_error': error, **paired_summary(records),
            'note': 'Exclude calibration from transfer/reachability; oracle sees gold during optimization only. Failed search is not proof of impossibility.'})
        write_json(folder / 'checksums.json', {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob('*'))
                                             if p.is_file() and p.name != 'checksums.json'})
    print(f'COMPLETED={folder}')


if __name__ == '__main__':
    main()

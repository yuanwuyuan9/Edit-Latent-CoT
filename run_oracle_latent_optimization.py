"""Gold-guided, norm-bounded feasibility of changing one feedback input.

Full-context differentiable feedback matches the cached inference path. Model
weights and prefix input vectors are frozen. Only teacher forcing sees gold;
the selected input is evaluated by the unchanged free-generation interface.
"""
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
from experiments.first_round.run import (
    ROOT, digest, get_tokens, load_config, preflight, provenance, reference_forward,
    source_files, stable_hash, write_json,
)
from run_counterfactual_family import checked_run, delta_hash
from run_same_question_donors import check_state, check_trace, identity_state, runtime_info


def validate_protocol(protocol, steps):
    if type(protocol['target_step']) is not int or not 1 <= protocol['target_step'] <= steps:
        raise ValueError('Invalid target step')
    for key in ('relative_radii', 'restart_seeds', 'random_control_seeds'):
        if not protocol[key] or len(set(protocol[key])) != len(protocol[key]):
            raise ValueError(f'Expected unique nonempty {key}')
    if (any(not math.isfinite(r) or r <= 0 for r in protocol['relative_radii'])
            or any(type(s) is not int or s < 0 for s in protocol['restart_seeds'] + protocol['random_control_seeds'])
            or protocol['restart_seeds'][0] != 0
            or type(protocol['updates']) is not int or protocol['updates'] < 1
            or not 0 < protocol['step_fraction_of_radius'] <= 1
            or not 0 <= protocol['initial_fraction_of_radius'] <= 1
            or protocol['target_template'] != '### {answer}' or protocol['append_eos'] is not True):
        raise ValueError('Invalid optimization protocol')


def teacher_forward(model, state, modified, target_ids):
    """Recompute future feedback differentiably; gold is appended after the prompt."""
    import torch
    embeds = state['inputs_embeds'].detach().clone()
    positions = state['latent_lists'][0]
    slot = state['pass_idx']
    if modified.shape != (1, embeds.shape[-1]) or target_ids.ndim != 2 or target_ids.shape[0] != 1 or not target_ids.numel():
        raise ValueError('Expected batch-one modified latent and nonempty target IDs')
    embeds = model._inject_latents(embeds, [(0, positions[slot])], [modified[0]])
    for next_slot in range(slot + 1, len(positions)):
        end = positions[next_slot]
        output = model.base_model(inputs_embeds=embeds[:, :end],
            attention_mask=state['attention_mask'][:, :end], position_ids=state['position_ids'][:, :end],
            output_hidden_states=True, use_cache=False)
        # The input to the next latent is the preceding token's final hidden state.
        embeds = model._inject_latents(embeds, [(0, end)], [output.hidden_states[-1][0, -1]])
    prompt_length = embeds.shape[1]
    teacher_embeds = model.coconut_model.embedding(target_ids[:, :-1])
    all_embeds = torch.cat((embeds, teacher_embeds), dim=1)
    if all_embeds.shape[1] > model.base_model.config.n_positions:
        raise ValueError('Teacher-forcing context budget exceeded')
    logits = model.base_model(inputs_embeds=all_embeds, use_cache=False,
        attention_mask=torch.ones(all_embeds.shape[:2], device=all_embeds.device, dtype=torch.long),
        position_ids=torch.arange(all_embeds.shape[1], device=all_embeds.device).unsqueeze(0)).logits
    logits = logits[:, prompt_length - 1:prompt_length - 1 + target_ids.shape[1]]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))
    return loss, logits, embeds


def projected_search(model, state, h, target_ids, radius, seed, protocol):
    """Normalized projected gradient descent in u, where modified = h + ||h|| u."""
    import torch
    h = h.detach()
    norm = h.norm().detach()
    if not torch.isfinite(h).all() or norm == 0:
        raise ValueError('Invalid original latent')
    if seed == 0:
        initial = torch.zeros_like(h)
    else:
        noise = torch.randn(h.shape, generator=torch.Generator().manual_seed(seed)).to(h)
        initial = noise / noise.norm() * radius * protocol['initial_fraction_of_radius']
    u = initial.detach().requires_grad_(True)
    history, best_loss, best_u, best_iteration = [], math.inf, None, None
    for iteration in range(protocol['updates'] + 1):
        loss, _, _ = teacher_forward(model, state, h + norm * u, target_ids)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite optimization loss')
        value = float(loss.detach())
        relative_norm = float(u.detach().norm())
        if relative_norm > radius + 1e-6:
            raise ValueError('Projection violated the norm bound')
        if value < best_loss:
            best_loss, best_u, best_iteration = value, u.detach().clone(), iteration
        entry = {'iteration': iteration, 'teacher_nll': value, 'relative_norm': relative_norm}
        if iteration < protocol['updates']:
            gradient, = torch.autograd.grad(loss, u)
            if not torch.isfinite(gradient).all():
                raise ValueError('Non-finite latent gradient')
            grad_norm = gradient.norm()
            entry['gradient_norm'] = float(grad_norm)
            with torch.no_grad():
                if grad_norm > 0:
                    u -= protocol['step_fraction_of_radius'] * radius * gradient / grad_norm
                u *= min(1.0, radius / max(float(u.norm()), 1e-30))
        history.append(entry)
        del loss
    return norm * best_u, {'best_iteration': best_iteration, 'best_teacher_nll': best_loss,
                           'initial_teacher_nll': history[0]['teacher_nll'], 'history': history}


def optimization_summary(records):
    result = []
    for sid in dict.fromkeys(r['sample_id'] for r in records):
        rows = [r for r in records if r['sample_id'] == sid]
        baseline = next(r for r in rows if r['operation'] == 'baseline')
        groups = []
        for radius in sorted({r['strength'] for r in rows if r['operation'] == 'optimized'}):
            subset = [r for r in rows if r['strength'] == radius and r['operation'] not in ('baseline', 'identity', 'frozen_delta')]
            groups.append({'relative_radius': radius, 'by_branch': {op: summarize([r for r in subset if r['operation'] == op])
                          for op in ('optimized', 'reverse', 'matched_random')},
                          'any_optimized_correct': any(r['correct'] for r in subset if r['operation'] == 'optimized')})
        result.append({'sample_id': sid, 'chicken_count': baseline['chicken_count'], 'is_calibration': baseline['is_calibration'],
                       'baseline_answer': baseline['predicted_answer'], 'gold_answer': baseline['gold_answer'],
                       'any_optimized_correct': any(r['correct'] for r in rows if r['operation'] == 'optimized'), 'by_radius': groups})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--machine-config', required=True)
    parser.add_argument('--family-run', required=True)
    parser.add_argument('--protocol', default=str(ROOT / 'configs/first_round/gsm8k_oracle_latent_optimization.json'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    cfg = load_config(args)
    parent_dir = Path(args.family_run)
    parent, prior_rows = checked_run(parent_dir, 'counterfactual_family')
    if stable_hash(parent['identity']) != parent['signature']:
        raise ValueError('Invalid parent signature')
    cfg.update(dataset_path=str(parent_dir / 'family.jsonl'), split=parent['identity']['split'], max_samples=0)
    report, samples = preflight(cfg, False)
    if not report['ok'] or cfg['dataset'] != 'gsm8k':
        raise ValueError(report['errors'])
    raw = [json.loads(line) for line in (parent_dir / 'family.jsonl').read_text().splitlines()]
    for sample, row in zip(samples, raw):
        sample.update(chicken_count=row['chicken_count'], is_calibration=row['is_calibration'])
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol, cfg['num_latent_steps'])
    if protocol['target_step'] != json.loads((parent_dir / 'protocol.json').read_text())['target_step']:
        raise ValueError('Keep the previous family target position')
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
    baselines = {r['sample_id']: r for r in prior_rows if r['operation'] == 'baseline'}
    fixed_rows = {r['sample_id']: r for r in prior_rows if r['operation'] == 'frozen_delta'}
    if len(baselines) != len(samples) or len(fixed_rows) != len(samples):
        raise ValueError('Parent baseline/frozen edit coverage incomplete')
    fixed = torch.load(parent_dir / 'frozen_edit.pt', map_location='cpu', weights_only=True)['delta']
    if delta_hash(fixed) != parent['identity']['delta_values_sha256']:
        raise ValueError('Frozen edit mismatch')
    if Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        raise ValueError('run-id must be a directory name')
    folder = Path(cfg['output_root']) / args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    for name in ('traces', 'optimization', 'logs'):
        (folder / name).mkdir()
    shutil.copy2(parent_dir / 'family.jsonl', folder / 'family.jsonl')
    shutil.copy2(parent_dir / 'frozen_edit.pt', folder / 'frozen_edit.pt')
    cfg['dataset_path'] = str(folder / 'family.jsonl')
    extra = [ROOT / name for name in ('run_counterfactual_family.py', 'test_counterfactual_family.py',
             'run_same_question_donors.py', 'test_same_question_donors.py',
             'run_oracle_latent_optimization.py', 'test_oracle_latent_optimization.py')]
    prov['identity']['source_hashes'].update({p.name: digest(p) for p in extra})
    prov['identity'].update(protocol_sha256=digest(protocol_path), parent_signature=parent['signature'],
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
    attempts = len(protocol['relative_radii']) * len(protocol['restart_seeds'])
    expected = len(samples) * (3 + attempts * (2 + len(protocol['random_control_seeds'])))
    manifest = {**prov, 'run_id': args.run_id, 'stage': 'oracle_latent_optimization', 'status': 'running',
                'expected_records': expected, 'record_count': 0, 'family_run_id': parent['run_id'], 'command': sys.argv,
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'interpretation': 'Gold-guided norm-bounded reachability at one latent input; free generation determines success. Not automatic or semantic repair. Random controls compare geometry, not equal computational search budgets.'}
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
        weight_checksums = {name: delta_hash(p) for name, p in model.base_model.named_parameters()}
        manifest['checkpoint_load_report'] = model.checkpoint_load_report
        with (folder / 'predictions.jsonl').open('w') as handle:
            for sample in samples:
                prompt = cfg['prompt_template'].format(question=sample['question'])
                tokens = get_tokens(model, prompt, cfg)
                positions = (tokens['input_ids'][0] == model.latent_token_id).nonzero().flatten()
                with torch.no_grad():
                    reference = reference_forward(model, tokens)
                    base = reference.inputs_embeds[:, positions].clone()
                    first_logits = reference.logits[:, -1:, :].clone()
                    del reference
                    baseline = model.run_baseline(prompt)
                    if baseline['generated_token_ids'][0] != baselines[sample['sample_id']]['generated_token_ids']:
                        raise ValueError('Baseline differs from previous family run')
                    h, state, identity, snapshot = identity_state(model, prompt, protocol['target_step'], positions,
                        base, first_logits, baseline['generated_token_ids'][0], cfg)
                target_text = protocol['target_template'].format(answer=sample['gold_answer'])
                target_ids = torch.tensor([model.tokenizer(target_text, add_special_tokens=False)['input_ids'] + [model.eos_token_id]], device=model.device)
                write_json(folder / 'optimization' / f"{sample['index']:03d}_target.json", {'text': target_text, 'token_ids': target_ids[0].tolist(), 'eos_appended': True})
                def check_teacher(inserted, output):
                    with torch.no_grad():
                        loss, logits, embeds = teacher_forward(model, state, inserted, target_ids)
                        cached_logits = model.compute_logits(inserted, output, target_ids, allow_grad=False)
                    if (not torch.isfinite(loss) or not torch.allclose(embeds[:, positions], output['inputs_embeds'][:, positions], atol=cfg['atol'], rtol=cfg['rtol'])
                            or not torch.allclose(logits, cached_logits, atol=cfg['atol'], rtol=cfg['rtol'])):
                        raise ValueError('Differentiable feedback/teacher logits differ from cached inference')
                    token_nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                        target_ids.reshape(-1), reduction='none').tolist()
                    return {'teacher_nll': float(loss), 'teacher_token_nll': token_nll}
                original_metrics = check_teacher(h, identity)
                original_nll = original_metrics['teacher_nll']
                baseline_score = score(baseline['continuation'][0], sample, 'gsm8k')
                def emit(op, output, inserted, radius=0.0, restart=None, control_seed=None, **fields):
                    actual = check_trace(torch, output, positions, base, protocol['target_step'], inserted, cfg)
                    delta = inserted - h
                    path = f"traces/{len(records):04d}_{op}.pt"
                    torch.save({'sample_id': sample['sample_id'], 'baseline_latents': base.cpu(), 'latent_inputs': actual.cpu(), 'delta': delta.detach().cpu()}, folder / path)
                    row = {'sample_id': sample['sample_id'], 'sample_index': sample['index'], 'status': 'ok', 'operation': op,
                           'chicken_count': sample['chicken_count'], 'is_calibration': sample['is_calibration'], 'gold_answer': sample['gold_answer'],
                           'step': protocol['target_step'], 'strength': radius,
                           'noise_seed': restart if control_seed is None else restart * 1000000 + control_seed,
                           'restart_seed': restart, 'control_noise_seed': control_seed,
                           'generated_token_ids': output['generated_token_ids'][0], 'continuation': output['continuation'][0],
                           **score(output['continuation'][0], sample, 'gsm8k'), 'trace_path': path,
                           'edit_norm': float(delta.norm()), 'relative_edit_norm': float(delta.norm() / h.norm()),
                           'latent_delta_norms': (actual - base).norm(dim=-1)[0].tolist(), **fields}
                    if op != 'baseline':
                        row.update(baseline_correct=baseline_score['correct'], baseline_answer=baseline_score['predicted_answer'],
                                   baseline_parse_status=baseline_score['parse_status'], baseline_continuation=baseline['continuation'][0])
                    records.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    handle.flush()
                emit('baseline', baseline, h, **original_metrics)
                emit('identity', identity, h, **original_metrics, passed=True)
                with torch.no_grad():
                    inserted = h + fixed.to(h)
                    output = model.rollout_from_step(inserted, state)
                    if output['generated_token_ids'][0] != fixed_rows[sample['sample_id']]['generated_token_ids']:
                        raise ValueError('Frozen comparison differs from previous run')
                    emit('frozen_delta', output, inserted, **check_teacher(inserted, output))
                for radius in protocol['relative_radii']:
                    for seed in protocol['restart_seeds']:
                        delta, log = projected_search(model, state, h, target_ids, radius, seed, protocol)
                        log_path = f"optimization/{sample['index']:03d}_r{radius}_seed{seed}.json"
                        write_json(folder / log_path, log)
                        branches = [('optimized', None, h + delta), ('reverse', None, h - delta)]
                        for control_seed in protocol['random_control_seeds']:
                            noise = torch.randn(h.shape, generator=torch.Generator().manual_seed(control_seed)).to(h)
                            branches.append(('matched_random', control_seed, h + noise / noise.norm() * delta.norm()))
                        for op, control_seed, inserted in branches:
                            with torch.no_grad():
                                output = model.rollout_from_step(inserted, state)
                                metrics = check_teacher(inserted, output)
                                nll = metrics['teacher_nll']
                            if op == 'optimized' and abs(nll - log['best_teacher_nll']) > 1e-4:
                                raise ValueError('Selected optimization iterate did not reproduce its loss')
                            if float((inserted - h).norm() / h.norm()) > radius + 1e-5:
                                raise ValueError('Evaluated edit violated the radius')
                            emit(op, output, inserted, radius, seed, control_seed, **metrics,
                                 original_teacher_nll=original_nll, optimization_log=log_path,
                                 optimized_edit_norm=float(delta.norm()))
                        print(f"count={sample['chicken_count']} radius={radius} restart={seed} best_nll={log['best_teacher_nll']:.4f} records={len(records)}/{expected}", flush=True)
                        check_state(model, state, snapshot)
                with torch.no_grad():
                    replay = model.rollout_from_step(h, state)
                    check_teacher(h, replay)
                check_state(model, state, snapshot)
                if replay['generated_token_ids'][0] != baseline['generated_token_ids'][0]:
                    raise ValueError('Identity drift after optimization')
        if len(records) != expected or any(p.grad is not None or p.requires_grad or delta_hash(p) != weight_checksums[name]
                                         for name, p in model.base_model.named_parameters()):
            raise ValueError('Incomplete coverage or model parameters changed')
        manifest['weights_unchanged'] = True
        manifest['status'] = 'completed'
    except BaseException:
        error = traceback.format_exc()
        (folder / 'logs/error.log').write_text(error)
        manifest['status'] = 'failed'
        raise
    finally:
        manifest.update(record_count=len(records), ended_utc=datetime.now(timezone.utc).isoformat())
        write_json(folder / 'manifest.json', manifest)
        write_json(folder / 'summary.json', {'status': manifest['status'], 'record_count': len(records), 'expected_records': expected,
            'uncompleted_records': expected - len(records), 'run_error': error, 'per_question': optimization_summary(records),
            'note': 'Gold-guided oracle reachability; free generation scores success. Controls have equal displacement norm, not equal optimization budgets.'})
        write_json(folder / 'checksums.json', {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'checksums.json'})
    print(f"COMPLETED={folder}")


if __name__ == '__main__':
    main()

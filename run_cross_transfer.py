"""Cross-transfer five saved h5 edits across the same five afternoon questions."""
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
from run_oracle_latent_optimization import comparison_details
from run_same_question_donors import check_state, check_trace, identity_state, runtime_info


def load_sources(folder, rows, samples, protocol):
    """Read the five prespecified successful inputs, preserving saved delta values."""
    import torch
    step = protocol['target_step']-1
    original = torch.load(folder / 'frozen_edits.pt', map_location='cpu', weights_only=True)['edits']['gold']['delta']
    sources = {}
    for sample in samples:
        cups = sample['afternoon_cups']
        op = 'frozen_gold' if sample['is_calibration'] else 'optimized_gold'
        candidates = [r for r in rows if r['sample_id'] == sample['sample_id'] and r['operation'] == op]
        if len(candidates) != 1:
            raise ValueError(f'Expected one prespecified source: {cups}')
        row = candidates[0]
        if (row['step'] != step+1 or not row['correct'] or row['gold_answer'] != sample['gold_answer']
                or row['predicted_answer'] != sample['gold_answer']
                or (op == 'optimized_gold' and row['restart_seed'] != 0)
                or digest(folder / row['trace_path']) != protocol['source_trace_hashes'][str(cups)]):
            raise ValueError(f'Source result or trace hash mismatch: {cups}')
        saved = torch.load(folder / row['trace_path'], map_location='cpu', weights_only=True)
        actual, base, h = saved['latent_inputs'], saved['baseline_latents'], saved['original_h']
        delta = original.clone() if sample['is_calibration'] else saved['delta'].clone()
        cap = row['absolute_norm_cap']
        if (saved['sample_id'] != sample['sample_id'] or actual.shape != base.shape
                or actual.ndim != 3 or actual.shape[0] != 1
                or h.shape != (1, actual.shape[-1]) or delta.shape != h.shape
                or any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in (actual, base, h, delta))
                or not torch.equal(actual[:, :step], base[:, :step])
                or not torch.equal(saved['delta'], actual[:, step]-h)
                or not torch.equal(actual[:, step], h+delta)
                or delta_hash(delta) != protocol['source_delta_hashes'][str(cups)]
                or not math.isfinite(cap) or cap <= 0 or not 0 < float(delta.norm()) <= cap+1e-5):
            raise ValueError(f'Invalid source values: {cups}')
        sources[cups] = {'row': row, 'delta': delta, 'saved': saved,
            'delta_sha256': delta_hash(delta), 'absolute_norm_cap': cap}
    if len({s['absolute_norm_cap'] for s in sources.values()}) != 1:
        raise ValueError('Expected the same source norm cap')
    return sources


def transfer_summary(records, cups):
    cells = [r for r in records if r['operation'].startswith('transfer_from_')]
    def counts(rows):
        return {'records': len(rows), 'recipient_correct': sum(r['correct'] for r in rows),
            'source_answer_repeated': sum(r['source_answer_repeated'] for r in rows),
            'same_baseline_tokens': sum(r['same_baseline_tokens'] for r in rows),
            'parse_failed': sum(r['parse_status'] != 'ok' for r in rows)}
    index = {(r['afternoon_cups'], r['source_afternoon_cups']): r for r in cells}
    matrix = []
    for recipient in cups:
        row = []
        for source in cups:
            cell = index.get((recipient, source))
            row.append(None if cell is None else {k: cell[k] for k in ('predicted_answer', 'correct',
                'source_answer_repeated', 'same_baseline_tokens', 'is_diagonal', 'previously_evaluated')})
        matrix.append(row)
    return {'recipient_afternoon_cups': cups, 'source_afternoon_cups': cups, 'matrix': matrix,
        'diagonal': counts([r for r in cells if r['is_diagonal']]),
        'off_diagonal': counts([r for r in cells if not r['is_diagonal']]),
        'new_combinations': counts([r for r in cells if not r['previously_evaluated']]),
        'by_source_off_diagonal': [{'source_afternoon_cups': c, **counts([r for r in cells
            if r['source_afternoon_cups'] == c and not r['is_diagonal']])} for c in cups]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--machine-config', required=True)
    parser.add_argument('--oracle-run', required=True)
    parser.add_argument('--protocol', default=str(ROOT / 'configs/first_round/gsm8k_cross_transfer.json'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    cfg = load_config(args)
    parent_dir = Path(args.oracle_run)
    parent, prior = checked_run(parent_dir, 'afternoon_oracle')
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    cups, step = protocol['afternoon_cups'], protocol['target_step']
    if (type(step) is not int or not 1 <= step < cfg['num_latent_steps'] or cups != [25, 17, 21, 29, 33]
            or set(protocol['source_trace_hashes']) != {str(c) for c in cups}
            or set(protocol['source_delta_hashes']) != {str(c) for c in cups}
            or stable_hash(parent['identity']) != parent['signature'] or parent['signature'] != protocol['parent_signature']
            or not parent.get('weights_unchanged') or parent['completed_searches'] != 4):
        raise ValueError('Unexpected parent, family, position, or fixed source protocol')
    cfg.update(dataset_path=str(parent_dir / 'family.jsonl'), split=parent['identity']['split'], max_samples=0)
    report, samples = preflight(cfg, False)
    if not report['ok'] or cfg['dataset'] != 'gsm8k':
        raise ValueError(report['errors'])
    raw = [json.loads(line) for line in (parent_dir / 'family.jsonl').read_text().splitlines()]
    if len(raw) != len(samples) or [r['afternoon_cups'] for r in raw] != cups:
        raise ValueError('Keep all five previous questions in the same order')
    for sample, row in zip(samples, raw):
        sample.update({k: row[k] for k in ('chicken_count', 'afternoon_cups', 'afternoon_change', 'is_calibration')})
        if (sample['chicken_count'] != 20 or sample['is_calibration'] != (sample['afternoon_cups'] == 25)
                or sample['gold_answer'] != str(45-sample['afternoon_cups'])):
            raise ValueError('Family condition/gold mismatch')
    if step != json.loads((parent_dir / 'protocol.json').read_text())['target_step']:
        raise ValueError('Keep the previous latent position')
    baselines = {r['sample_id']: r for r in prior if r['operation'] == 'baseline'}
    old_frozen = {r['sample_id']: r for r in prior if r['operation'] == 'frozen_gold'}
    ids = {s['sample_id'] for s in samples}
    if (set(baselines) != ids or set(old_frozen) != ids
            or sum(r['operation'] in ('baseline', 'frozen_gold') for r in prior) != 2*len(samples)):
        raise ValueError('Incomplete parent baseline/frozen coverage')
    import torch
    from common.models.coconut_model import CoconutWrapper
    sources = load_sources(parent_dir, prior, samples, protocol)
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
    for name in ('traces', 'sources', 'logs'):
        (folder / name).mkdir()
    shutil.copy2(parent_dir / 'family.jsonl', folder / 'family.jsonl')
    cfg['dataset_path'] = str(folder / 'family.jsonl')
    hashes = {**parent['identity']['source_hashes'], 'run_cross_transfer.py': digest(ROOT / 'run_cross_transfer.py')}
    prov['identity'].update(source_hashes=hashes, protocol_sha256=digest(protocol_path),
        parent_signature=parent['signature'], parent_checksums_sha256=digest(parent_dir / 'checksums.json'),
        frozen_delta_hashes={str(c): s['delta_sha256'] for c, s in sources.items()})
    prov['signature'] = stable_hash(prov['identity'])
    for name in hashes:
        path = folder / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    for c, source in sources.items():
        shutil.copy2(parent_dir / source['row']['trace_path'], folder / 'sources' / f'{c}.pt')
    torch.save({'deltas': {str(c): s['delta'] for c, s in sources.items()}}, folder / 'frozen_deltas.pt')
    write_json(folder / 'sources.json', {str(c): {'sample_id': s['row']['sample_id'], 'source_answer': s['row']['gold_answer'],
        'parent_trace_path': s['row']['trace_path'], 'trace_sha256': protocol['source_trace_hashes'][str(c)],
        'delta_sha256': s['delta_sha256'], 'delta_norm_cpu': float(s['delta'].norm()),
        'absolute_norm_cap': s['absolute_norm_cap']} for c, s in sources.items()})
    write_json(folder / 'config.resolved.json', cfg)
    write_json(folder / 'protocol.json', protocol)
    write_json(folder / 'samples.json', samples)
    write_json(folder / 'environment.json', {**prov['identity']['runtime'], 'packages': report['packages']})
    expected = len(samples)*(2+len(sources))
    manifest = {**prov, 'run_id': args.run_id, 'stage': 'cross_transfer', 'status': 'running',
        'record_count': 0, 'expected_records': expected, 'oracle_run_id': parent['run_id'], 'command': sys.argv,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'interpretation': 'Frozen additive deltas; recipient gold, source answer repetition, and unchanged baseline are distinct outcomes. No new optimization.'}
    write_json(folder / 'manifest.json', manifest)
    records, error, replay_count = [], None, 0
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
        with torch.no_grad(), (folder / 'predictions.jsonl').open('w') as handle:
            for sample in samples:
                sid, recipient = sample['sample_id'], sample['afternoon_cups']
                prompt = cfg['prompt_template'].format(question=sample['question'])
                tokens = get_tokens(model, prompt, cfg)
                positions = (tokens['input_ids'][0] == model.latent_token_id).nonzero().flatten()
                reference = reference_forward(model, tokens)
                base, logits = reference.inputs_embeds[:, positions].clone(), reference.logits[:, -1:, :].clone()
                del reference
                baseline = model.run_baseline(prompt)
                h, state, identity, snapshot = identity_state(model, prompt, step, positions, base, logits,
                    baseline['generated_token_ids'][0], cfg)
                if not torch.isfinite(h).all() or h.norm() == 0:
                    raise ValueError('Invalid original latent for relative norm reporting')
                baseline_score = score(baseline['continuation'][0], sample, 'gsm8k')
                def emit(op, output, inserted, source_cups=None):
                    nonlocal replay_count
                    actual = check_trace(torch, output, positions, base, step, inserted, cfg)
                    delta = inserted-h
                    source = sources.get(source_cups)
                    expected_row = (baselines[sid] if op == 'baseline' else source['row'] if source_cups == recipient
                                    else old_frozen[sid] if source_cups == 25 else None)
                    replay_check = None
                    if expected_row is not None:
                        saved = torch.load(parent_dir / expected_row['trace_path'], map_location=model.device, weights_only=True)['latent_inputs']
                        replay_check = comparison_details(actual, saved, cfg)
                        replay_check.update(tokens_equal=output['generated_token_ids'][0] == expected_row['generated_token_ids'],
                                            parent_trace_path=expected_row['trace_path'])
                        if not replay_check['close'] or not replay_check['tokens_equal']:
                            write_json(folder / 'logs/replay_mismatch.json', {'sample_id': sid, 'operation': op, **replay_check})
                            torch.save({'parent_latents': saved.cpu(), 'replay_latents': actual.cpu()}, folder / 'logs/replay_mismatch.pt')
                            raise ValueError('Source or previous frozen replay mismatch')
                        replay_count += 1
                    path = f'traces/{len(records):04d}_{op}.pt'
                    torch.save({'sample_id': sid, 'baseline_latents': base.cpu(), 'original_h': h.cpu(),
                        'latent_inputs': actual.cpu(), 'delta': delta.cpu()}, folder / path)
                    row = {'sample_id': sid, 'sample_index': sample['index'], 'status': 'ok', 'operation': op,
                        **{k: sample[k] for k in ('chicken_count', 'afternoon_cups', 'afternoon_change', 'is_calibration', 'gold_answer')},
                        'step': step, 'strength': 0.0 if source is None else 1.0, 'noise_seed': None,
                        'generated_token_ids': output['generated_token_ids'][0], 'continuation': output['continuation'][0],
                        **score(output['continuation'][0], sample, 'gsm8k'), 'trace_path': path,
                        'edit_norm': float(delta.norm()), 'relative_edit_norm': float(delta.norm()/h.norm()),
                        'latent_delta_norms': (actual-base).norm(dim=-1)[0].tolist()}
                    if op != 'baseline':
                        row.update(baseline_correct=baseline_score['correct'], baseline_answer=baseline_score['predicted_answer'],
                            baseline_parse_status=baseline_score['parse_status'], baseline_continuation=baseline['continuation'][0])
                    if op == 'identity':
                        row['passed'] = True
                    if source is not None:
                        if row['edit_norm'] > source['absolute_norm_cap']+1e-5:
                            raise ValueError('Transfer displacement exceeded original norm cap')
                        row.update(source_afternoon_cups=source_cups, source_sample_id=source['row']['sample_id'],
                            source_answer=source['row']['gold_answer'], source_delta_sha256=source['delta_sha256'],
                            source_delta_norm=float(source['delta'].to(h).norm()), absolute_norm_cap=source['absolute_norm_cap'],
                            is_diagonal=source_cups == recipient, previously_evaluated=source_cups == recipient or source_cups == 25,
                            source_answer_repeated=row['predicted_answer'] == source['row']['gold_answer'],
                            same_baseline_tokens=output['generated_token_ids'] == baseline['generated_token_ids'])
                    if replay_check is not None:
                        row['parent_replay'] = replay_check
                    records.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    handle.flush()
                emit('baseline', baseline, h)
                emit('identity', identity, h)
                # Validate this question's source before its off-diagonal inputs.
                for source_cups in [recipient]+[c for c in cups if c != recipient]:
                    inserted = h+sources[source_cups]['delta'].to(h)
                    output = model.rollout_from_step(inserted, state)
                    emit(f'transfer_from_{source_cups}', output, inserted, source_cups)
                    check_state(model, state, snapshot)
                replay = model.rollout_from_step(h, state)
                replay_actual = check_trace(torch, replay, positions, base, step, h, cfg)
                if (replay['generated_token_ids'] != baseline['generated_token_ids']
                        or not torch.allclose(replay_actual, base, atol=cfg['atol'], rtol=cfg['rtol'])):
                    raise ValueError('Identity drift after transfers')
                check_state(model, state, snapshot)
                print(f'afternoon={recipient} records={len(records)}/{expected}', flush=True)
        conditions = {(r['sample_id'], r['operation']) for r in records}
        if (len(records) != expected or len(conditions) != expected or replay_count != 14 or any(
                p.grad is not None or p.requires_grad or delta_hash(p) != weights[name] for name, p in model.base_model.named_parameters())):
            raise ValueError('Incomplete matrix, replay checks, or changed weights')
        manifest.update(status='completed', weights_unchanged=True)
    except BaseException:
        error = traceback.format_exc()
        (folder / 'logs/error.log').write_text(error)
        manifest['status'] = 'failed'
        raise
    finally:
        manifest.update(record_count=len(records), parent_replays_passed=replay_count, ended_utc=datetime.now(timezone.utc).isoformat())
        write_json(folder / 'manifest.json', manifest)
        write_json(folder / 'summary.json', {'status': manifest['status'], 'record_count': len(records), 'expected_records': expected,
            'uncompleted_records': expected-len(records), 'run_error': error, **transfer_summary(records, cups),
            'note': 'Matrix rows are recipients, columns are sources. Exclude five diagonal replays from transfer; 20 off-diagonal cells include 4 known source-25 cases and 16 new cases.'})
        write_json(folder / 'checksums.json', {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob('*'))
                                             if p.is_file() and p.name != 'checksums.json'})
    print(f'COMPLETED={folder}')


if __name__ == '__main__':
    main()

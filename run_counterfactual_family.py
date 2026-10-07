"""Test one frozen, oracle-selected edit on a controlled flock-size family.

The same additive delta is applied to each question's own latent state. Prefixes
are preserved and later feedback is recomputed. These are synthetic variants,
not an independent GSM8K benchmark or an automatic repair method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from experiments.first_round.data import load_samples, score, summarize
from experiments.first_round.run import (
    ROOT, digest, get_tokens, load_baseline, load_config, load_gate, preflight,
    provenance, reference_forward, source_files, stable_hash, write_json,
)


def family_rows(sample, protocol):
    counts = protocol["counts"]
    calibration = protocol["calibration_count"]
    if (not counts or len(set(counts)) != len(counts) or counts[0] != calibration
            or any(type(n) is not int or 3 * n - 40 < 0 for n in counts)):
        raise ValueError("Unique valid counts must start with the calibration question")
    marker = f"{calibration} chickens"
    if sample["question"].count(marker) != 1 or sample["gold_answer"] != str(3 * calibration - 40):
        raise ValueError("Original question does not match the fixed flock-size protocol")
    return [{"question": sample["question"].replace(marker, f"{n} chickens"),
             "answer": f"{n} * 3 - 15 - 25 = {3 * n - 40}\n#### {3 * n - 40}",
             "origin_sample_id": sample["sample_id"], "chicken_count": n,
             "is_calibration": n == calibration} for n in counts]


def delta_hash(delta):
    return hashlib.sha256(delta.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


def edit_directions(delta, protocol):
    """Fixed absolute norm, never scaled by a new question's latent norm."""
    import torch
    scale, seeds = protocol["scale"], protocol["random_seeds"]
    if not math.isfinite(scale) or scale <= 0 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Positive finite scale and unique nonempty random seeds required")
    delta = delta.detach().cpu().float()
    if not torch.isfinite(delta).all() or delta.norm() == 0:
        raise ValueError("Frozen delta must be finite and nonzero")
    fixed = delta * scale
    result = [("identity", None, torch.zeros_like(fixed)),
              ("frozen_delta", None, fixed), ("negative_delta", None, -fixed)]
    for seed in seeds:
        noise = torch.randn(delta.shape, generator=torch.Generator().manual_seed(seed))
        result.append(("norm_matched_random", seed, noise / noise.norm() * fixed.norm()))
    return result


def checked_run(folder, stage):
    manifest = json.loads((folder / "manifest.json").read_text())
    checksums = json.loads((folder / "checksums.json").read_text())
    if manifest["status"] != "completed" or manifest["stage"] != stage:
        raise ValueError(f"Expected completed {stage} run")
    if "predictions.jsonl" not in checksums or "manifest.json" not in checksums:
        raise ValueError("Missing result checksum coverage")
    for name, checksum in checksums.items():
        path = (folder / name).resolve()
        if folder.resolve() not in path.parents or digest(path) != checksum:
            raise ValueError(f"Modified input result: {name}")
    rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
    return manifest, rows


def frozen_source(source_dir, transplant_dir, sample, protocol):
    import torch
    source_manifest, sources = checked_run(source_dir, "interventions")
    transplant_manifest, transplants = checked_run(transplant_dir, "latent_transplants")
    if (transplant_manifest["input_run_id"] != source_manifest["run_id"]
            or transplant_manifest["input_checksums_sha256"] != digest(source_dir / "checksums.json")
            or source_manifest["signature"] != transplant_manifest["signature"]):
        raise ValueError("Transplant/source provenance mismatch")
    def matches(row):
        return (row["sample_id"] == sample["sample_id"]
                and row["strength"] == protocol["source_strength"]
                and row["noise_seed"] == protocol["source_noise_seed"])
    sources = [r for r in sources if matches(r) and r["operation"] == "random" and r["step"] == protocol["source_step"]]
    transplants = [r for r in transplants if matches(r) and r["source_step"] == protocol["source_step"]
                  and r["step"] == protocol["target_step"]]
    if len(sources) != 1 or len(transplants) != 1 or not transplants[0]["correct"]:
        raise ValueError("Expected one previously successful, fixed transplant condition")
    source, transplant = sources[0], transplants[0]
    if transplant["source_trace_path"] != source["trace_path"]:
        raise ValueError("Donor trace mismatch")
    for folder, row, key in ((source_dir, source, "source_trace_sha256"),
                             (transplant_dir, transplant, "transplant_trace_sha256")):
        if digest(folder / row["trace_path"]) != protocol[key]:
            raise ValueError(f"Frozen protocol mismatch: {key}")
    saved = torch.load(source_dir / source["trace_path"], map_location="cpu", weights_only=True)
    repaired = torch.load(transplant_dir / transplant["trace_path"], map_location="cpu", weights_only=True)
    base, donor = saved["baseline_latents"], saved["modified_latents"]
    target = protocol["target_step"]
    if (saved["sample_id"] != sample["sample_id"] or repaired["sample_id"] != sample["sample_id"]
            or not 1 <= target <= base.shape[1]
            or not torch.equal(repaired["latent_inputs"][:, :target - 1], base[:, :target - 1])
            or not torch.equal(repaired["latent_inputs"][:, target - 1], donor[:, target - 1])):
        raise ValueError("Saved transplant does not preserve the original prefix and specified donor")
    delta = (donor[:, target - 1] - base[:, target - 1]).float()
    if delta_hash(delta) != protocol["delta_values_sha256"]:
        raise ValueError("Frozen delta values mismatch")
    return source_manifest, transplant, base, donor[:, target - 1], delta


def evaluate_question(model, sample, cfg, target, directions):
    """Independent reference + identity gate + branches sharing an immutable prefix."""
    import torch
    prompt = cfg["prompt_template"].format(question=sample["question"])
    tokens = get_tokens(model, prompt, cfg)
    positions = (tokens["input_ids"][0] == model.latent_token_id).nonzero().flatten()
    with torch.no_grad():
        baseline = model.run_baseline(prompt)
        reference = reference_forward(model, tokens)
        base = reference.inputs_embeds[:, positions].detach().clone()
        ref_logits = reference.logits[:, -1:, :].detach().clone()
        del reference
        h, state = model.forward_until_step(prompt, target)
        def close(a, b):
            return a.shape == b.shape and bool(torch.isfinite(a).all() and torch.isfinite(b).all()
                                               and torch.allclose(a, b, atol=cfg["atol"], rtol=cfg["rtol"]))
        if not close(h, base[:, target - 1]) or not close(baseline["inputs_embeds"][:, positions], base):
            raise ValueError("Baseline/prefix differs from independent reference")
        before_embeds = state["inputs_embeds"].clone()
        before_logits = [t.clone() for t in state["logits"]]
        before_cache = [(k.clone(), v.clone()) for k, v in model._kv_cache_to_legacy_pairs(state["past_key_values"])]
        outputs = [("baseline", None, baseline, base, 0.0)]
        for operation, seed, delta in directions:
            change = delta.to(device=h.device, dtype=h.dtype)
            modified = h + change
            output = model.rollout_from_step(modified, state)
            actual = output["inputs_embeds"][:, positions].detach().clone()
            if (not torch.isfinite(actual).all() or not close(actual[:, :target - 1], base[:, :target - 1])
                    or not torch.equal(actual[:, target - 1], modified)):
                raise ValueError("Non-finite trajectory, prefix changed, or edit was not inserted")
            if operation == "identity" and (output["generated_token_ids"] != baseline["generated_token_ids"]
                    or not close(actual, base) or not close(output["logits"][:, -1:, :], ref_logits)):
                raise ValueError("Identity recovery failed; aborting family experiment")
            outputs.append((operation, seed, output, actual, float(change.norm())))
        replay = model.rollout_from_step(h.clone(), state)
        after_cache = model._kv_cache_to_legacy_pairs(state["past_key_values"])
        unchanged = (torch.equal(before_embeds, state["inputs_embeds"])
                     and len(before_logits) == len(state["logits"])
                     and all(torch.equal(a, b) for a, b in zip(before_logits, state["logits"]))
                     and len(before_cache) == len(after_cache)
                     and all(torch.equal(k1, k2) and torch.equal(v1, v2)
                             for (k1, v1), (k2, v2) in zip(before_cache, after_cache)))
        if not unchanged or replay["generated_token_ids"] != baseline["generated_token_ids"] or not close(replay["inputs_embeds"][:, positions], base):
            raise ValueError("Shared prefix mutated or identity drifted after edit branches")
    return outputs, base


def family_summary(records):
    result = {}
    for operation, seed in sorted({(r["operation"], r["noise_seed"]) for r in records}, key=str):
        rows = [r for r in records if (r["operation"], r["noise_seed"]) == (operation, seed)]
        groups = {}
        for label, calibration in (("calibration", True), ("held_out_variants", False)):
            subset = [r for r in rows if r["is_calibration"] == calibration]
            groups[label] = {**summarize(subset), "answers_equal_20": sum(r["predicted_answer"] == "20" for r in subset)}
        result[f"{operation}/seed={seed}"] = groups
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--machine-config", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--transplant-run", required=True)
    parser.add_argument("--protocol", default=str(ROOT / "configs/first_round/gsm8k_counterfactual_family.json"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    cfg = load_config(args)
    report, original_samples = preflight(cfg, False)
    if not report["ok"] or cfg["dataset"] != "gsm8k":
        raise ValueError(f"Requires GSM8K with passed preflight: {report['errors']}")
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    original = next(s for s in original_samples if s["index"] == protocol["origin_sample_index"])
    raw_family = family_rows(original, protocol)
    source_dir, transplant_dir = Path(args.source_run), Path(args.transplant_run)
    parent, calibration_transplant, original_latents, donor, delta = frozen_source(source_dir, transplant_dir, original, protocol)
    directions = edit_directions(delta, protocol)
    if not protocol["source_step"] < protocol["target_step"] <= cfg["num_latent_steps"]:
        raise ValueError("Expected a later transplant target within the configured latent steps")
    import torch
    from common.models.coconut_model import CoconutWrapper
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = {"python": sys.version, "platform": platform.platform(), "torch_cuda": torch.version.cuda,
               "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
               "dtype": "float32", "batch_size": 1, "tf32": False, "torch_threads": torch.get_num_threads(),
               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True) if shutil.which("nvidia-smi") else None
    runtime["driver"] = driver.stdout.strip() if driver and driver.returncode == 0 else None
    original_prov = provenance(cfg, report["packages"], runtime)
    if original_prov["signature"] != parent["signature"]:
        raise ValueError("Original source does not match current code/model/data/environment")
    baseline_manifest, baselines = load_baseline(str(source_dir.parent / parent["baseline_run_id"]), parent["signature"], [original])
    load_gate(str(source_dir.parent / parent["resume_run_id"]), parent["signature"], baseline_manifest["run_id"])
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise ValueError("run-id must be a directory name")
    folder = Path(cfg["output_root"]) / args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "logs").mkdir()
    (folder / "traces").mkdir()
    family_path = folder / "family.jsonl"
    family_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in raw_family))
    cfg = {**cfg, "dataset_path": str(family_path), "split": "synthetic_flock_family", "max_samples": len(raw_family)}
    samples = load_samples(family_path, "gsm8k", cfg["split"])
    for sample, raw in zip(samples, raw_family):
        sample.update(chicken_count=raw["chicken_count"], is_calibration=raw["is_calibration"])
    extra_sources = [Path(__file__), ROOT / "test_counterfactual_family.py"]
    prov = provenance(cfg, report["packages"], runtime)
    prov["identity"]["source_hashes"].update({p.name: digest(p) for p in extra_sources})
    prov["identity"].update(protocol_sha256=digest(protocol_path), delta_values_sha256=delta_hash(delta),
                            source_checksums_sha256=digest(source_dir / "checksums.json"),
                            transplant_checksums_sha256=digest(transplant_dir / "checksums.json"))
    prov["signature"] = stable_hash(prov["identity"])
    for path in [*source_files(), *extra_sources]:
        target = folder / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write_json(folder / "protocol.json", protocol)
    write_json(folder / "config.resolved.json", cfg)
    write_json(folder / "samples.json", samples)
    write_json(folder / "environment.json", {**runtime, "packages": report["packages"]})
    torch.save({"delta": delta, "original_latent": original_latents[:, protocol["target_step"] - 1],
                "donor": donor}, folder / "frozen_edit.pt")
    expected = len(samples) * (1 + len(directions))
    manifest = {**prov, "run_id": args.run_id, "stage": "counterfactual_family", "status": "running",
                "source_signature": parent["signature"], "source_run_id": parent["run_id"],
                "transplant_run_id": transplant_dir.name, "expected_records": expected, "record_count": 0,
                "command": sys.argv, "started_utc": datetime.now(timezone.utc).isoformat(),
                "interpretation": "Synthetic variants of one calibration question; frozen oracle-selected additive edit. Not automatic repair or an independent benchmark."}
    write_json(folder / "manifest.json", manifest)
    records, error = [], None
    print(f"RUN_DIR={folder}", flush=True)
    try:
        model = CoconutWrapper()
        model.load_from_config({"base_model_name_or_path": cfg["base_model_path"], "tokenizer_name_or_path": cfg["base_model_path"],
                               "checkpoint_path": cfg["checkpoint_path"], "strict_checkpoint": True,
                               "device": cfg["device"], "num_latent_placeholders": cfg["num_latent_steps"],
                               "use_coconut_question_only": False, "generation_kwargs": {"max_new_tokens": cfg["max_new_tokens"]}})
        manifest["checkpoint_load_report"] = model.checkpoint_load_report
        with (folder / "predictions.jsonl").open("w") as handle:
            for sample in samples:
                outputs, base = evaluate_question(model, sample, cfg, protocol["target_step"], directions)
                baseline = outputs[0][2]
                baseline_score = score(baseline["continuation"][0], sample, "gsm8k")
                if sample["is_calibration"]:
                    edited = next(o for op, _, o, _, _ in outputs if op == "frozen_delta")
                    if (baseline["generated_token_ids"][0] != baselines[original["sample_id"]]["generated_token_ids"]
                            or not torch.allclose(base.cpu(), original_latents, atol=cfg["atol"], rtol=cfg["rtol"])
                            or edited["generated_token_ids"][0] != calibration_transplant["generated_token_ids"]
                            or not score(edited["continuation"][0], sample, "gsm8k")["correct"]):
                        raise ValueError("Calibration failed to reproduce the original successful edit")
                for operation, seed, output, actual, norm in outputs:
                    trace_path = f"traces/{sample['index']:03d}_{operation}_{seed}.pt"
                    torch.save({"sample_id": sample["sample_id"], "baseline_latents": base.cpu(), "latent_inputs": actual.cpu()}, folder / trace_path)
                    row = {"sample_id": sample["sample_id"], "sample_index": sample["index"], "status": "ok",
                           "chicken_count": sample["chicken_count"], "is_calibration": sample["is_calibration"],
                           "gold_answer": sample["gold_answer"], "operation": operation, "step": protocol["target_step"],
                           "strength": 0.0 if operation in ("baseline", "identity") else protocol["scale"], "noise_seed": seed,
                           "continuation": output["continuation"][0], "generated_token_ids": output["generated_token_ids"][0],
                           **score(output["continuation"][0], sample, "gsm8k"), "trace_path": trace_path,
                           "edit_norm": norm, "latent_delta_norms": (actual - base).norm(dim=-1)[0].tolist()}
                    if operation != "baseline":
                        row.update(baseline_correct=baseline_score["correct"], baseline_answer=baseline_score["predicted_answer"],
                                   baseline_parse_status=baseline_score["parse_status"], baseline_continuation=baseline["continuation"][0])
                    records.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                print(f"count={sample['chicken_count']} gold={sample['gold_answer']} " + " ".join(
                    f"{op}/{seed}={score(out['continuation'][0], sample, 'gsm8k')['predicted_answer']}" for op, seed, out, _, _ in outputs), flush=True)
        manifest["status"] = "completed"
    except BaseException:
        error = traceback.format_exc()
        (folder / "logs/error.log").write_text(error)
        manifest["status"] = "failed"
        raise
    finally:
        manifest.update(record_count=len(records), ended_utc=datetime.now(timezone.utc).isoformat())
        write_json(folder / "manifest.json", manifest)
        summary = {"status": manifest["status"], "record_count": len(records), "expected_records": expected,
                   "uncompleted_records": expected - len(records), "run_error": error,
                   "question_count": len(samples), "held_out_variant_count": len(samples) - 1,
                   "by_condition": family_summary(records)}
        write_json(folder / "summary.json", summary)
        write_json(folder / "checksums.json", {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob("*"))
                                              if p.is_file() and p.name != "checksums.json"})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

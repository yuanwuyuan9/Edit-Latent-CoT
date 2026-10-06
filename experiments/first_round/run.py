"""Small staged Coconut experiments. Heavy imports happen after preflight."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .data import load_samples, score, summarize

ROOT = Path(__file__).resolve().parents[2]
COCONUT_COMMIT = "27273cb8cca4bb763c041a63b036d0c3b7cbbb48"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def git_info(path: Path) -> dict:
    def command(*args):
        result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": command("rev-parse", "HEAD"), "status": command("status", "--short")}


def load_config(args) -> dict:
    cfg = json.loads(Path(args.config).read_text())
    machine = {"PROJECT_ROOT": str(ROOT), "DATA_DIR": str(ROOT.parent / "data"),
               "MODEL_DIR": str(ROOT.parent / "models"), "OUTPUT_DIR": str(ROOT.parent / "outputs"), "device": "cpu"}
    if args.machine_config:
        machine.update(json.loads(Path(args.machine_config).read_text()))
    for key, value in machine.items():
        if key != "device" and isinstance(value, str) and not Path(value).is_absolute():
            machine[key] = str((ROOT / value).resolve())
    for key in ("dataset_path", "checkpoint_path", "base_model_path"):
        for name, value in machine.items():
            cfg[key] = cfg[key].replace("${" + name + "}", str(value))
        if "${" in cfg[key]:
            raise ValueError(f"Unresolved path variable in {key}")
        path = Path(cfg[key]).expanduser()
        cfg[key] = str(path if path.is_absolute() else (ROOT / path).resolve())
    cfg["device"] = machine["device"]
    cfg["output_root"] = machine["OUTPUT_DIR"]
    if args.max_samples is not None:
        cfg["max_samples"] = args.max_samples
    if args.checkpoint:
        cfg["checkpoint_path"] = str(Path(args.checkpoint).expanduser().resolve())
    if cfg["max_samples"] < 0 or cfg["num_latent_steps"] < 1 or cfg["max_new_tokens"] < 1:
        raise ValueError("Invalid sample count, latent steps or max_new_tokens")
    for key in ("atol", "rtol"):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"Invalid {key}")
    if any(not math.isfinite(value) or value <= 0 for value in cfg["random_strengths"]):
        raise ValueError("Random strengths must be positive finite values")
    if len(set(cfg["noise_seeds"])) != len(cfg["noise_seeds"]) or not cfg["noise_seeds"]:
        raise ValueError("noise_seeds must be nonempty and unique")
    return cfg


def preflight(cfg: dict, data_only: bool) -> tuple[dict, list[dict]]:
    samples = load_samples(Path(cfg["dataset_path"]), cfg["dataset"], cfg["split"])
    selected = samples[:cfg["max_samples"]] if cfg["max_samples"] else samples
    errors = []
    packages = {}
    for name in ("torch", "transformers", "numpy", "huggingface_hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    if not data_only:
        for name, version in packages.items():
            if version is None:
                errors.append(f"Missing package: {name}")
        if packages["transformers"] and packages["transformers"] != "4.44.2":
            errors.append("First round requires transformers==4.44.2 for the pinned official Coconut cache API")
        external = ROOT / "external/coconut"
        if not (external / "coconut.py").is_file():
            errors.append("Missing external/coconut/coconut.py; see FIRST_ROUND_RUNBOOK.md")
        elif git_info(external)["commit"] != COCONUT_COMMIT:
            errors.append(f"external/coconut must use commit {COCONUT_COMMIT}")
        if not Path(cfg["checkpoint_path"]).is_file():
            errors.append(f"Missing checkpoint: {cfg['checkpoint_path']}")
        base = Path(cfg["base_model_path"])
        if not (base / "config.json").is_file() or not any((base / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
            errors.append(f"Incomplete GPT-2 base model directory: {base}")
        if not (base / "vocab.json").is_file() or not (base / "merges.txt").is_file():
            errors.append(f"Missing GPT-2 tokenizer vocab/merges: {base}")
    return {"ok": not errors, "data_only": data_only, "errors": errors,
            "dataset": cfg["dataset"], "total_samples": len(samples), "selected_samples": len(selected),
            "sample_ids": [row["sample_id"] for row in selected], "packages": packages,
            "config": cfg}, selected


def source_files() -> list[Path]:
    paths = list((ROOT / "experiments/first_round").glob("*.py"))
    paths += [ROOT / name for name in (
        "common/models/coconut_model.py", "common/model_interface.py", "common/model_registry.py",
        "common/path_utils.py", "external/coconut/coconut.py", "requirements-first-round.txt",
        "run_baseline.py", "check_resume.py", "run_interventions.py", "analyze_first_round.py")]
    return sorted(path for path in paths if path.is_file())


def provenance(cfg: dict, packages: dict, runtime: dict) -> dict:
    base = Path(cfg["base_model_path"])
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_files()}
    identity = {key: cfg[key] for key in (
        "dataset", "split", "prompt_template", "num_latent_steps", "max_new_tokens", "seed", "device", "atol", "rtol")}
    identity.update({"dataset_sha256": digest(Path(cfg["dataset_path"])),
                     "checkpoint_sha256": digest(Path(cfg["checkpoint_path"])),
                     "base_files": {path.name: digest(path) for path in sorted(base.iterdir()) if path.is_file()},
                     "source_hashes": source_hashes, "packages": packages, "runtime": runtime})
    return {"signature": stable_hash(identity), "identity": identity,
            "project_git": git_info(ROOT), "coconut_git": git_info(ROOT / "external/coconut")}


def load_baseline(path: str, signature: str, samples: list[dict]) -> tuple[dict, dict]:
    folder = Path(path)
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["stage"] != "baseline" or manifest["status"] != "completed" or manifest["signature"] != signature:
        raise ValueError("Baseline must be completed and match code/model/data/generation/environment signature")
    rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
    mapping = {row["sample_id"]: row for row in rows if row["status"] == "ok"}
    if len(mapping) != len(rows) or any(row["sample_id"] not in mapping for row in samples):
        raise ValueError("Baseline contains failed/duplicate records or does not cover the selected samples")
    return manifest, mapping


def load_gate(path: str, signature: str, baseline_id: str) -> dict:
    folder = Path(path)
    manifest = json.loads((folder / "manifest.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    if (manifest["stage"] != "resume" or manifest["status"] != "completed"
            or manifest["signature"] != signature or manifest["baseline_run_id"] != baseline_id
            or summary.get("checks_total", 0) == 0
            or summary.get("checks_passed") != summary["checks_total"]):
        raise ValueError("A completed, matching, fully passed resume check is required")
    return manifest


def tensor_comparison(torch, expected, actual, cfg) -> dict:
    if expected.shape != actual.shape or not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
        raise ValueError("Tensor shape mismatch or NaN/Inf")
    return {"close": bool(torch.allclose(expected, actual, atol=cfg["atol"], rtol=cfg["rtol"])),
            "max_abs_error": float((expected - actual).abs().max().item())}


def get_tokens(model, prompt: str, cfg: dict):
    untruncated = model.tokenizer(prompt, add_special_tokens=True)["input_ids"]
    total = len(untruncated) + cfg["num_latent_steps"] + 2 + cfg["max_new_tokens"]
    if total > model.base_model.config.n_positions:
        raise ValueError(f"Context budget exceeded: {total}; refusing silent truncation")
    tokens = model._prepare_inputs(prompt)
    expected = untruncated + [model.start_latent_id] + [model.latent_token_id] * cfg["num_latent_steps"] + [model.end_latent_id]
    if tokens["input_ids"][0].tolist() != expected:
        raise ValueError("Prepared prompt differs from checkpoint's question-newline-latent format")
    return tokens


def generated_record(output: dict, sample: dict, cfg: dict) -> dict:
    return {"full_text": output["text"][0], "continuation": output["continuation"][0],
            "generated_token_ids": output["generated_token_ids"][0],
            **score(output["continuation"][0], sample, cfg["dataset"])}


def reference_forward(model, tokens):
    return model.coconut_model(
        input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"],
        position_ids=tokens["position_ids"], labels=tokens["input_ids"].clone(),
    )


def random_edit(torch, h, strength: float, seed: int):
    # Use a private CPU generator. Noise direction is shared across strengths.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(h.shape, generator=generator, dtype=torch.float32).to(device=h.device, dtype=h.dtype)
    norm = noise.norm(dim=-1, keepdim=True)
    if (norm == 0).any() or (h.norm(dim=-1) == 0).any():
        raise ValueError("Cannot normalize zero-norm latent/noise")
    return h + strength * h.norm(dim=-1, keepdim=True) * noise / norm


def run(stage: str) -> None:
    parser = argparse.ArgumentParser(description=f"First-round {stage}")
    parser.add_argument("--config", required=True)
    parser.add_argument("--machine-config")
    parser.add_argument("--max-samples", type=int, help="0 = entire split; otherwise first N stable samples")
    parser.add_argument("--checkpoint")
    parser.add_argument("--run-id")
    parser.add_argument("--preflight", action="store_true", help="Validate without loading the model")
    parser.add_argument("--data-only", action="store_true", help="With --preflight: only validate data/config")
    parser.add_argument("--baseline-run", required=False)
    parser.add_argument("--resume-check", required=False)
    args = parser.parse_args()
    if args.data_only and not args.preflight:
        parser.error("--data-only requires --preflight")
    cfg = load_config(args)
    report, samples = preflight(cfg, args.data_only)
    if args.preflight:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["ok"]:
            raise SystemExit(1)
        return
    if not report["ok"]:
        raise SystemExit("Preflight failed:\n" + "\n".join(report["errors"]))
    if stage != "baseline" and not args.baseline_run:
        parser.error("--baseline-run is required")
    if stage == "interventions" and not args.resume_check:
        parser.error("--resume-check is required")

    import torch
    from common.models.coconut_model import CoconutWrapper

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if cfg["device"].startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    runtime = {"python": sys.version, "platform": platform.platform(), "torch_cuda": torch.version.cuda,
               "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
               "dtype": "float32", "batch_size": 1, "tf32": False,
               "torch_threads": torch.get_num_threads(),
               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True) if shutil.which("nvidia-smi") else None
    runtime["driver"] = driver.stdout.strip() if driver and driver.returncode == 0 else None
    prov = provenance(cfg, report["packages"], runtime)
    baseline_manifest, baselines, gate = None, {}, None
    if stage != "baseline":
        baseline_manifest, baselines = load_baseline(args.baseline_run, prov["signature"], samples)
    if stage == "interventions":
        gate = load_gate(args.resume_check, prov["signature"], baseline_manifest["run_id"])

    run_id = args.run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{cfg['dataset']}_{stage}_{uuid.uuid4().hex[:8]}")
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a directory name")
    folder = Path(cfg["output_root"]) / run_id
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "traces").mkdir()
    (folder / "logs").mkdir()
    (folder / "predictions.jsonl").touch()
    for path in source_files():
        target = folder / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write_json(folder / "config.resolved.json", cfg)
    write_json(folder / "samples.json", [{key: row[key] for key in ("sample_id", "index", "question", "gold_answer")} for row in samples])
    environment = {**runtime, "packages": report["packages"]}
    write_json(folder / "environment.json", environment)
    operations = [("identity", 0.0, 0), ("zero", 0.0, 0)] + [("random", value, seed) for value in cfg["random_strengths"] for seed in cfg["noise_seeds"]]
    expected = len(samples) * (1 if stage == "baseline" else cfg["num_latent_steps"] * (len(operations) if stage == "interventions" else 1))
    manifest = {"run_id": run_id, "stage": stage, "status": "running", "expected_records": expected,
                "baseline_run_id": baseline_manifest["run_id"] if baseline_manifest else None,
                "resume_run_id": gate["run_id"] if gate else None, "command": sys.argv,
                "started_utc": datetime.now(timezone.utc).isoformat(), **prov}
    write_json(folder / "manifest.json", manifest)
    print(f"RUN_DIR={folder}", flush=True)
    records = []
    status = "failed"
    log = (folder / "logs/run.log").open("w")
    try:
        model = CoconutWrapper()
        model.load_from_config({
            "base_model_name_or_path": cfg["base_model_path"], "tokenizer_name_or_path": cfg["base_model_path"],
            "checkpoint_path": cfg["checkpoint_path"], "strict_checkpoint": True, "device": cfg["device"],
            "num_latent_placeholders": cfg["num_latent_steps"], "use_coconut_question_only": False,
            "generation_kwargs": {"max_new_tokens": cfg["max_new_tokens"]},
        })
        manifest["checkpoint_load_report"] = model.checkpoint_load_report
        manifest["token_ids"] = {"start": model.start_latent_id, "end": model.end_latent_id, "latent": model.latent_token_id}
        write_json(folder / "manifest.json", manifest)

        def emit(row: dict):
            row.update({"run_id": run_id, "dataset": cfg["dataset"], "split": cfg["split"]})
            records.append(row)
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            message = f"{len(records)}/{expected} {row['sample_id']} {row.get('operation')} correct={row.get('correct')} passed={row.get('passed')}"
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()

        with (folder / "predictions.jsonl").open("w", encoding="utf-8") as handle, torch.no_grad():
            for sample_number, sample in enumerate(samples):
                prompt = cfg["prompt_template"].format(question=sample["question"])
                tokens = get_tokens(model, prompt, cfg)
                positions = (tokens["input_ids"][0] == model.latent_token_id).nonzero().flatten()
                common = {"sample_id": sample["sample_id"], "sample_index": sample["index"],
                          "gold_answer": sample["gold_answer"], "num_latent_steps": cfg["num_latent_steps"], "status": "ok"}
                save_trace = sample_number < cfg["save_trace_samples"]
                started = time.monotonic()
                if stage == "baseline":
                    output = model.run_baseline(prompt)
                    trace = output["inputs_embeds"][0, positions].detach().cpu()
                    row = {**common, "operation": "baseline", **generated_record(output, sample, cfg),
                           "latent_norms": trace.norm(dim=-1).tolist(), "elapsed_seconds": time.monotonic() - started}
                    if save_trace:
                        path = f"traces/{sample['index']:06d}_baseline.pt"
                        torch.save({"sample_id": sample["sample_id"], "latent_vectors": trace}, folder / path)
                        row["trace_path"] = path
                    emit(row)
                    del output
                    continue

                baseline = baselines[sample["sample_id"]]
                paired = {"baseline_correct": baseline["correct"], "baseline_answer": baseline["predicted_answer"],
                          "baseline_parse_status": baseline["parse_status"],
                          "baseline_continuation": baseline["continuation"]}
                reference = reference_forward(model, tokens)
                ref_trace = reference.inputs_embeds[:, positions, :].clone()
                ref_logits = reference.logits[:, -1:, :].clone()
                del reference
                for step in range(1, cfg["num_latent_steps"] + 1):
                    started = time.monotonic()
                    h, state = model.forward_until_step(prompt, step=step)
                    h_comparison = tensor_comparison(torch, ref_trace[:, step - 1, :], h, cfg)
                    snapshot_logits_len = len(state["logits"])
                    snapshot_embeds = state["inputs_embeds"].clone()
                    snapshot_cache = [(k.clone(), v.clone()) for k, v in model._kv_cache_to_legacy_pairs(state["past_key_values"])]
                    identity = model.rollout_from_step(h.clone(), state)
                    identity_trace = identity["inputs_embeds"][:, positions, :].clone()
                    trace_check = tensor_comparison(torch, ref_trace, identity_trace, cfg)
                    logits_check = tensor_comparison(torch, ref_logits, identity["logits"][:, -1:, :], cfg)
                    generation_matches = identity["generated_token_ids"][0] == baseline["generated_token_ids"]
                    # Deliberately reuse the prefix for a different branch, then identity again.
                    zero_probe = model.rollout_from_step(torch.zeros_like(h), state)
                    del zero_probe
                    replay = model.rollout_from_step(h.clone(), state)
                    state_unchanged = len(state["logits"]) == snapshot_logits_len and torch.equal(state["inputs_embeds"], snapshot_embeds)
                    cache_after = model._kv_cache_to_legacy_pairs(state["past_key_values"])
                    state_unchanged = state_unchanged and len(cache_after) == len(snapshot_cache) and all(torch.equal(k1, k2) and torch.equal(v1, v2) for (k1, v1), (k2, v2) in zip(snapshot_cache, cache_after))
                    replay_matches = replay["generated_token_ids"] == identity["generated_token_ids"] and tensor_comparison(torch, identity_trace, replay["inputs_embeds"][:, positions, :], cfg)["close"]
                    passed = h_comparison["close"] and trace_check["close"] and logits_check["close"] and generation_matches and state_unchanged and replay_matches
                    check = {**common, **paired, "operation": "identity", "step": step, "strength": 0.0, "noise_seed": 0,
                             **generated_record(identity, sample, cfg), "passed": passed, "h_check": h_comparison,
                             "trace_check": trace_check, "logits_check": logits_check, "generation_matches": generation_matches,
                             "state_unchanged": bool(state_unchanged), "replay_matches": replay_matches,
                             "elapsed_seconds": time.monotonic() - started}
                    if stage == "resume":
                        emit(check)
                    elif not passed:
                        raise RuntimeError(f"Identity check failed for {sample['sample_id']} step {step}; aborting interventions")
                    del replay, identity, snapshot_cache, snapshot_embeds
                    if stage == "resume":
                        continue
                    for operation, strength, noise_seed in operations:
                        started = time.monotonic()
                        modified = h.clone() if operation == "identity" else torch.zeros_like(h) if operation == "zero" else random_edit(torch, h, strength, noise_seed)
                        output = model.rollout_from_step(modified, state)
                        actual_trace = output["inputs_embeds"][:, positions, :]
                        difference = (actual_trace - ref_trace).norm(dim=-1)[0].tolist()
                        row = {**common, **paired, "operation": operation, "step": step, "strength": strength, "noise_seed": noise_seed,
                               **generated_record(output, sample, cfg), "latent_delta_norms": difference,
                               "edit_relative_norm": float((modified - h).norm().item() / h.norm().item()),
                               "elapsed_seconds": time.monotonic() - started}
                        if operation == "identity" and output["generated_token_ids"][0] != baseline["generated_token_ids"]:
                            raise RuntimeError("Identity drift during intervention sweep")
                        if save_trace:
                            path = f"traces/{sample['index']:06d}_s{step}_{operation}_{strength}_{noise_seed}.pt"
                            torch.save({"sample_id": sample["sample_id"], "baseline_latents": ref_trace.detach().cpu(),
                                        "modified_latents": actual_trace.detach().cpu()}, folder / path)
                            row["trace_path"] = path
                        emit(row)
                        del output
                    del state
        checks_failed = stage == "resume" and any(not row["passed"] for row in records)
        status = "completed" if len(records) == expected and not checks_failed else "failed"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception:
        error = traceback.format_exc()
        (folder / "logs/error.log").write_text(error)
        manifest["error"] = error
        raise
    finally:
        log.close()
        summary = summarize(records)
        summary.update({"expected_records": expected, "uncompleted_records": expected - len(records),
                        "run_error": manifest.get("error"), "status": status})
        write_json(folder / "summary.json", summary)
        manifest.update({"status": status, "record_count": len(records), "ended_utc": datetime.now(timezone.utc).isoformat()})
        write_json(folder / "manifest.json", manifest)
        write_json(folder / "checksums.json", {str(path.relative_to(folder)): digest(path) for path in sorted(folder.rglob("*")) if path.is_file() and path.name != "checksums.json"})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if status != "completed":
        raise SystemExit(1)

"""Oracle diagnostic: cross saved current-slot edits with saved future latents.

All latent input vectors are clamped during answer generation. Transformer
hidden states are recomputed; this does not freeze downstream computation or KV.
The transplant mode instead injects one saved donor vector into the original
prefix, then recomputes all later latent feedback naturally.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from experiments.first_round.data import score, summarize
from experiments.first_round.run import (
    ROOT, digest, get_tokens, load_baseline, load_config, load_gate,
    preflight, provenance, reference_forward, source_files, write_json,
)


def crossed_latents(baseline, edited, step):
    if baseline.shape != edited.shape or baseline.ndim != 3:
        raise ValueError("Expected matching [batch, steps, hidden] tensors")
    if not 1 <= step <= baseline.shape[1]:
        raise ValueError("Step out of range")
    result = {name: baseline.clone() for name in (
        "baseline_replay", "edited_replay", "edit_only", "suffix_only")}
    result["edited_replay"][:, step - 1:] = edited[:, step - 1:]
    result["edit_only"][:, step - 1] = edited[:, step - 1]
    result["suffix_only"][:, step:] = edited[:, step:]
    return result


def decode_fixed_latents(model, prompt, latents):
    """Decode the full prompt with supplied latent inputs and no feedback loop."""
    import torch
    tokens = model._prepare_inputs(prompt)
    positions = (tokens["input_ids"][0] == model.latent_token_id).nonzero().flatten()
    expected = (1, len(positions), model.base_model.config.n_embd)
    if tuple(latents.shape) != expected or not torch.isfinite(latents).all():
        raise ValueError(f"Expected finite latent matrix {expected}")
    embeds = model.coconut_model.embedding(tokens["input_ids"]).clone()
    embeds[:, positions] = latents.to(device=embeds.device, dtype=embeds.dtype)
    with torch.no_grad():
        output = model.base_model(inputs_embeds=embeds, attention_mask=tokens["attention_mask"],
                                  position_ids=tokens["position_ids"], use_cache=False)
        first_logits = output.logits[:, -1, :].detach().clone()
        token = first_logits[0].argmax().item()
        generated = [token]
        current = torch.cat((embeds, model.coconut_model.embedding(
            torch.tensor(token, device=model.device)).view(1, 1, -1)), dim=1)
        # Match the existing official-style batch-one greedy decoder.
        for _ in range(model.generation_kwargs.get("max_new_tokens", 64) - 1):
            token = model.base_model(inputs_embeds=current).logits[0, -1].argmax().item()
            if token == model.eos_token_id:
                break
            generated.append(token)
            current = torch.cat((current, model.coconut_model.embedding(
                torch.tensor(token, device=model.device)).view(1, 1, -1)), dim=1)
    return {"generated_token_ids": generated,
            "continuation": model.tokenizer.decode(generated, skip_special_tokens=True),
            "first_logits": first_logits, "inputs_embeds": embeds}


def transplant_latent(model, prompt, donor, target_step, baseline_ids):
    """Use the original prefix, change one feedback input, and continue naturally."""
    import torch
    with torch.no_grad():
        h, state = model.forward_until_step(prompt, target_step)
        identity = model.rollout_from_step(h.clone(), state)
        if identity["generated_token_ids"][0] != baseline_ids:
            raise ValueError("Transplant target identity does not reproduce baseline")
        if donor.shape != h.shape or not torch.isfinite(donor).all():
            raise ValueError("Invalid donor vector")
        output = model.rollout_from_step(donor.clone(), state)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--machine-config", required=True)
    parser.add_argument("--input-run", required=True)
    parser.add_argument("--sample-index", required=True, type=int)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--min-strength", type=float, default=0.5)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=["path_controls", "transplant"], default="path_controls")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    cfg = load_config(args)
    report, samples = preflight(cfg, False)
    if not report["ok"]:
        raise ValueError(report["errors"])
    inp = Path(args.input_run)
    input_manifest = json.loads((inp / "manifest.json").read_text())
    if input_manifest["status"] != "completed" or input_manifest["stage"] != "interventions":
        raise ValueError("A completed intervention run is required")
    for name, checksum in json.loads((inp / "checksums.json").read_text()).items():
        if digest(inp / name) != checksum:
            raise ValueError(f"Modified input result: {name}")
    selected = [r for r in (json.loads(line) for line in (inp / "predictions.jsonl").read_text().splitlines())
                if r["sample_index"] == args.sample_index and r["operation"] == "random"
                and r["step"] in args.steps and r["strength"] >= args.min_strength]
    if not selected or any("trace_path" not in r for r in selected):
        raise ValueError("Selected conditions require saved full trajectories")
    expected = (4 * len(selected) if args.mode == "path_controls" else
                sum(cfg["num_latent_steps"] - r["step"] for r in selected))
    if expected <= 0:
        raise ValueError("Selected sources have no later latent positions for transplantation")
    sample = next(s for s in samples if s["index"] == args.sample_index)
    import torch
    from common.models.coconut_model import CoconutWrapper
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = {"python": sys.version, "platform": platform.platform(), "torch_cuda": torch.version.cuda,
               "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
               "dtype": "float32", "batch_size": 1, "tf32": False,
               "torch_threads": torch.get_num_threads(),
               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True) if shutil.which("nvidia-smi") else None
    runtime["driver"] = driver.stdout.strip() if driver and driver.returncode == 0 else None
    prov = provenance(cfg, report["packages"], runtime)
    if prov["signature"] != input_manifest["signature"]:
        raise ValueError("Input run does not match current code/model/data/environment")
    baseline_manifest, baselines = load_baseline(
        str(inp.parent / input_manifest["baseline_run_id"]), prov["signature"], [sample])
    load_gate(str(inp.parent / input_manifest["resume_run_id"]), prov["signature"], baseline_manifest["run_id"])
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise ValueError("run-id must be a directory name")
    folder = Path(cfg["output_root"]) / args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "logs").mkdir()
    (folder / "traces").mkdir()
    for path in [*source_files(), Path(__file__), ROOT / "test_path_controls.py"]:
        target = folder / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write_json(folder / "config.resolved.json", cfg)
    write_json(folder / "samples.json", [{k: sample[k] for k in ("sample_id", "index", "question", "gold_answer")}])
    write_json(folder / "environment.json", {**runtime, "packages": report["packages"]})
    manifest = {**prov, "run_id": args.run_id, "stage": "path_controls" if args.mode == "path_controls" else "latent_transplants", "status": "running",
                "baseline_run_id": input_manifest["baseline_run_id"], "resume_run_id": input_manifest["resume_run_id"],
                "input_run_id": input_manifest["run_id"], "input_checksums_sha256": digest(inp / "checksums.json"),
                "path_control_source_hashes": {path.name: digest(path) for path in [Path(__file__), ROOT / "test_path_controls.py"]},
                "command": sys.argv, "expected_records": expected, "record_count": 0,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "interpretation": ("Oracle cross-trajectory input clamping; hidden states are recomputed. Not automatic semantic repair."
                                   if args.mode == "path_controls" else
                                   "Oracle same-question donor vector transplantation into the original prefix; later feedback is recomputed naturally. Not automatic semantic repair.")}
    write_json(folder / "manifest.json", manifest)
    records = []
    error = None
    print(f"RUN_DIR={folder}", flush=True)
    try:
        model = CoconutWrapper()
        model.load_from_config({"base_model_name_or_path": cfg["base_model_path"],
                               "tokenizer_name_or_path": cfg["base_model_path"],
                               "checkpoint_path": cfg["checkpoint_path"], "strict_checkpoint": True,
                               "device": cfg["device"], "num_latent_placeholders": cfg["num_latent_steps"],
                               "use_coconut_question_only": False,
                               "generation_kwargs": {"max_new_tokens": cfg["max_new_tokens"]}})
        manifest["checkpoint_load_report"] = model.checkpoint_load_report
        prompt = cfg["prompt_template"].format(question=sample["question"])
        tokens = get_tokens(model, prompt, cfg)
        with torch.no_grad():
            reference = reference_forward(model, tokens)
        positions = (tokens["input_ids"][0] == model.latent_token_id).nonzero().flatten()
        ref_latents = reference.inputs_embeds[:, positions].detach().clone()
        ref_logits = reference.logits[:, -1, :].detach().clone()
        del reference
        with (folder / "predictions.jsonl").open("w") as handle:
            for case_number, source in enumerate(selected):
                saved = torch.load(inp / source["trace_path"], map_location=model.device, weights_only=True)
                base, edited = saved["baseline_latents"], saved["modified_latents"]
                if saved["sample_id"] != sample["sample_id"] or not torch.allclose(base, ref_latents, atol=cfg["atol"], rtol=cfg["rtol"]):
                    raise ValueError("Saved baseline trajectory differs from independent reference")
                if not torch.equal(base[:, :source["step"] - 1], edited[:, :source["step"] - 1]):
                    raise ValueError("Saved intervention changed its preceding latent inputs")
                if args.mode == "path_controls":
                    branches = crossed_latents(base, edited, source["step"])
                    outputs = {name: decode_fixed_latents(model, prompt, vectors) for name, vectors in branches.items()}
                    if (outputs["baseline_replay"]["generated_token_ids"] != baselines[sample["sample_id"]]["generated_token_ids"]
                            or not torch.allclose(outputs["baseline_replay"]["first_logits"], ref_logits, atol=cfg["atol"], rtol=cfg["rtol"])
                            or outputs["edited_replay"]["generated_token_ids"] != source["generated_token_ids"]):
                        raise ValueError("Full-context baseline/edited trajectory replay failed; do not interpret hybrids")
                else:
                    with torch.no_grad():
                        _, source_state = model.forward_until_step(prompt, source["step"])
                        replay = model.rollout_from_step(edited[:, source["step"] - 1].clone(), source_state)
                    if (replay["generated_token_ids"][0] != source["generated_token_ids"] or
                            not torch.allclose(replay["inputs_embeds"][:, positions], edited, atol=cfg["atol"], rtol=cfg["rtol"])):
                        raise ValueError("Source natural trajectory replay failed")
                    outputs, branches = {}, {}
                    for target_step in range(source["step"] + 1, cfg["num_latent_steps"] + 1):
                        natural = transplant_latent(model, prompt, edited[:, target_step - 1], target_step,
                                                    baselines[sample["sample_id"]]["generated_token_ids"])
                        actual = natural["inputs_embeds"][:, positions].detach().clone()
                        if not torch.isfinite(actual).all():
                            raise ValueError("Non-finite transplanted trajectory")
                        if (not torch.allclose(actual[:, :target_step - 1], base[:, :target_step - 1], atol=cfg["atol"], rtol=cfg["rtol"]) or
                                not torch.equal(actual[:, target_step - 1], edited[:, target_step - 1])):
                            raise ValueError("Transplant changed the original prefix or failed to insert the donor")
                        name = f"latent_transplant_from_s{source['step']}_to_s{target_step}"
                        outputs[name] = {"generated_token_ids": natural["generated_token_ids"][0],
                                         "continuation": natural["continuation"][0], "target_step": target_step,
                                         "donor_relative_norm": float((edited[:, target_step - 1] - base[:, target_step - 1]).norm() / base[:, target_step - 1].norm()),
                                         "latent_delta_norms": (actual - base).norm(dim=-1)[0].tolist()}
                        branches[name] = actual
                for name, output in outputs.items():
                    trace_path = f"traces/{case_number:03d}_{name}.pt"
                    torch.save({"sample_id": sample["sample_id"], "latent_inputs": branches[name].detach().cpu()}, folder / trace_path)
                    row = {"run_id": args.run_id, "sample_id": sample["sample_id"], "sample_index": sample["index"],
                           "dataset": cfg["dataset"], "split": cfg["split"], "gold_answer": sample["gold_answer"],
                           "status": "ok", "operation": name, "step": output.get("target_step", source["step"]),
                           "source_step": source["step"], "strength": source["strength"],
                           "noise_seed": source["noise_seed"], "source_correct": source["correct"],
                           "source_answer": source["predicted_answer"], "source_trace_path": source["trace_path"],
                           "baseline_correct": source["baseline_correct"], "baseline_answer": source["baseline_answer"],
                           "baseline_parse_status": source["baseline_parse_status"], "baseline_continuation": source["baseline_continuation"],
                           "continuation": output["continuation"], "generated_token_ids": output["generated_token_ids"],
                           **score(output["continuation"], sample, cfg["dataset"]), "trace_path": trace_path}
                    if args.mode == "transplant":
                        row.update(donor_relative_norm=output["donor_relative_norm"], latent_delta_norms=output["latent_delta_norms"])
                    records.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                print(f"{case_number + 1}/{len(selected)} step={source['step']} strength={source['strength']} seed={source['noise_seed']} "
                      + " ".join(f"{name}={score(output['continuation'], sample, cfg['dataset'])['predicted_answer']}" for name, output in outputs.items()), flush=True)
        manifest["status"] = "completed"
    except BaseException:
        error = traceback.format_exc()
        (folder / "logs/error.log").write_text(error)
        manifest["status"] = "failed"
        raise
    finally:
        manifest.update(record_count=len(records), ended_utc=datetime.now(timezone.utc).isoformat())
        write_json(folder / "manifest.json", manifest)
        summary = {**summarize(records), "status": manifest["status"], "expected_records": manifest["expected_records"],
                   "uncompleted_records": manifest["expected_records"] - len(records), "run_error": error,
                   "source_conditions": len(selected), "by_branch": {name: summarize([r for r in records if r["operation"] == name])
                   for name in sorted({r["operation"] for r in records})}}
        write_json(folder / "summary.json", summary)
        write_json(folder / "checksums.json", {str(path.relative_to(folder)): digest(path)
                   for path in sorted(folder.rglob("*")) if path.is_file() and path.name != "checksums.json"})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

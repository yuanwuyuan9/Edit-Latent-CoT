"""Verify a returned result directory and rebuild statistics without model deps."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from experiments.first_round.data import prosqa_options, score, summarize
from experiments.first_round.run import digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", help="Derived reports only; defaults to analysis/<run_id>")
    args = parser.parse_args()
    folder = Path(args.run_dir)
    checksums = json.loads((folder / "checksums.json").read_text())
    for name, expected in checksums.items():
        path = folder / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Missing or modified result file: {name}")
    manifest = json.loads((folder / "manifest.json").read_text())
    cfg = json.loads((folder / "config.resolved.json").read_text())
    samples = {row["sample_id"]: row for row in json.loads((folder / "samples.json").read_text())}
    rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["sample_id"], row.get("step"), row["operation"], row.get("strength"), row.get("noise_seed"))
        if key in seen:
            raise ValueError(f"Duplicate experiment record: {key}")
        seen.add(key)
        if row["status"] == "ok":
            sample = dict(samples[row["sample_id"]])
            if cfg["dataset"] == "prosqa":
                sample["options"] = prosqa_options(sample["question"])
            rescored = score(row["continuation"], sample, cfg["dataset"])
            if any(row[field] != rescored[field] for field in rescored):
                raise ValueError(f"Stored score differs from local scoring: {key}")
        group = (row["operation"], row.get("step"), row.get("strength"), row.get("noise_seed"))
        groups[group].append(row)
    if manifest["record_count"] != len(rows):
        raise ValueError("Manifest record count does not match JSONL")
    overall = summarize(rows)
    overall.update({"run_id": manifest["run_id"], "status": manifest["status"],
                    "expected_records": manifest["expected_records"],
                    "note": "Intervention overall accuracy is per record; compare per-condition groups."})
    by_condition = [{"operation": key[0], "step": key[1], "strength": key[2], "noise_seed": key[3], **summarize(values)} for key, values in sorted(groups.items(), key=lambda item: str(item[0]))]
    target = Path(args.output_dir) if args.output_dir else Path("analysis") / manifest["run_id"]
    if target.resolve() == folder.resolve() or folder.resolve() in target.resolve().parents:
        raise ValueError("Analysis output must be outside the original run directory")
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "summary.recomputed.json", overall)
    write_json(target / "by_condition.json", by_condition)
    write_json(target / "analysis_manifest.json", {"input_run_id": manifest["run_id"], "input_checksums_sha256": digest(folder / "checksums.json"), "analysis_source_sha256": digest(Path(__file__)), "evaluator_source_sha256": digest(Path(__file__).parent / "experiments/first_round/data.py")})
    if manifest["stage"] == "baseline":
        write_json(target / "natural_errors.json", [samples[row["sample_id"]] | {"continuation": row["continuation"]} for row in rows if row["status"] == "ok" and row["parse_status"] == "ok" and not row["correct"]])
        write_json(target / "parse_failures.json", [samples[row["sample_id"]] | {"continuation": row["continuation"]} for row in rows if row["status"] == "ok" and row["parse_status"] == "failed"])
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"ANALYSIS_DIR={target}")


if __name__ == "__main__":
    main()

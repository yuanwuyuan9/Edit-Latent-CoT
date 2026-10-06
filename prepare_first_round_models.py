"""Download only the GPT-2 base and the requested trained Coconut checkpoint."""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine-config", required=True)
    parser.add_argument("--dataset", choices=["gsm8k", "prosqa"], required=True)
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    machine = json.loads(Path(args.machine_config).read_text())
    root = Path(machine["MODEL_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = {}
    base_repo = "openai-community/gpt2"
    base = root / "gpt2"
    if all((base / name).is_file() for name in ("model.safetensors", "config.json", "vocab.json", "merges.txt")):
        info["base"] = {"path": str(base), "reused_existing_files": True}
    else:
        revision = api.model_info(base_repo).sha
        snapshot_download(base_repo, revision=revision, local_dir=str(base), allow_patterns=[
            "model.safetensors", "config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.json", "merges.txt",
        ])
        info["base"] = {"repo_id": base_repo, "revision": revision, "path": str(base)}
    checkpoint_name = "checkpoint_33" if args.dataset == "gsm8k" else "checkpoint_40"
    repo = f"connordilgren/gpt2-{args.dataset}-coconut"
    target = root / f"{args.dataset}-coconut"
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / checkpoint_name
    if checkpoint.exists():
        info["checkpoint"] = {"path": str(checkpoint), "reused_existing_files": True}
    else:
        revision = api.model_info(repo).sha
        hf_hub_download(repo, checkpoint_name, revision=revision, local_dir=str(target))
        info["checkpoint"] = {"repo_id": repo, "revision": revision, "path": str(checkpoint)}
    info_path = root / f"download_{args.dataset}_{uuid.uuid4().hex[:8]}.json"
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

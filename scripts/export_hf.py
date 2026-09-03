from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HF_REPO = "MagistrTheOne/NULLXES-AMATERASU"
INIT_DIR = Path("/workspace/checkpoints/amaterasu_32b_v0.1_init")
EXPECTED_SHARDS = 51


def _require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return path


def write_hub_weight_index(init_dir: Path, dest: Path) -> Path:
    manifest = json.loads(_require(init_dir / "manifest.json").read_text(encoding="utf-8"))
    weight_map = manifest["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    if len(shard_names) != EXPECTED_SHARDS:
        raise SystemExit(f"manifest weight_map has {len(shard_names)} shards, expected {EXPECTED_SHARDS}")
    total_size = 0
    for name in shard_names:
        path = _require(init_dir / name)
        total_size += path.stat().st_size
    payload = {
        "metadata": {
            "format": "amaterasu-ckpt-v1",
            "model_id": manifest["model_id"],
            "frozen_total": manifest["frozen_total"],
            "total_parameters": manifest["frozen_total"],
            "total_size": total_size,
        },
        "weight_map": weight_map,
    }
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description="Upload AMATERASU-32B shards + Circuit-0 overlay to Hugging Face")
    p.add_argument("--init-dir", type=Path, default=INIT_DIR)
    p.add_argument("--circuit0-dir", type=Path, default=Path("/workspace/checkpoints/circuit0"))
    p.add_argument("--repo", default=HF_REPO)
    p.add_argument("--skip-init", action="store_true", help="card/metrics/index/overlay only, no 119G shards")
    p.add_argument("--skip-overlay", action="store_true", help="do not replace nces.safetensors")
    args = p.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=False)

    readme = _require(ROOT / "hub" / "README.md")
    cfg = _require(ROOT / "configs" / "model" / "amaterasu_32b_v0.1.json")
    api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md", repo_id=args.repo, commit_message="model card: AMATERASU-32B v0.1")
    api.upload_file(path_or_fileobj=str(cfg), path_in_repo="amaterasu_32b_v0.1.json", repo_id=args.repo, commit_message="freeze config json")

    metrics = args.circuit0_dir / "metrics.json"
    if metrics.exists():
        api.upload_file(
            path_or_fileobj=str(metrics),
            path_in_repo="metrics.json",
            repo_id=args.repo,
            commit_message="circuit-0 metrics",
        )

    if not args.skip_init:
        init_dir = _require(args.init_dir)
        shards = sorted(init_dir.glob("*.safetensors"))
        if len(shards) != EXPECTED_SHARDS:
            raise SystemExit(f"expected {EXPECTED_SHARDS} init shards in {init_dir}, got {len(shards)}")
        manifest = _require(init_dir / "manifest.json")
        print(f"uploading {len(shards)} init shards + {manifest.name} from {init_dir}", flush=True)
        api.upload_folder(
            folder_path=str(init_dir),
            repo_id=args.repo,
            allow_patterns=["*.safetensors", "manifest.json"],
            commit_message="AMATERASU-32B v0.1 freeze init shards (amaterasu-ckpt-v1)",
        )

    if args.init_dir.exists():
        index_path = write_hub_weight_index(args.init_dir, args.init_dir / "model.safetensors.index.json")
        api.upload_file(
            path_or_fileobj=str(index_path),
            path_in_repo="model.safetensors.index.json",
            repo_id=args.repo,
            commit_message="Hub safetensors index from amaterasu-ckpt-v1 weight_map",
        )

    nces = args.circuit0_dir / "nces-circuit0.safetensors"
    legacy = Path("/workspace/checkpoints/nces-circuit0.safetensors")
    if not nces.exists() and legacy.exists():
        args.circuit0_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, nces)
    if nces.exists() and not args.skip_overlay:
        api.upload_file(
            path_or_fileobj=str(nces),
            path_in_repo="nces.safetensors",
            repo_id=args.repo,
            commit_message="circuit-0 NCES overlay on freeze nces shard",
        )
        try:
            api.delete_file(
                path_in_repo="nces-circuit0.safetensors",
                repo_id=args.repo,
                commit_message="drop duplicate NCES file so Hub param count stays frozen_total",
            )
        except Exception as exc:
            print(f"delete nces-circuit0.safetensors skipped: {exc}", flush=True)

    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()

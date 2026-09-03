from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HF_REPO = "MagistrTheOne/NULLXES-AMATERASU"


def main() -> None:
    p = argparse.ArgumentParser(description="Upload Circuit-0 artifacts to Hugging Face")
    p.add_argument("--circuit0-dir", type=Path, default=Path("/workspace/checkpoints/circuit0"))
    p.add_argument("--repo", default=HF_REPO)
    args = p.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=False)

    readme = ROOT / "hub" / "README.md"
    cfg = ROOT / "configs" / "model" / "amaterasu_32b_v0.1.json"
    api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md", repo_id=args.repo, commit_message="model card: AMATERASU-32B v0.1")
    api.upload_file(path_or_fileobj=str(cfg), path_in_repo="amaterasu_32b_v0.1.json", repo_id=args.repo, commit_message="freeze config json")

    nces = args.circuit0_dir / "nces-circuit0.safetensors"
    legacy = Path("/workspace/checkpoints/nces-circuit0.safetensors")
    if not nces.exists() and legacy.exists():
        args.circuit0_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, nces)
    if nces.exists():
        api.upload_file(
            path_or_fileobj=str(nces),
            path_in_repo="nces-circuit0.safetensors",
            repo_id=args.repo,
            commit_message="circuit-0 NCES encoder safetensors",
        )
    metrics = args.circuit0_dir / "metrics.json"
    if metrics.exists():
        api.upload_file(
            path_or_fileobj=str(metrics),
            path_in_repo="metrics.json",
            repo_id=args.repo,
            commit_message="circuit-0 metrics",
        )
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()

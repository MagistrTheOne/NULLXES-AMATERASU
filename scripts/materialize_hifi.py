from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amaterasu.data.hifi_schema import HIFI_REPO


VIDEO_MARKERS = (".mp4", "/videos/", "observation.images")


def _is_parquet(name: str) -> bool:
    n = name.replace("\\", "/").lower()
    if not n.endswith(".parquet"):
        return False
    return not any(m in n for m in VIDEO_MARKERS)


def main() -> None:
    p = argparse.ArgumentParser(description="Bounded HiFi-UMI-2K parquet materializer. Never downloads video.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-episodes", type=int, default=64)
    p.add_argument("--max-bytes", type=int, default=512 * 1024 * 1024)
    p.add_argument("--max-files", type=int, default=2)
    args = p.parse_args()
    if args.max_bytes <= 0 or args.max_episodes <= 0:
        raise SystemExit("max-bytes and max-episodes must be > 0")

    from huggingface_hub import HfApi, hf_hub_download

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    files = [f for f in api.list_repo_files(HIFI_REPO, repo_type="dataset") if _is_parquet(f)]
    files.sort()
    if not files:
        raise SystemExit("no parquet files listed on Hub")

    used = 0
    saved = 0
    meta = []
    for name in files:
        if saved >= args.max_files:
            break
        try:
            info = api.get_paths_info(HIFI_REPO, [name], repo_type="dataset")
            size = int(getattr(info[0], "size", 0) or 0) if info else 0
        except Exception:
            size = 0
        if size and used + size > args.max_bytes:
            print(f"stop before {name}: {used}+{size} > max-bytes {args.max_bytes}", flush=True)
            break
        path = hf_hub_download(
            repo_id=HIFI_REPO,
            filename=name,
            repo_type="dataset",
            local_dir=str(out),
        )
        actual = Path(path).stat().st_size
        if used + actual > args.max_bytes:
            Path(path).unlink(missing_ok=True)
            print(f"deleted {name}: would exceed max-bytes", flush=True)
            break
        used += actual
        saved += 1
        meta.append(f"{name}\t{actual}")
        print(f"saved {name} ({actual:,} bytes) total={used:,}", flush=True)

    (out / "MATERIALIZE.txt").write_text(
        f"repo={HIFI_REPO}\nmax_episodes={args.max_episodes}\nmax_bytes={args.max_bytes}\n"
        f"files={saved}\nbytes={used}\n" + "\n".join(meta) + "\n",
        encoding="utf-8",
    )
    if saved == 0:
        raise SystemExit("AMATERASU DATA GATE FAILED: zero parquet files materialized")
    print(f"HIFI MATERIALIZE PASSED files={saved} bytes={used}")


if __name__ == "__main__":
    main()

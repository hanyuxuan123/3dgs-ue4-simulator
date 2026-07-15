#!/usr/bin/env python3
"""Copy the ClassLab 3DGS-CARLA bridge scripts listed in bridge_bundle_manifest.json."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("bridge_bundle_manifest.json"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--include-data-examples", action="store_true")
    return parser.parse_args()


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        print(f"[bundle] missing: {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[bundle] copied: {src} -> {dst}")
    return True


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = list(manifest.get("files", []))
    if args.include_data_examples:
        files.extend(manifest.get("optional_data_examples", []))

    copied = 0
    missing = 0
    for rel in files:
        src = repo_root / rel
        dst = args.output_dir / rel
        if copy_file(src, dst):
            copied += 1
        else:
            missing += 1

    print(f"[bundle] done copied={copied} missing={missing} output={args.output_dir}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

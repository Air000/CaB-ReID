#!/usr/bin/env python3
"""Verify evaluation checkpoints, the shared encoder, and the region adapter."""

import argparse
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mvti", "trc31k", "all"), default="all")
    parser.add_argument("--backbone", choices=("clipreid", "transreid", "all"), default="all")
    parser.add_argument("--weights-dir", type=Path, default=Path(__file__).resolve().parents[1] / "weights")
    args = parser.parse_args()
    root = args.weights_dir
    manifest = dict(row.split("  ", 1)[::-1] for row in (root / "CHECKSUMS.sha256").read_text().splitlines())
    datasets = ("mvti", "trc31k") if args.dataset == "all" else (args.dataset,)
    backbones = ("clipreid", "transreid") if args.backbone == "all" else (args.backbone,)
    names = ["clip_region_adapter.pt", "cabreid_region_encoder.pth"]
    names += [f"cabreid_{backbone}_{dataset}.pth"
              for dataset in datasets for backbone in backbones]
    for name in names:
        if name not in manifest:
            raise RuntimeError(f"Checksum manifest is missing: {name}")
        expected = manifest[name]
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Place the checkpoint at {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise RuntimeError(f"Checksum mismatch: {name}")
        print(f"OK: {name}")


if __name__ == "__main__":
    main()

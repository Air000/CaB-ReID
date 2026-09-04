#!/usr/bin/env python3
"""Unified CaB-ReID training and evaluation entry point."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
BACKBONES = {
    "clipreid": {
        "directory": "clipreid",
        "train": "train_clipreid.py",
        "evaluate": "test_clipreid.py",
        "config": "configs/trc31k/cabreid_clipreid.yml",
    },
    "transreid": {
        "directory": "transreid",
        "train": "train.py",
        "evaluate": "test.py",
        "config": "configs/trc31k/cabreid_transreid.yml",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "evaluate"))
    parser.add_argument("--backbone", choices=tuple(BACKBONES), required=True)
    parser.add_argument("--dataset", choices=("trc31k", "mvti"), default="trc31k")
    parser.add_argument("--data-root", type=Path, help="Extracted dataset directory or its parent")
    parser.add_argument("--config", help="Override the backbone's default configuration")
    parser.add_argument("--weight", help="Checkpoint used for evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Print the delegated command")
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    if args.action == "evaluate" and not args.weight:
        parser.error("--weight is required for evaluation")
    return args


def main():
    args = parse_args()
    choice = BACKBONES[args.backbone]
    directory = ROOT / choice["directory"]
    config = args.config or choice["config"].replace("/trc31k/", f"/{args.dataset}/")
    overrides = list(args.overrides)
    if overrides[:1] == ["--"]:
        overrides = overrides[1:]
    command = [sys.executable, choice[args.action], "--config_file", config]
    if args.data_root:
        command.extend(("DATASETS.ROOT_DIR", str(args.data_root.expanduser().resolve())))
    if args.weight:
        weight = Path(args.weight)
        if not weight.is_absolute():
            weight = ROOT / weight
        command.extend(("TEST.WEIGHT", str(weight)))
    if args.action == "evaluate":
        command.extend(("MODEL.PRETRAIN_CHOICE", "self"))
    command.extend(overrides)
    if args.dry_run:
        print("Working directory:", directory)
        print("Command:", " ".join(command))
        return
    environment = os.environ.copy()
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(ROOT) if not python_path else str(ROOT) + os.pathsep + python_path
    subprocess.run(command, cwd=directory, env=environment, check=True)


if __name__ == "__main__":
    main()

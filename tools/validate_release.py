#!/usr/bin/env python3
"""Run the validator shipped with an extracted TRC-31K dataset."""

import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset_path import resolve_dataset_path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("dataset", type=Path, nargs="?", default=ROOT.parent / "dataset")
DATASET = resolve_dataset_path(parser.parse_args().dataset)
VALIDATOR = DATASET / "validate_release.py"

if not VALIDATOR.is_file():
    raise RuntimeError(f"Dataset validator is missing: {VALIDATOR}")

sys.argv = [str(VALIDATOR), str(DATASET)]
runpy.run_path(str(VALIDATOR), run_name="__main__")

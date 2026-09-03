#!/usr/bin/env python3
"""Run the validator shipped with the sibling TRC-31K dataset."""

import runpy
import sys
from pathlib import Path


DATASET = Path(__file__).resolve().parents[2] / "dataset" / "TRC_31K_v1.0"
VALIDATOR = DATASET / "validate_release.py"

if not VALIDATOR.is_file():
    raise RuntimeError(f"Dataset validator is missing: {VALIDATOR}")

sys.argv = [str(VALIDATOR), str(DATASET)]
runpy.run_path(str(VALIDATOR), run_name="__main__")

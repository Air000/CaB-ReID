#!/usr/bin/env python3
"""Check v4.1 source layout, optionally comparing a local v4.0 reference."""

import argparse
import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def python_sources():
    return sorted(path for path in ROOT.rglob("*.py") if "__pycache__" not in path.parts)


def normalized_config(path):
    text = path.read_text(encoding="utf-8")
    return re.sub(r"^OUTPUT_DIR:.*$", "OUTPUT_DIR: <release>", text, flags=re.MULTILINE)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-code", type=Path, help="Optional v4.0 code directory")
    args = parser.parse_args()
    sources = python_sources()
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    clip_model = (ROOT / "clipreid/model/make_model_clipreid.py").read_text(encoding="utf-8")
    trans_model = (ROOT / "transreid/model/make_model.py").read_text(encoding="utf-8")
    for name, source in (("CLIP-ReID", clip_model), ("TransReID", trans_model)):
        require("CaBReIDModule" in source, f"{name} does not use CaBReIDModule")
        require("PromptRegionMasker" in source, f"{name} does not use PromptRegionMasker")
        require("class RegionAdapter" not in source, f"{name} still duplicates RegionAdapter")

    duplicate_plugins = list(ROOT.rglob("prompt_part_plugin.py"))
    require(not duplicate_plugins, "Backbone-local prompt/part plugins remain")

    for backbone in ("clipreid", "transreid"):
        new = ROOT / backbone / "configs/trc31k" / f"cabreid_v4_1_{backbone}.yml"
        require(new.is_file(), f"Missing configuration: {new}")
        if args.reference_code is not None:
            old = args.reference_code / backbone / "configs/trc31k" / f"cabreid_v4_{backbone}.yml"
            require(old.is_file(), f"Missing reference configuration: {old}")
            require(normalized_config(old) == normalized_config(new), f"Configuration drift: {new}")

    print(f"CaB-ReID v4.1 source validation passed ({len(sources)} Python files).")
    print("Both backbones use the shared modules.")
    if args.reference_code is not None:
        print("Configuration parity with the v4.0 reference passed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check the portable CaB-ReID source layout."""

import argparse
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def python_sources():
    return sorted(path for path in ROOT.rglob("*.py") if "__pycache__" not in path.parts)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
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
        for dataset in ("trc31k", "mvti"):
            config = ROOT / backbone / "configs" / dataset / f"cabreid_{backbone}.yml"
            require(config.is_file(), f"Missing configuration: {config}")

    print(f"CaB-ReID source validation passed ({len(sources)} Python files).")
    print("Both backbones use the shared modules.")


if __name__ == "__main__":
    main()

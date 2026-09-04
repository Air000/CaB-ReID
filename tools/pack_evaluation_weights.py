#!/usr/bin/env python3
"""Export lossless evaluation weights from the four full paper checkpoints."""

import argparse
from collections import OrderedDict
from pathlib import Path
import shutil
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cabreid.checkpoint import (
    EVALUATION_FORMAT, SHARED_ENCODER, is_training_parameter, normalise_state, sha256,
)
from cabreid.masking import PromptRegionMasker


def shared_parameter(name):
    return (name.startswith("cabreid.region_masker.encoder.")
            and name != "cabreid.region_masker.encoder.positional_embedding")


def pack(source_dir, output_dir):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source_dir == output_dir:
        raise ValueError("Export to a separate directory; preserve the full checkpoints.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    expected = {}
    for row in (source_dir / "CHECKSUMS.sha256").read_text().splitlines():
        digest, name = row.split("  ", 1)
        expected[name] = digest
    names = [f"cabreid_{backbone}_{dataset}.pth"
             for backbone in ("clipreid", "transreid") for dataset in ("mvti", "trc31k")]
    for name in [*names, "clip_region_adapter.pt"]:
        if sha256(source_dir / name) != expected[name]:
            raise RuntimeError(f"Source checkpoint checksum mismatch: {name}")

    shared, shared_hash = None, None
    for name in names:
        _, backbone, dataset = Path(name).stem.split("_")
        source = torch.load(source_dir / name, map_location="cpu", weights_only=True, mmap=True)
        state, buffers = normalise_state(source)
        common = OrderedDict((key, value) for key, value in state.items() if shared_parameter(key))
        if shared is None:
            shared = OrderedDict((key, value.clone()) for key, value in common.items())
            torch.save({"format": "cabreid-shared-encoder", "state_dict": shared}, output_dir / SHARED_ENCODER)
            shared_hash = sha256(output_dir / SHARED_ENCODER)
        elif common.keys() != shared.keys() or any(
            value.dtype != shared[key].dtype or not torch.equal(value, shared[key])
            for key, value in common.items()
        ):
            raise RuntimeError(f"Frozen encoder is not shared: {name}")

        config_path = ROOT / backbone / "configs" / dataset / f"cabreid_{backbone}.yml"
        config = yaml.safe_load(config_path.read_text())
        masker = PromptRegionMasker(
            torch.nn.Identity(), source_dir / "clip_region_adapter.pt", (1, 1),
            config["INPUT"]["PIXEL_MEAN"], config["INPUT"]["PIXEL_STD"],
        )
        for key in ("text_features", "reid_mean", "reid_std", "clip_mean", "clip_std"):
            buffers.setdefault(f"cabreid.region_masker.{key}", getattr(masker, key))
        retained = OrderedDict(
            (key, value.clone()) for key, value in state.items()
            if not shared_parameter(key) and not is_training_parameter(key, backbone)
        )
        bundle = {
            "format": EVALUATION_FORMAT,
            "model": {"backbone": backbone, "dataset": dataset,
                      "image_size": config["INPUT"]["SIZE_TRAIN"],
                      "stride": config["MODEL"]["STRIDE_SIZE"]},
            "source_sha256": expected[name],
            "shared_encoder": SHARED_ENCODER,
            "shared_sha256": shared_hash,
            "state_dict": retained,
            "buffers": {key: value.clone() for key, value in buffers.items()},
        }
        torch.save(bundle, output_dir / name)
        print(f"{name}: {(output_dir / name).stat().st_size / 1e6:.2f} MB", flush=True)
        del source, state, common, retained, bundle

    shutil.copy2(source_dir / "clip_region_adapter.pt", output_dir / "clip_region_adapter.pt")
    files = sorted([*names, SHARED_ENCODER, "clip_region_adapter.pt"])
    (output_dir / "CHECKSUMS.sha256").write_text("".join(
        f"{sha256(output_dir / name)}  {name}\n" for name in files
    ))
    print(f"Total: {sum((output_dir / name).stat().st_size for name in files) / 1e9:.3f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pack(args.source_dir, args.output_dir)

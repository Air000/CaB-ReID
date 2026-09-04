"""Load native-backbone and portable CaB-ReID checkpoints."""

from collections import OrderedDict
import hashlib
from pathlib import Path

import torch

from .evaluation import TRAINING_MODULES


EVALUATION_FORMAT = "cabreid-evaluation"
SHARED_ENCODER = "cabreid_region_encoder.pth"

PREFIXES = {
    "online_part_encoder.": "cabreid.region_masker.encoder.",
    "online_region_adapter.": "cabreid.region_masker.adapter.",
}
BUFFERS = {
    "online_text_features": "cabreid.region_masker.text_features",
    "online_input_mean": "cabreid.region_masker.reid_mean",
    "online_input_std": "cabreid.region_masker.reid_std",
    "online_clip_mean": "cabreid.region_masker.clip_mean",
    "online_clip_std": "cabreid.region_masker.clip_std",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_state(source):
    if "state_dict" in source:
        source = source["state_dict"]
    state, buffers = OrderedDict(), {}
    for name, value in source.items():
        name = name.removeprefix("module.")
        if name in BUFFERS:
            buffers[BUFFERS[name]] = value
            continue
        for old, new in PREFIXES.items():
            if name.startswith(old):
                name = new + name[len(old):]
                break
        if name in state:
            raise ValueError(f"Duplicate checkpoint parameter: {name}")
        state[name] = value
    return state, buffers


def is_training_parameter(name, backbone):
    return name.split(".")[0] in TRAINING_MODULES[backbone]


def load_checkpoint(model, path):
    path = Path(path)
    source = torch.load(path, map_location="cpu", weights_only=True)
    spec = getattr(model, "evaluation_spec", None)
    compact = source.get("format") == EVALUATION_FORMAT
    if compact:
        if spec is None:
            raise RuntimeError("Use make_model(..., evaluation_only=True) for evaluation checkpoints.")
        if source.get("model") != spec:
            raise RuntimeError(f"Checkpoint configuration mismatch: expected {spec}, got {source.get('model')}")
        if source.get("shared_encoder") != SHARED_ENCODER:
            raise ValueError("Unrecognised shared encoder filename")
        shared_path = path.parent / SHARED_ENCODER
        if not shared_path.is_file():
            raise FileNotFoundError(f"Place the shared encoder beside the checkpoint: {shared_path}")
        if sha256(shared_path) != source.get("shared_sha256"):
            raise RuntimeError("Shared encoder checksum mismatch")
        shared = torch.load(shared_path, map_location="cpu", weights_only=True)
        if shared.get("format") != "cabreid-shared-encoder":
            raise ValueError("Unrecognised shared encoder format")
        state = OrderedDict(source["state_dict"])
        if state.keys() & shared["state_dict"].keys():
            raise ValueError("Overlapping model and shared encoder parameters")
        state.update(shared["state_dict"])
        buffers = source["buffers"]
    else:
        state, buffers = normalise_state(source)
        if spec is not None:
            state = OrderedDict((name, value) for name, value in state.items()
                                if not is_training_parameter(name, spec["backbone"]))

    target_buffers = dict(model.named_buffers())
    if compact:
        required_buffers = {name for name in target_buffers if name in BUFFERS.values()}
        if buffers.keys() != required_buffers:
            raise RuntimeError("Evaluation checkpoint mask buffers are incomplete or unexpected")
        target = model.state_dict()
        for name, value in state.items():
            if name in target and value.dtype != target[name].dtype:
                raise RuntimeError(f"Checkpoint dtype mismatch: {name}")
    for name, value in buffers.items():
        if name not in target_buffers or target_buffers[name].shape != value.shape:
            raise RuntimeError(f"Checkpoint buffer does not match the model: {name}")
        if compact and target_buffers[name].dtype != value.dtype:
            raise RuntimeError(f"Checkpoint buffer dtype mismatch: {name}")
    model.load_state_dict(state, strict=True)
    with torch.no_grad():
        for name, value in buffers.items():
            target_buffers[name].copy_(value)
    print(f"Loaded {len(state)} checkpoint tensors from {path}")

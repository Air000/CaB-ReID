"""MV-TI single-view, combined-gallery protocol used in the paper."""

from pathlib import Path
import re


class MVTISingleView:
    views = (("front", "F", 0), ("side", "S", 1), ("back", "B", 2))

    def __init__(self, root="", verbose=True, **kwargs):
        root = Path(root).expanduser().resolve()
        candidates = (root, root / "MV-TI", root / "MV-TI" / "truck", root / "truck")
        dataset = next((p for p in candidates if (p / "train_names.txt").is_file()), None)
        if dataset is None:
            raise FileNotFoundError(f"MV-TI split lists not found under {root}")
        self.dataset_dir = str(dataset)
        for split in ("train", "query", "gallery"):
            samples = self._read(dataset, split)
            setattr(self, split, samples)
            counts = (len({r[1] for r in samples}), len(samples),
                      len({r[2] for r in samples}), len({r[3] for r in samples}))
            for field, count in zip(("pids", "imgs", "cams", "vids"), counts):
                setattr(self, f"num_{split}_{field}", count)
            if verbose:
                print(f"MV-TI {split}: {counts[1]} images, {counts[0]} identities")

    def _read(self, root, split):
        names = (root / f"{split}_names.txt").read_text(encoding="utf-8").splitlines()
        names = [name.strip() for name in names if name.strip()]
        parsed = []
        for name in names:
            match = re.match(r"^(-?\d+)_c(\d+)_", name)
            if match is None or Path(name).name != name or "\\" in name:
                raise ValueError(f"Invalid MV-TI sample name: {name}")
            parsed.append((name, int(match[1]), int(match[2]) - 1))
        labels = {pid: i for i, pid in enumerate(sorted({r[1] for r in parsed}))}
        samples = []
        for name, pid, camera in parsed:
            for directory, suffix, view in self.views:
                path = root / directory / split / f"{name}_{suffix}.jpg"
                if not path.is_file():
                    raise FileNotFoundError(f"MV-TI image is missing: {path}")
                samples.append((str(path), labels[pid] if split == "train" else pid, camera, view))
        return samples

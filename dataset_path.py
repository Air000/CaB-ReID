"""Locate an extracted TRC-31K dataset by its split manifests."""

from pathlib import Path


def resolve_dataset_path(root):
    root = Path(root).expanduser().resolve()

    def has_splits(path):
        return all(
            (path / split).is_dir() and (path / "metadata" / (split + ".csv")).is_file()
            for split in ("train", "query", "gallery")
        )

    if has_splits(root):
        return root
    candidates = (
        sorted(path for path in root.iterdir() if path.is_dir() and has_splits(path))
        if root.is_dir()
        else []
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No dataset with train/query/gallery manifests found in {root}")
    raise ValueError(f"Multiple datasets found in {root}; specify the dataset directory directly")

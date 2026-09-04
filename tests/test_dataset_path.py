from pathlib import Path
import tempfile
import unittest

from dataset_path import resolve_dataset_path


class DatasetPathTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def make_dataset(self, name):
        dataset = self.root / name
        (dataset / "metadata").mkdir(parents=True)
        for split in ("train", "query", "gallery"):
            (dataset / split).mkdir()
            (dataset / "metadata" / (split + ".csv")).touch()
        return dataset

    def test_direct_dataset(self):
        dataset = self.make_dataset("TRC-31K")
        self.assertEqual(resolve_dataset_path(dataset), dataset.resolve())

    def test_folder_name_is_not_fixed(self):
        dataset = self.make_dataset("extracted_dataset")
        self.assertEqual(resolve_dataset_path(self.root), dataset.resolve())

    def test_ambiguous_parent_requires_explicit_path(self):
        dataset = self.make_dataset("first")
        self.make_dataset("second")
        with self.assertRaises(ValueError):
            resolve_dataset_path(self.root)
        self.assertEqual(resolve_dataset_path(dataset), dataset.resolve())

    def test_missing_dataset(self):
        with self.assertRaises(FileNotFoundError):
            resolve_dataset_path(self.root)

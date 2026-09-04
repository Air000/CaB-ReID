from pathlib import Path
import tempfile
import unittest

from mvti_dataset import MVTISingleView


class MVTIDatasetTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.parent = Path(self.directory.name)
        self.root = self.parent / "truck"
        self.root.mkdir()
        for split in ("train", "query", "gallery"):
            names = ["0009_c2_a", "0003_c1_b"]
            (self.root / f"{split}_names.txt").write_text("\n".join(names))
            for view, suffix, _ in MVTISingleView.views:
                directory = self.root / view / split
                directory.mkdir(parents=True)
                for name in names:
                    (directory / f"{name}_{suffix}.jpg").touch()

    def test_order_labels_cameras_and_views(self):
        dataset = MVTISingleView(self.parent, verbose=False)
        self.assertEqual([(r[1], r[2], r[3]) for r in dataset.train],
                         [(1, 1, 0), (1, 1, 1), (1, 1, 2), (0, 0, 0), (0, 0, 1), (0, 0, 2)])
        self.assertEqual([r[1] for r in dataset.query], [9, 9, 9, 3, 3, 3])
        self.assertEqual(dataset.num_train_pids, 2)
        self.assertEqual(dataset.num_train_imgs, 6)

    def test_direct_root(self):
        self.assertEqual(MVTISingleView(self.root, verbose=False).dataset_dir, str(self.root.resolve()))

    def test_missing_view_is_not_silently_skipped(self):
        (self.root / "side" / "query" / "0009_c2_a_S.jpg").unlink()
        with self.assertRaises(FileNotFoundError):
            MVTISingleView(self.root, verbose=False)

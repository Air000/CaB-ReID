import csv
import os.path as osp
from dataset_path import resolve_dataset_path

from .bases import BaseImageDataset


class TRC31K(BaseImageDataset):
    """TRC-31K image-level truck Re-ID dataset."""

    def __init__(self, root="", verbose=True, **kwargs):
        super().__init__()
        self.dataset_dir = str(resolve_dataset_path(root))
        metadata = osp.join(self.dataset_dir, "metadata")
        train = self._read(osp.join(metadata, "train.csv"), relabel=True)
        query = self._read(osp.join(metadata, "query.csv"), relabel=False)
        gallery = self._read(osp.join(metadata, "gallery.csv"), relabel=False)

        if verbose:
            print("=> TRC-31K loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train, self.query, self.gallery = train, query, gallery
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(gallery)

    def _read(self, manifest, relabel):
        if not osp.isfile(manifest):
            raise RuntimeError("Dataset manifest is missing: {}".format(manifest))
        with open(manifest, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        pid2label = {pid: label for label, pid in enumerate(sorted({int(row["pid"]) for row in rows}))}
        samples = []
        for row in rows:
            path = osp.join(self.dataset_dir, row["relative_path"])
            if not osp.isfile(path):
                raise RuntimeError("Dataset image is missing: {}".format(path))
            pid = int(row["pid"])
            samples.append((path, pid2label[pid] if relabel else pid, int(row["cam_index"]), 0))
        return samples

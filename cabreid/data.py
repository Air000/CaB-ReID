from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
import torch
from timm.data.random_erasing import RandomErasing
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(path):
    while True:
        try:
            return Image.open(path).convert("RGB")
        except IOError:
            print("Unable to read '{}'; retrying".format(path))


def read_mask(path):
    path = Path(path)
    return Image.open(path).convert("L") if path.exists() else Image.new("L", (1, 1), 0)


class PartRegionImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None, dataset_root="", cab_mask_root="", body_mask_root="", body_fallback_mask_root=""):
        self.dataset = dataset
        self.transform = transform
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.cab_mask_root = Path(cab_mask_root) if cab_mask_root else None
        self.body_mask_root = Path(body_mask_root) if body_mask_root else None
        self.body_fallback_mask_root = Path(body_fallback_mask_root) if body_fallback_mask_root else None

    def __len__(self):
        return len(self.dataset)

    def _relative_path(self, image_path):
        path = Path(image_path)
        if self.dataset_root is not None:
            try:
                return path.relative_to(self.dataset_root)
            except ValueError:
                pass
        return Path(path.parent.name) / path.name

    def _candidate_paths(self, root, image_path):
        if root is None:
            return []
        relative = self._relative_path(image_path)
        paths = [root / relative]
        if relative.suffix.lower() != ".png":
            paths.append(root / relative.with_suffix(".png"))
        return paths

    def _part_mask(self, root, image_path, image_size, fallback=None):
        for candidate_root in (root, fallback):
            for path in self._candidate_paths(candidate_root, image_path):
                if path.exists():
                    return read_mask(path)
        return Image.new("L", image_size, 0)

    def __getitem__(self, index):
        image_path, pid, camera, view = self.dataset[index]
        image = read_image(image_path)
        cab_mask = self._part_mask(self.cab_mask_root, image_path, image.size)
        body_mask = self._part_mask(self.body_mask_root, image_path, image.size, self.body_fallback_mask_root)
        if self.transform is not None:
            image, part_mask = self.transform(image, cab_mask, body_mask)
        else:
            part_mask = torch.stack(
                (
                    torch.from_numpy(np.asarray(cab_mask, dtype=np.float32) / 255.0),
                    torch.from_numpy(np.asarray(body_mask, dtype=np.float32) / 255.0),
                )
            )
        return image, part_mask, pid, camera, view, image_path


class PartRegionTrainTransform:
    def __init__(self, cfg):
        self.size = tuple(cfg.INPUT.SIZE_TRAIN)
        self.probability = cfg.INPUT.PROB
        self.padding = cfg.INPUT.PADDING
        self.mean = cfg.INPUT.PIXEL_MEAN
        self.std = cfg.INPUT.PIXEL_STD
        self.soft_mask = bool(cfg.PART_MEMORY.SOFT_MASK)
        self.online_mask = bool(cfg.PART_MEMORY.ONLINE_MASK)
        self.random_erasing = RandomErasing(
            probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu"
        )

    def __call__(self, image, cab_mask, body_mask):
        image = TF.resize(image, self.size, interpolation=InterpolationMode.BICUBIC)
        interpolation = InterpolationMode.BILINEAR if self.soft_mask else InterpolationMode.NEAREST
        cab_mask = TF.resize(cab_mask, self.size, interpolation=interpolation)
        body_mask = TF.resize(body_mask, self.size, interpolation=interpolation)
        if torch.rand(1).item() < self.probability:
            image, cab_mask, body_mask = TF.hflip(image), TF.hflip(cab_mask), TF.hflip(body_mask)
        if self.padding > 0:
            image = TF.pad(image, self.padding)
            cab_mask = TF.pad(cab_mask, self.padding, fill=0)
            body_mask = TF.pad(body_mask, self.padding, fill=0)
        top, left, height, width = self._crop_parameters(image)
        image = TF.crop(image, top, left, height, width)
        cab_mask = TF.crop(cab_mask, top, left, height, width)
        body_mask = TF.crop(body_mask, top, left, height, width)
        image = TF.normalize(TF.to_tensor(image), mean=self.mean, std=self.std)
        clean_image = image.clone()
        image = self.random_erasing(image)
        masks = torch.cat((self._mask_tensor(cab_mask), self._mask_tensor(body_mask)))
        if self.online_mask:
            masks = torch.cat((masks, clean_image))
        return image, masks

    def _mask_tensor(self, mask):
        tensor = TF.to_tensor(mask).float()
        return tensor.clamp(0, 1) if self.soft_mask else (tensor > 0.5).float()

    def _crop_parameters(self, image):
        width, height = TF.get_image_size(image)
        target_h, target_w = self.size
        if (height, width) == self.size:
            return 0, 0, height, width
        top = int(torch.randint(0, height - target_h + 1, (1,)).item())
        left = int(torch.randint(0, width - target_w + 1, (1,)).item())
        return top, left, target_h, target_w


class PartRegionEvalTransform:
    def __init__(self, cfg):
        self.size = tuple(cfg.INPUT.SIZE_TEST)
        self.mean = cfg.INPUT.PIXEL_MEAN
        self.std = cfg.INPUT.PIXEL_STD
        self.soft_mask = bool(cfg.PART_MEMORY.SOFT_MASK)
        self.online_mask = bool(cfg.PART_MEMORY.ONLINE_MASK)

    def __call__(self, image, cab_mask, body_mask):
        image = TF.resize(image, self.size, interpolation=InterpolationMode.BICUBIC)
        interpolation = InterpolationMode.BILINEAR if self.soft_mask else InterpolationMode.NEAREST
        cab_mask = TF.resize(cab_mask, self.size, interpolation=interpolation)
        body_mask = TF.resize(body_mask, self.size, interpolation=interpolation)
        image = TF.normalize(TF.to_tensor(image), mean=self.mean, std=self.std)
        masks = torch.cat((self._mask_tensor(cab_mask), self._mask_tensor(body_mask)))
        if self.online_mask:
            masks = torch.cat((masks, image.clone()))
        return image, masks

    def _mask_tensor(self, mask):
        tensor = TF.to_tensor(mask).float()
        return tensor.clamp(0, 1) if self.soft_mask else (tensor > 0.5).float()

import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dinov3.data import make_dataset

try:
    import tifffile
except ImportError:
    tifffile = None


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
MASK_SUFFIXES = {".tif", ".tiff", ".png", ".bmp"}


def _read_raster(path: str) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix in {".tif", ".tiff"} and tifffile is not None:
        return tifffile.imread(path)
    with Image.open(path) as image:
        return np.array(image)


def _to_chw_tensor(array: np.ndarray) -> torch.Tensor:
    if array.ndim == 2:
        return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)
    if array.ndim == 3:
        if array.shape[0] <= 8 and array.shape[1] > 8 and array.shape[2] > 8:
            return torch.from_numpy(np.ascontiguousarray(array))
        return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    if array.ndim == 4 and array.shape[0] == 1:
        return _to_chw_tensor(array[0])
    raise ValueError(f"Unsupported raster shape: {array.shape}")


def _to_mask_tensor(array: np.ndarray) -> torch.Tensor:
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Unsupported mask shape: {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array))


def _collect_files(root: str, suffixes: set[str]) -> dict[str, str]:
    root_path = Path(root)
    mapping = {}
    for path in root_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        rel_key = str(path.relative_to(root_path).with_suffix("")).replace("\\", "/")
        mapping[rel_key] = str(path)
    return mapping


class TiffMaskPairsSegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transforms: Callable | None = None,
        image_suffixes: set[str] | None = None,
        mask_suffixes: set[str] | None = None,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        image_suffixes = image_suffixes or IMAGE_SUFFIXES
        mask_suffixes = mask_suffixes or MASK_SUFFIXES

        image_files = _collect_files(image_dir, image_suffixes)
        mask_files = _collect_files(mask_dir, mask_suffixes)
        shared_keys = sorted(set(image_files) & set(mask_files))
        if not shared_keys:
            raise RuntimeError(f"No matched image/mask pairs found in {image_dir} and {mask_dir}")

        self.image_paths = [image_files[key] for key in shared_keys]
        self.mask_paths = [mask_files[key] for key in shared_keys]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image = _to_chw_tensor(_read_raster(self.image_paths[index]))
        mask = _to_mask_tensor(_read_raster(self.mask_paths[index]))
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        return image, mask


def _resolve_split_dirs(config, split: str) -> tuple[str, str]:
    if split == "train":
        image_dir = config.datasets.train_images or os.path.join(config.datasets.root, "train", "images")
        mask_dir = config.datasets.train_masks or os.path.join(config.datasets.root, "train", "masks")
    else:
        image_dir = config.datasets.val_images or os.path.join(config.datasets.root, "val", "images")
        mask_dir = config.datasets.val_masks or os.path.join(config.datasets.root, "val", "masks")
    return image_dir, mask_dir


def build_segmentation_dataset(config, split: str, transforms: Callable):
    if config.datasets.format == "builtin":
        dataset_name = config.datasets.train if split == "train" else config.datasets.val
        return make_dataset(
            dataset_str=f"{dataset_name}:root={config.datasets.root}",
            transforms=transforms,
        )

    if config.datasets.format != "tif_mask_pairs":
        raise ValueError(f"Unsupported segmentation dataset format: {config.datasets.format}")

    image_dir, mask_dir = _resolve_split_dirs(config, split)
    return TiffMaskPairsSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transforms=transforms,
    )

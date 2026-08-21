import glob
import gzip
from math import ceil
from pathlib import Path
from typing import Literal

import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ..constants import PLANES

Split = Literal["train", "test"]


def _load_gzipped_array(path: Path) -> np.ndarray:
    with gzip.GzipFile(path, "rb") as f:
        return np.load(f)


class JawsDataset(Dataset):
    """Loads a DICOM slice + label pair from gzip-compressed .npy files.

    The image is resized with the default (bilinear) interpolation, but the
    label mask MUST be resized with nearest-neighbor interpolation: it holds
    integer class indices (0=background, 1=maxilla, 2=mandible), and bilinear
    interpolation would blend adjacent class indices into invalid fractional
    values at every mask boundary.
    """

    def __init__(self, dicom_file_list, image_height: int, image_width: int):
        self.dicom_file_list = dicom_file_list
        self.image_height = image_height
        self.image_width = image_width
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((image_height, image_width)),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((image_height, image_width), interpolation=InterpolationMode.NEAREST),
            ]
        )

    def __len__(self):
        return len(self.dicom_file_list)

    def _load_pair(self, idx):
        dicom_path = self.dicom_file_list[idx]
        label_path = str(dicom_path).replace(".dicom.npy.gz", ".label.npy.gz")
        dicom = _load_gzipped_array(dicom_path)
        label = _load_gzipped_array(label_path)
        return dicom, label

    def __getitem__(self, idx):
        dicom, label = self._load_pair(idx)
        return self.image_transform(dicom), self.mask_transform(label)


class TrainJawsDataset(JawsDataset):
    """Training-split dataset: adds random rotation (p=0.7) and horizontal
    flip (p=0.6) augmentation, applied identically to image and mask.

    The rotation angle and flip decision are drawn once per sample and applied
    to both tensors directly (via torchvision.transforms.functional), rather
    than through a shared transforms.Compose pipeline. A Compose of random
    transforms draws fresh randomness every time it's called, so calling it
    separately on the image and the mask would desync them — this manual,
    single-draw approach is a deliberate choice, not an oversight.
    """

    ROTATION_DEGREES = 35
    ROTATION_PROB = 0.7
    FLIP_PROB = 0.6

    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)

        if np.random.rand() < self.ROTATION_PROB:
            angle = self.ROTATION_DEGREES * (np.random.rand() * 2 - 1)
            center = (self.image_height // 2, self.image_width // 2)
            image = TF.rotate(image, angle, center=center)
            mask = TF.rotate(mask, angle, center=center)

        if np.random.rand() < self.FLIP_PROB:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        return image, mask


def _glob_slices(data_dir: Path, plane: str, split: Split) -> list[str]:
    if plane not in PLANES:
        raise ValueError(f"Unknown plane {plane!r}, expected one of {PLANES}")
    pattern = str(Path(data_dir) / plane / split / "**" / "*.dicom.npy.gz")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No slices found for plane={plane!r} split={split!r} under {pattern}")
    return files


def build_dataset(
    plane: str,
    split: Split,
    data_dir: Path,
    image_height: int,
    image_width: int,
    augment: bool = False,
):
    """Builds a dataset strictly scoped to one plane/split combination.

    Each split's files are globbed only from that split's own directory
    (dataset/<plane>/<split>/**), so it is structurally impossible to
    accidentally build a "test" dataset out of training files.
    """
    files = _glob_slices(data_dir, plane, split)
    cls = TrainJawsDataset if augment else JawsDataset
    return cls(files, image_height, image_width)


def split_train_val(dataset: JawsDataset, val_percent: float):
    n_val = ceil(len(dataset) * val_percent)
    val_files = dataset.dicom_file_list[:n_val]
    train_files = dataset.dicom_file_list[n_val:]
    cls = type(dataset)
    train_ds = cls(train_files, dataset.image_height, dataset.image_width)
    val_ds = JawsDataset(val_files, dataset.image_height, dataset.image_width)
    return train_ds, val_ds


def build_dataloader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

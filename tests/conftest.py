import gzip
from pathlib import Path

import numpy as np
import pytest

SLICE_H, SLICE_W = 32, 48


def _make_synthetic_slice(seed: int):
    rng = np.random.default_rng(seed)
    dicom = rng.random((SLICE_H, SLICE_W)).astype(np.float32)

    label = np.zeros((SLICE_H, SLICE_W), dtype=np.int64)
    label[4:12, 4:16] = 1  # class 1 = maxilla
    label[20:28, 20:36] = 2  # class 2 = mandible
    return dicom, label


def _write_gzipped_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(path, "wb") as f:
        np.save(f, array)


def write_synthetic_split(root: Path, plane: str, split: str, num_slices: int) -> list[Path]:
    dicom_paths = []
    for i in range(num_slices):
        case_dir = root / plane / split / "case1"
        dicom_path = case_dir / f"slice{i}.dicom.npy.gz"
        label_path = case_dir / f"slice{i}.label.npy.gz"
        dicom, label = _make_synthetic_slice(seed=i)
        _write_gzipped_npy(dicom_path, dicom)
        _write_gzipped_npy(label_path, label)
        dicom_paths.append(dicom_path)
    return dicom_paths


@pytest.fixture
def synthetic_dataset_dir(tmp_path):
    """Builds dataset/<plane>/{train,test}/case1/*.{dicom,label}.npy.gz under tmp_path."""

    def _build(plane="coronal", n_train=6, n_test=3):
        write_synthetic_split(tmp_path, plane, "train", n_train)
        write_synthetic_split(tmp_path, plane, "test", n_test)
        return tmp_path

    return _build

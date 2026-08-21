import torch
import torchvision.transforms.functional as TF

from jaws_seg.data.dataset import JawsDataset, TrainJawsDataset, build_dataset
from tests.conftest import write_synthetic_split


def test_jaws_dataset_len_and_shapes(synthetic_dataset_dir):
    root = synthetic_dataset_dir(plane="coronal", n_train=5, n_test=2)
    dataset = build_dataset("coronal", "train", root, image_height=64, image_width=96)
    assert len(dataset) == 5
    image, mask = dataset[0]
    assert image.shape == (1, 64, 96)
    assert mask.shape == (1, 64, 96)


def test_mask_resize_preserves_class_labels(synthetic_dataset_dir):
    """Regression test: mask must be resized with nearest-neighbor, not
    bilinear, or interpolated boundary pixels produce invalid fractional
    class indices once truncated to long."""
    root = synthetic_dataset_dir(plane="axial", n_train=3, n_test=1)
    dataset = build_dataset("axial", "train", root, image_height=64, image_width=96)
    for i in range(len(dataset)):
        _, mask = dataset[i]
        unique_values = set(torch.unique(mask).tolist())
        assert unique_values <= {0.0, 1.0, 2.0}, f"invalid mask values after resize: {unique_values}"


def test_glob_based_split_functions_dont_leak_test_into_train(synthetic_dataset_dir):
    root = synthetic_dataset_dir(plane="sagittal", n_train=4, n_test=2)
    train_ds = build_dataset("sagittal", "train", root, image_height=32, image_width=48)
    test_ds = build_dataset("sagittal", "test", root, image_height=32, image_width=48)

    train_files = {str(p) for p in train_ds.dicom_file_list}
    test_files = {str(p) for p in test_ds.dicom_file_list}

    assert len(train_files) == 4
    assert len(test_files) == 2
    assert train_files.isdisjoint(test_files)
    assert all("train" in f for f in train_files)
    assert all("test" in f for f in test_files)


def test_train_dataset_augmentation_synced(tmp_path, monkeypatch):
    files = write_synthetic_split(tmp_path, "coronal", "train", num_slices=1)
    base_dataset = JawsDataset(files, image_height=32, image_width=48)
    train_dataset = TrainJawsDataset(files, image_height=32, image_width=48)

    # Force: rotate branch taken with angle=21deg, flip branch skipped.
    responses = iter([0.1, 0.8, 0.9])
    monkeypatch.setattr("jaws_seg.data.dataset.np.random.rand", lambda: next(responses))

    base_image, base_mask = base_dataset[0]
    aug_image, aug_mask = train_dataset[0]

    expected_angle = TrainJawsDataset.ROTATION_DEGREES * (0.8 * 2 - 1)
    center = (16, 24)
    expected_image = TF.rotate(base_image, expected_angle, center=center)
    expected_mask = TF.rotate(base_mask, expected_angle, center=center)

    assert torch.allclose(aug_image, expected_image)
    assert torch.allclose(aug_mask, expected_mask)

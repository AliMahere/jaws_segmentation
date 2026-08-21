import pytest
import torch

from jaws_seg import evaluate, predict, train
from jaws_seg.models import UNet


@pytest.mark.parametrize("module", [train, evaluate, predict])
def test_help_exits_zero(module):
    with pytest.raises(SystemExit) as exc_info:
        module.build_arg_parser().parse_args(["--help"])
    assert exc_info.value.code == 0


def _write_random_checkpoint(path):
    model = UNet(n_channels=1, n_classes=3, bilinear=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def test_evaluate_end_to_end_on_synthetic_data(synthetic_dataset_dir, tmp_path):
    root = synthetic_dataset_dir(plane="coronal", n_train=4, n_test=3)
    checkpoint = tmp_path / "ckpt" / "coronal" / "checkpoint_epoch1.pth"
    _write_random_checkpoint(checkpoint)

    result = evaluate.main(
        [
            "--plane",
            "coronal",
            "--data-dir",
            str(root),
            "--checkpoint",
            str(checkpoint),
            "--split",
            "test",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--image-height",
            "32",
            "--image-width",
            "48",
        ]
    )

    assert 0.0 <= result["mean_dice"] <= 1.0
    assert result["num_samples"] == 3
    assert set(result["per_class_dice"]) == {"background", "maxilla", "mandible"}


def test_predict_saves_output_image(synthetic_dataset_dir, tmp_path):
    root = synthetic_dataset_dir(plane="axial", n_train=3, n_test=2)
    checkpoint = tmp_path / "ckpt" / "axial" / "checkpoint_epoch1.pth"
    _write_random_checkpoint(checkpoint)
    output_dir = tmp_path / "predictions"

    saved_paths = predict.main(
        [
            "--plane",
            "axial",
            "--data-dir",
            str(root),
            "--checkpoint",
            str(checkpoint),
            "--split",
            "test",
            "--num-samples",
            "2",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--image-height",
            "32",
            "--image-width",
            "48",
        ]
    )

    assert len(saved_paths) == 2
    for p in saved_paths:
        assert p.exists()

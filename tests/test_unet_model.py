from pathlib import Path

import pytest
import torch

from jaws_seg.models import UNet

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CHECKPOINT = REPO_ROOT / "checkpoints" / "axial" / "checkpoint_epoch19.pth"


def test_forward_shape_transposed_conv():
    model = UNet(n_channels=1, n_classes=3, bilinear=False)
    x = torch.randn(2, 1, 160, 240)
    out = model(x)
    assert out.shape == (2, 3, 160, 240)


def test_forward_shape_bilinear():
    model = UNet(n_channels=1, n_classes=3, bilinear=True)
    x = torch.randn(2, 1, 160, 240)
    out = model(x)
    assert out.shape == (2, 3, 160, 240)


def test_forward_shape_nonstandard_input():
    model = UNet(n_channels=1, n_classes=3, bilinear=False)
    x = torch.randn(1, 1, 65, 97)
    out = model(x)
    assert out.shape == (1, 3, 65, 97)


def test_state_dict_keys_match_transposed_conv_arch():
    model = UNet(n_channels=1, n_classes=3, bilinear=False)
    state_dict = model.state_dict()
    assert "up1.up.weight" in state_dict
    assert "up1.up.bias" in state_dict


@pytest.mark.skipif(not REAL_CHECKPOINT.exists(), reason="real 2022 checkpoint not present locally")
def test_load_real_checkpoint():
    model = UNet(n_channels=1, n_classes=3, bilinear=False)
    state_dict = torch.load(REAL_CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)

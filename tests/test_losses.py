import torch

from jaws_seg.losses import dice_loss
from jaws_seg.metrics import dice_coeff, multiclass_dice_coeff


def test_dice_coeff_perfect_overlap():
    mask = torch.zeros(4, 4)
    mask[1:3, 1:3] = 1
    score = dice_coeff(mask, mask.clone())
    assert torch.isclose(score, torch.tensor(1.0), atol=1e-4)


def test_dice_coeff_no_overlap():
    a = torch.zeros(4, 4)
    a[0, 0] = 1
    b = torch.zeros(4, 4)
    b[3, 3] = 1
    score = dice_coeff(a, b)
    assert torch.isclose(score, torch.tensor(0.0), atol=1e-4)


def test_dice_coeff_empty_masks_no_div_by_zero():
    a = torch.zeros(4, 4)
    b = torch.zeros(4, 4)
    score = dice_coeff(a, b)
    assert torch.isfinite(score)
    assert torch.isclose(score, torch.tensor(1.0), atol=1e-4)


def test_multiclass_dice_coeff_matches_manual_average():
    torch.manual_seed(0)
    pred = torch.zeros(2, 3, 4, 4)
    true = torch.zeros(2, 3, 4, 4)
    pred[:, 0] = 1
    true[:, 0] = 1
    pred[:, 1, 0, 0] = 1
    true[:, 1, 0, 0] = 1
    # channel 2 fully mismatched
    pred[:, 2, 0, 0] = 1
    true[:, 2, 1, 1] = 1

    result = multiclass_dice_coeff(pred, true, reduce_batch_first=False)
    manual = sum(dice_coeff(pred[:, c], true[:, c], reduce_batch_first=False) for c in range(3)) / 3
    assert torch.isclose(result, manual, atol=1e-4)


def test_dice_loss_bounds():
    pred = torch.rand(2, 3, 8, 8).softmax(dim=1)
    true = torch.zeros(2, 3, 8, 8)
    true[:, 0] = 1
    loss = dice_loss(pred, true, multiclass=True)
    assert 0.0 <= loss.item() <= 1.0


def test_dice_loss_identical_inputs_near_zero():
    x = torch.zeros(2, 3, 8, 8)
    x[:, 1] = 1
    loss = dice_loss(x, x.clone(), multiclass=True)
    assert loss.item() < 1e-3

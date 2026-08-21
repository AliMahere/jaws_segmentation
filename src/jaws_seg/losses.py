from torch import Tensor

from .metrics import dice_coeff, multiclass_dice_coeff


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False) -> Tensor:
    """Dice loss (objective to minimize), between 0 and 1."""
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .constants import CLASS_NAMES
from .metrics import dice_coeff, multiclass_dice_coeff


def _predict_one_hot(model, image, device):
    output = model(image)
    return F.one_hot(output.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()


@torch.no_grad()
def evaluate(model, dataloader, device) -> float:
    """Mean multiclass Dice score on a val/test loader, excluding background."""
    model.eval()
    num_batches = len(dataloader)
    if num_batches == 0:
        return 0.0

    dice_score = 0.0
    progress = tqdm(dataloader, total=num_batches, desc="Validation round", unit="batch", leave=False)
    for image, mask_true in progress:
        mask_true = mask_true.squeeze(1)
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true_onehot = F.one_hot(mask_true, model.n_classes).permute(0, 3, 1, 2).float()

        mask_pred_onehot = _predict_one_hot(model, image, device)
        dice_score += multiclass_dice_coeff(
            mask_pred_onehot[:, 1:, ...], mask_true_onehot[:, 1:, ...], reduce_batch_first=False
        ).item()

    model.train()
    return dice_score / num_batches


@torch.no_grad()
def evaluate_detailed(model, dataloader, device) -> dict:
    """Per-class Dice score (including background) plus the mean excluding it.

    Used by the `jaws-evaluate` CLI to produce the results table numbers,
    as opposed to `evaluate()` above which is the scalar used by the
    training loop's LR scheduler.
    """
    model.eval()
    num_batches = len(dataloader)
    per_class_sum = [0.0] * model.n_classes

    for image, mask_true in tqdm(dataloader, total=num_batches, desc="Evaluation", unit="batch", leave=False):
        mask_true = mask_true.squeeze(1)
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true_onehot = F.one_hot(mask_true, model.n_classes).permute(0, 3, 1, 2).float()

        mask_pred_onehot = _predict_one_hot(model, image, device)
        for c in range(model.n_classes):
            per_class_sum[c] += dice_coeff(
                mask_pred_onehot[:, c, ...], mask_true_onehot[:, c, ...], reduce_batch_first=False
            ).item()

    model.train()
    per_class = {
        CLASS_NAMES[c]: per_class_sum[c] / num_batches if num_batches else 0.0 for c in range(model.n_classes)
    }
    foreground = [v for name, v in per_class.items() if name != "background"]
    mean_dice = sum(foreground) / len(foreground) if foreground else 0.0
    return {"per_class_dice": per_class, "mean_dice": mean_dice, "num_batches": num_batches}

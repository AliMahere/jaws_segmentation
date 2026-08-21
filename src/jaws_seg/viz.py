from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def plot_triptych(
    image: torch.Tensor,
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    title: str,
    save_path: Path | None = None,
):
    """Plots input / predicted mask / ground-truth mask side by side.

    image, pred_mask, true_mask are expected as squeezable single-sample
    tensors (no batch dim, or batch dim of 1).
    """
    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("input")
    plt.axis("off")
    plt.imshow(image.squeeze().cpu().numpy(), cmap="bone")

    plt.subplot(1, 3, 2)
    plt.title("predicted mask")
    plt.axis("off")
    plt.imshow(pred_mask.squeeze().cpu().numpy())

    plt.subplot(1, 3, 3)
    plt.title("ground truth")
    plt.axis("off")
    plt.imshow(true_mask.squeeze().cpu().numpy())

    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@torch.no_grad()
def predict_mask(model, image: torch.Tensor, device: str) -> torch.Tensor:
    """Runs a single (unbatched or batch-of-1) image through the model and
    returns the predicted class-index mask (argmax over softmax probs)."""
    model.eval()
    output = model(image.to(device))
    probs = F.softmax(output, dim=1)
    return probs.argmax(dim=1).cpu()

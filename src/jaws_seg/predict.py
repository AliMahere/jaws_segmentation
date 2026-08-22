import argparse
from pathlib import Path

import torch

from .cli_common import add_common_args, resolve_checkpoint
from .config import default_device, set_seed
from .data.dataset import build_dataset
from .models import UNet
from .viz import plot_triptych, predict_mask


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference on a few real samples and save input/pred/true triptychs"
    )
    add_common_args(parser)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predictions"))
    return parser


def main(argv=None) -> list[Path]:
    args = build_arg_parser().parse_args(argv)
    set_seed(args.seed)
    device = args.device or default_device()

    checkpoint_path = resolve_checkpoint(args.checkpoint, args.checkpoint_dir, args.plane)

    dataset = build_dataset(
        args.plane, args.split, args.data_dir, args.image_height, args.image_width, augment=False
    )

    model = UNet(n_channels=1, n_classes=3, bilinear=args.bilinear).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    indices = torch.randperm(len(dataset))[: args.num_samples].tolist()
    saved_paths = []
    for i, idx in enumerate(indices):
        image, true_mask = dataset[idx]
        pred_mask = predict_mask(model, image.unsqueeze(0), device)

        save_path = Path(args.output_dir) / f"{args.plane}_sample{i}.png"
        plot_triptych(image, pred_mask, true_mask, title=f"{args.plane} sample {i}", save_path=save_path)
        saved_paths.append(save_path)
        print(f"saved {save_path}")

    return saved_paths


def cli(argv=None) -> None:
    """Entry point for the `jaws-predict` console script (see evaluate.cli)."""
    main(argv)


if __name__ == "__main__":
    main()

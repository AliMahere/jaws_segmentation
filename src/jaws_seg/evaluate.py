import argparse
import json
from pathlib import Path

import torch

from .cli_common import add_common_args, resolve_checkpoint
from .config import default_device, set_seed
from .data.dataset import build_dataloader, build_dataset
from .engine import evaluate_detailed
from .models import UNet


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a per-plane checkpoint on the real test/val split")
    add_common_args(parser)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv=None) -> dict:
    args = build_arg_parser().parse_args(argv)
    set_seed(args.seed)
    device = args.device or default_device()

    checkpoint_path = resolve_checkpoint(args.checkpoint, args.checkpoint_dir, args.plane)

    dataset = build_dataset(
        args.plane, args.split, args.data_dir, args.image_height, args.image_width, augment=False
    )
    loader = build_dataloader(dataset, args.batch_size, args.num_workers, shuffle=False)

    model = UNet(n_channels=1, n_classes=3, bilinear=args.bilinear).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)

    result = evaluate_detailed(model, loader, device)
    result["plane"] = args.plane
    result["split"] = args.split
    result["checkpoint"] = str(checkpoint_path)
    result["num_samples"] = len(dataset)

    print(json.dumps(result, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    main()

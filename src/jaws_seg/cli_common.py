import argparse
import re
from pathlib import Path

from .constants import PLANES


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plane", required=True, choices=PLANES)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--image-height", type=int, default=160)
    parser.add_argument("--image-width", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto-detect)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bilinear",
        action="store_true",
        help="Use bilinear upsampling instead of transposed conv. "
        "WARNING: the shipped 2022 checkpoints were trained with bilinear=False "
        "(transposed conv) — only pass this flag for a model trained that way.",
    )


def resolve_checkpoint(checkpoint: Path | None, checkpoint_dir: Path, plane: str) -> Path:
    if checkpoint is not None:
        return Path(checkpoint)

    plane_dir = Path(checkpoint_dir) / plane
    candidates = sorted(
        plane_dir.glob("checkpoint_epoch*.pth"),
        key=lambda p: int(re.search(r"epoch(\d+)", p.stem).group(1)),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {plane_dir}")
    return candidates[-1]

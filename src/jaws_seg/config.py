import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .constants import NUM_CLASSES


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class BaseConfig:
    plane: str = "coronal"
    data_dir: Path = field(default_factory=lambda: Path("dataset"))
    image_height: int = 160
    image_width: int = 240
    num_classes: int = NUM_CLASSES
    batch_size: int = 16
    num_workers: int = 10
    device: str = field(default_factory=default_device)
    seed: int = 42
    bilinear: bool = False  # must stay False to match the shipped 2022 checkpoints


@dataclass(frozen=True)
class TrainConfig(BaseConfig):
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-8
    momentum: float = 0.9
    val_percent: float = 0.1
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    amp: bool = False
    wandb_mode: str = "disabled"  # "disabled" | "online" | "offline"
    resume: Path | None = None


@dataclass(frozen=True)
class EvalConfig(BaseConfig):
    checkpoint: Path | None = None
    split: str = "test"
    output_json: Path | None = None


@dataclass(frozen=True)
class PredictConfig(BaseConfig):
    checkpoint: Path | None = None
    split: str = "test"
    num_samples: int = 3
    output_dir: Path = field(default_factory=lambda: Path("outputs/predictions"))

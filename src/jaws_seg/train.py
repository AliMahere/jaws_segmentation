import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cli_common import add_common_args
from .config import default_device, set_seed
from .data.dataset import build_dataloader, build_dataset, split_train_val
from .engine import evaluate
from .losses import dice_loss
from .models import UNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a per-plane jaw segmentation U-Net")
    add_common_args(parser)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-8)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--val-percent", type=float, default=0.1)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--wandb-mode", choices=["disabled", "online", "offline"], default="disabled")
    parser.add_argument("--resume", type=Path, default=None)
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    set_seed(args.seed)
    device = args.device or default_device()

    train_full = build_dataset(
        args.plane, "train", args.data_dir, args.image_height, args.image_width, augment=True
    )
    train_dataset, val_dataset = split_train_val(train_full, args.val_percent)
    train_loader = build_dataloader(train_dataset, args.batch_size, args.num_workers, shuffle=True)
    val_loader = build_dataloader(val_dataset, args.batch_size, args.num_workers, shuffle=False)

    n_train, n_val = len(train_dataset), len(val_dataset)
    log.info(f"plane={args.plane} train={n_train} val={n_val} device={device}")

    model = UNet(n_channels=1, n_classes=3, bilinear=args.bilinear).to(device)
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        log.info(f"resumed from {args.resume}")

    experiment = None
    if args.wandb_mode != "disabled":
        import wandb

        experiment = wandb.init(project="U-Net", mode=args.wandb_mode)
        experiment.config.update(
            dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, plane=args.plane)
        )

    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=2)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for images, true_masks in train_loader:
            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long).squeeze(1)

            with torch.cuda.amp.autocast(enabled=args.amp):
                masks_pred = model(images)
                loss = criterion(masks_pred, true_masks) + dice_loss(
                    F.softmax(masks_pred, dim=1).float(),
                    F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                    multiclass=True,
                )

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            global_step += 1
            epoch_loss += loss.item()
            if experiment is not None:
                experiment.log({"train_loss": loss.item(), "step": global_step, "epoch": epoch})

        val_score = evaluate(model, val_loader, device)
        scheduler.step(val_score)
        avg_loss = epoch_loss / max(len(train_loader), 1)
        log.info(f"epoch {epoch + 1}/{args.epochs} loss={avg_loss:.4f} val_dice={val_score:.4f}")
        if experiment is not None:
            experiment.log({"val_dice": val_score, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"]})

        save_dir = Path(args.checkpoint_dir) / args.plane
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / f"checkpoint_epoch{epoch + 1}.pth")
        log.info(f"checkpoint {epoch + 1} saved")


if __name__ == "__main__":
    main()

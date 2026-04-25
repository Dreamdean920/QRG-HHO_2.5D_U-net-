from pathlib import Path
import sys

# ========= 把项目根目录加入 Python 路径 =========
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from datasets.dataset_2d import LungSliceDataset2D
from models.unet_2d import UNet2D
from utils.losses import BCEDiceLoss
from utils.metrics import binary_dice_score, binary_iou_score


def load_config():
    with open("configs/baseline.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    dice_list = []
    iou_list = []

    for batch in tqdm(loader, desc="Val", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item() * images.size(0)
        dice_list.append(binary_dice_score(logits, masks))
        iou_list.append(binary_iou_score(logits, masks))

    val_loss = total_loss / len(loader.dataset)
    val_dice = float(np.mean(dice_list))
    val_iou = float(np.mean(iou_list))

    return val_loss, val_dice, val_iou


def main():
    cfg = load_config()
    seed = int(cfg["seed"])
    set_seed(seed)

    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    lr = float(cfg["train"]["lr"])
    num_workers = int(cfg["train"].get("num_workers", 0))

    patience = int(cfg["early_stopping"]["patience"])

    bce_weight = float(cfg["loss"].get("bce_weight", 0.5))
    dice_weight = float(cfg["loss"].get("dice_weight", 0.5))

    ckpt_dir = Path("outputs/checkpoints")
    log_dir = Path("outputs/logs")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = LungSliceDataset2D(split="train")
    val_dataset = LungSliceDataset2D(split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model = UNet2D(
        in_channels=int(cfg["model"]["in_channels"]),
        num_classes=int(cfg["model"]["num_classes"]),
        base_ch=32
    ).to(device)

    criterion = BCEDiceLoss(
        bce_weight=bce_weight,
        dice_weight=dice_weight
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_dice = -1.0
    best_epoch = -1
    early_stop_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou = validate_one_epoch(model, val_loader, criterion, device)

        print(f"train_loss = {train_loss:.6f}")
        print(f"val_loss   = {val_loss:.6f}")
        print(f"val_dice   = {val_dice:.6f}")
        print(f"val_iou    = {val_iou:.6f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou
        })

        pd.DataFrame(history).to_csv(log_dir / "baseline_train_log.csv", index=False)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            early_stop_counter = 0

            ckpt_path = ckpt_dir / "baseline_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice
            }, ckpt_path)
            print(f"[保存] best checkpoint -> {ckpt_path}")
        else:
            early_stop_counter += 1
            print(f"[EarlyStopping] counter = {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    print("\n训练完成")
    print(f"best_val_dice = {best_val_dice:.6f}")
    print(f"best_epoch    = {best_epoch}")


if __name__ == "__main__":
    main()
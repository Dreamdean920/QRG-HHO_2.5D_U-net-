from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from datasets.dataset_2d import LungSliceDataset2D
from models.unet_2d import UNet2D
from utils.postprocess import postprocess_prediction


def load_config():
    with open("configs/baseline.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dice_from_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)

    return float((2.0 * intersection + eps) / (union + eps))


def iou_from_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    intersection = np.sum(pred * gt)
    union = np.sum((pred + gt) > 0)

    return float((intersection + eps) / (union + eps))


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, keep_k=2, min_area=50):
    model.eval()

    raw_dice_list = []
    raw_iou_list = []

    post_dice_list = []
    post_iou_list = []

    for batch in tqdm(loader, desc="Test", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()[:, 0]   # [B, H, W]
        gts = masks.cpu().numpy()[:, 0]                     # [B, H, W]

        for prob, gt in zip(probs, gts):
            raw_pred = (prob > threshold).astype(np.uint8)
            post_pred = postprocess_prediction(
                prob,
                threshold=threshold,
                keep_k=keep_k,
                min_area=min_area
            )

            raw_dice_list.append(dice_from_binary(raw_pred, gt))
            raw_iou_list.append(iou_from_binary(raw_pred, gt))

            post_dice_list.append(dice_from_binary(post_pred, gt))
            post_iou_list.append(iou_from_binary(post_pred, gt))

    return {
        "raw_test_dice": float(np.mean(raw_dice_list)),
        "raw_test_iou": float(np.mean(raw_iou_list)),
        "post_test_dice": float(np.mean(post_dice_list)),
        "post_test_iou": float(np.mean(post_iou_list)),
    }


def main():
    cfg = load_config()

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 0))

    ckpt_path = Path("outputs/checkpoints/baseline_best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_dataset = LungSliceDataset2D(split="test")
    test_loader = DataLoader(
        test_dataset,
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

    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # 后处理参数
    threshold = 0.5
    keep_k = 2
    min_area = 50

    results = evaluate(
        model,
        test_loader,
        device,
        threshold=threshold,
        keep_k=keep_k,
        min_area=min_area
    )

    print("\n===== Test Results =====")
    for k, v in results.items():
        print(f"{k} = {v:.6f}")

    results.update({
        "threshold": threshold,
        "keep_k": keep_k,
        "min_area": min_area
    })

    pd.DataFrame([results]).to_csv(
        metrics_dir / "baseline_test_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n测试结果已保存到: {metrics_dir / 'baseline_test_metrics.csv'}")


if __name__ == "__main__":
    main()
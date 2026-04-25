from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np

from datasets.dataset_2d import LungSliceDataset2D
from models.unet_2d import UNet2D
from utils.postprocess import postprocess_prediction


def load_config():
    with open("configs/baseline.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def main():
    random.seed(42)

    cfg = load_config()
    batch_size = 1

    vis_dir = Path("outputs/visuals")
    vis_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path("outputs/checkpoints/baseline_best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_dataset = LungSliceDataset2D(split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
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

    model.eval()

    saved = 0
    max_save = 6

    threshold = 0.5
    keep_k = 2
    min_area = 50

    for batch in test_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        case_name = batch["case_name"][0]
        slice_idx = int(batch["slice_idx"][0])

        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]  # [H, W]

        raw_pred = (probs > threshold).astype(np.uint8)
        post_pred = postprocess_prediction(
            probs,
            threshold=threshold,
            keep_k=keep_k,
            min_area=min_area
        )

        image = images[0, 0].cpu().numpy()
        gt = masks[0, 0].cpu().numpy()

        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("CT")
        axes[0].axis("off")

        axes[1].imshow(gt, cmap="gray")
        axes[1].set_title("GT")
        axes[1].axis("off")

        axes[2].imshow(raw_pred, cmap="gray")
        axes[2].set_title("Raw Pred")
        axes[2].axis("off")

        axes[3].imshow(post_pred, cmap="gray")
        axes[3].set_title("Post Pred")
        axes[3].axis("off")

        axes[4].imshow(image, cmap="gray")
        axes[4].imshow(post_pred, cmap="jet", alpha=0.35)
        axes[4].set_title("Overlay(Post)")
        axes[4].axis("off")

        fig.suptitle(f"{case_name} | slice={slice_idx}")
        plt.tight_layout()

        save_path = vis_dir / f"{case_name.replace('.nii.gz','')}_z{slice_idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        saved += 1
        if saved >= max_save:
            break

    print(f"预测可视化已保存到: {vis_dir}")


if __name__ == "__main__":
    main()
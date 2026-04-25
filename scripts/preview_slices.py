from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import random


def load_config():
    with open("configs/baseline.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_slice(x):
    x = x.astype(np.float32)
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def save_preview(ct_slice, mask_slice, save_path, title="preview"):
    ct_norm = normalize_slice(ct_slice)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(ct_norm, cmap="gray")
    axes[0].set_title("CT")
    axes[0].axis("off")

    axes[1].imshow(mask_slice, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    axes[2].imshow(ct_norm, cmap="gray")
    axes[2].imshow(mask_slice, cmap="jet", alpha=0.35)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def process_one_csv(csv_path, ct_dir, mask_dir, preview_dir, max_samples=5):
    if not csv_path.exists():
        print(f"[跳过] 不存在: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print(f"[跳过] 空文件: {csv_path}")
        return

    indices = list(range(len(df)))
    random.shuffle(indices)
    indices = indices[:min(max_samples, len(indices))]

    for i, idx in enumerate(indices):
        row = df.iloc[idx]
        case_name = row["case_name"]
        slice_idx = int(row["slice_idx"])

        ct_path = ct_dir / case_name
        mask_path = mask_dir / case_name

        ct = nib.load(str(ct_path)).get_fdata()
        mask = nib.load(str(mask_path)).get_fdata()

        ct_slice = ct[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]

        save_path = preview_dir / f"{csv_path.stem}_{i+1}_{case_name.replace('.nii.gz', '')}_z{slice_idx}.png"
        title = f"{csv_path.stem} | {case_name} | z={slice_idx}"
        save_preview(ct_slice, mask_slice, save_path, title=title)

    print(f"[完成] 预览图保存到: {preview_dir}")


def main():
    random.seed(42)

    cfg = load_config()
    ct_dir = Path(cfg["data"]["ct_dir"])
    mask_dir = Path(cfg["data"]["mask_dir"])
    results_dir = Path(cfg["data"]["report_dir"])
    preview_dir = Path(cfg["data"]["preview_dir"])

    preview_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        csv_path = results_dir / f"{split_name}_slices.csv"
        process_one_csv(csv_path, ct_dir, mask_dir, preview_dir, max_samples=5)


if __name__ == "__main__":
    main()
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
import pandas as pd
import nibabel as nib
import numpy as np
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


def main():
    random.seed(42)

    cfg = load_config()
    ct_dir = Path(cfg["data"]["ct_dir"])
    mask_dir = Path(cfg["data"]["mask_dir"])
    results_dir = Path(cfg["data"]["report_dir"])

    csv_path = results_dir / "train_slices.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到: {csv_path}，请先运行 build_slice_index.py")

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise RuntimeError("train_slices.csv 是空的，说明没有可用切片。")

    idx = random.randint(0, len(df) - 1)
    row = df.iloc[idx]

    case_name = row["case_name"]
    slice_idx = int(row["slice_idx"])

    ct_path = ct_dir / case_name
    mask_path = mask_dir / case_name

    ct = nib.load(str(ct_path)).get_fdata()
    mask = nib.load(str(mask_path)).get_fdata()

    ct_slice = ct[:, :, slice_idx]
    mask_slice = mask[:, :, slice_idx]

    ct_norm = normalize_slice(ct_slice)

    print("=== 2D Slice 数据测试 ===")
    print("case_name:", case_name)
    print("slice_idx:", slice_idx)
    print("ct_slice shape:", ct_slice.shape)
    print("mask_slice shape:", mask_slice.shape)
    print("ct min/max:", float(np.min(ct_slice)), float(np.max(ct_slice)))
    print("ct_norm min/max:", float(np.min(ct_norm)), float(np.max(ct_norm)))
    print("mask unique values:", np.unique(mask_slice))
    print("mask foreground pixels:", int(np.sum(mask_slice > 0)))


if __name__ == "__main__":
    main()
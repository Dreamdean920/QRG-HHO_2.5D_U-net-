from pathlib import Path
import yaml
import pandas as pd
import nibabel as nib
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


def load_config():
    with open("configs/baseline.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_ct_slice(x: np.ndarray) -> np.ndarray:
    """
    恢复版：逐切片 min-max 归一化
    这就是你之前能跑到 0.9084 的核心归一化方式
    """
    x = x.astype(np.float32)
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


class LungSliceDataset2D(Dataset):
    def __init__(self, split="train"):
        super().__init__()

        cfg = load_config()
        self.ct_dir = Path(cfg["data"]["ct_dir"])
        self.mask_dir = Path(cfg["data"]["mask_dir"])
        self.results_dir = Path(cfg["data"]["report_dir"])
        self.input_size = int(cfg["train"]["input_size"])
        self.split = split

        csv_path = self.results_dir / f"{split}_slices.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"找不到切片索引文件: {csv_path}")

        self.df = pd.read_csv(csv_path)
        if len(self.df) == 0:
            raise RuntimeError(f"{csv_path} 是空文件，没有可用样本。")

        self.cache = {}

    def __len__(self):
        return len(self.df)

    def _load_case(self, case_name):
        if case_name not in self.cache:
            ct_path = self.ct_dir / case_name
            mask_path = self.mask_dir / case_name

            ct = nib.load(str(ct_path)).get_fdata().astype(np.float32)
            mask = nib.load(str(mask_path)).get_fdata().astype(np.float32)

            self.cache[case_name] = (ct, mask)

        return self.cache[case_name]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_name = row["case_name"]
        slice_idx = int(row["slice_idx"])

        ct, mask = self._load_case(case_name)

        ct_slice = ct[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]

        # ===== 二分类：肺整体 =====
        mask_slice = (mask_slice > 0).astype(np.float32)

        # ===== 恢复版归一化：逐切片 min-max =====
        ct_slice = normalize_ct_slice(ct_slice)

        # ===== resize 到固定输入尺寸 =====
        ct_slice = cv2.resize(
            ct_slice,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR
        )

        mask_slice = cv2.resize(
            mask_slice,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_NEAREST
        )

        # 再次确保 mask 为二值
        mask_slice = (mask_slice > 0.5).astype(np.float32)

        # [H, W] -> [1, H, W]
        ct_slice = np.expand_dims(ct_slice, axis=0)
        mask_slice = np.expand_dims(mask_slice, axis=0)

        image = torch.tensor(ct_slice, dtype=torch.float32)
        mask = torch.tensor(mask_slice, dtype=torch.float32)

        return {
            "image": image,
            "mask": mask,
            "case_name": case_name,
            "slice_idx": slice_idx
        }
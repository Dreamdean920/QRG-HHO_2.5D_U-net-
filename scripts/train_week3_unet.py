# -*- coding: utf-8 -*-
r"""
第二周增强版 baseline / 第三周 HGS 兼容版：
1. 2D baseline
2. 2.5D 输入
3. Dice+BCE / Dice+BCE+BoundaryLoss
4. 后处理消融
5. 输出 json/csv/best checkpoint
6. 修复慢速问题：volume 预加载缓存，避免每个 slice 反复读取 nii.gz
7. 新增 HGS / QRG-HGS 兼容参数：
   - --dice_weight
   - --run_name
   - --output_json
   - --no_post
"""

import os
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import nibabel as nib
from scipy.ndimage import distance_transform_edt, label
from sklearn.model_selection import train_test_split


# =========================
# 1. 通用工具
# =========================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_nifti(path: str) -> np.ndarray:
    arr = nib.load(path).get_fdata()
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = np.squeeze(arr)
    return arr


def minmax_norm_slice(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    vmin, vmax = np.percentile(x, 1), np.percentile(x, 99)
    x = np.clip(x, vmin, vmax)
    if vmax - vmin < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - vmin) / (vmax - vmin + 1e-8)
    return x.astype(np.float32)


def resize_to(x: np.ndarray, out_h: int, out_w: int, order: int = 1) -> np.ndarray:
    xt = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()
    if order == 1:
        yt = F.interpolate(xt, size=(out_h, out_w), mode="bilinear", align_corners=False)
    else:
        yt = F.interpolate(xt, size=(out_h, out_w), mode="nearest")
    return yt.squeeze().cpu().numpy()


# =========================
# 2. 数据配对与切片索引
# =========================
def find_case_pairs(ct_dir: str, mask_dir: str) -> List[Tuple[str, str, str]]:
    ct_files = sorted([p for p in Path(ct_dir).glob("*.nii*")])
    mask_files = sorted([p for p in Path(mask_dir).glob("*.nii*")])

    mask_map = {p.name: str(p) for p in mask_files}
    pairs = []
    for ct in ct_files:
        if ct.name in mask_map:
            pairs.append((ct.name, str(ct), mask_map[ct.name]))
    return pairs


def split_by_case(
    pairs: List[Tuple[str, str, str]],
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
):
    case_names = [x[0] for x in pairs]
    trainval_cases, test_cases = train_test_split(case_names, test_size=test_size, random_state=seed)
    train_cases, val_cases = train_test_split(trainval_cases, test_size=val_size, random_state=seed)

    train_pairs = [x for x in pairs if x[0] in train_cases]
    val_pairs = [x for x in pairs if x[0] in val_cases]
    test_pairs = [x for x in pairs if x[0] in test_cases]
    return train_pairs, val_pairs, test_pairs


def build_slice_records(
    pairs: List[Tuple[str, str, str]],
    image_size: int = 256,
    min_mask_pixels: int = 64,
    remove_edge_dominant: bool = True,
    edge_ratio_thr: float = 0.60,
) -> List[Dict[str, Any]]:
    records = []

    for case_name, ct_path, mask_path in tqdm(pairs, desc="Building slice records"):
        ct_vol = load_nifti(ct_path)
        mask_vol = load_nifti(mask_path)

        if ct_vol.shape != mask_vol.shape:
            print(f"[WARN] shape mismatch, skip: {case_name}, ct={ct_vol.shape}, mask={mask_vol.shape}")
            continue

        if ct_vol.ndim != 3:
            print(f"[WARN] not 3D volume, skip: {case_name}, shape={ct_vol.shape}")
            continue

        zdim = ct_vol.shape[2]
        for z in range(zdim):
            mask_slice = mask_vol[:, :, z]
            fg = np.sum(mask_slice > 0)
            if fg < min_mask_pixels:
                continue

            if remove_edge_dominant:
                h, w = mask_slice.shape
                edge_band = 10
                edge_mask = np.zeros_like(mask_slice, dtype=np.uint8)
                edge_mask[:edge_band, :] = 1
                edge_mask[-edge_band:, :] = 1
                edge_mask[:, :edge_band] = 1
                edge_mask[:, -edge_band:] = 1

                fg_mask = (mask_slice > 0).astype(np.uint8)
                edge_fg = np.sum(fg_mask * edge_mask)
                all_fg = np.sum(fg_mask)
                if all_fg > 0 and (edge_fg / (all_fg + 1e-8)) > edge_ratio_thr:
                    continue

            records.append({
                "case_name": case_name,
                "ct_path": ct_path,
                "mask_path": mask_path,
                "z": z,
                "image_size": image_size,
            })

    return records


# =========================
# 3. 数据集（修复慢速版）
# =========================
class LungSliceDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        input_mode: str = "2d",   # "2d" or "2.5d"
        image_size: int = 256,
        augment: bool = False,
    ):
        self.records = records
        self.input_mode = input_mode
        self.image_size = image_size
        self.augment = augment

        self.case_cache = {}
        self._build_cache()

    def _build_cache(self):
        case_dict = {}
        for rec in self.records:
            case_name = rec["case_name"]
            if case_name not in case_dict:
                case_dict[case_name] = (rec["ct_path"], rec["mask_path"])

        print(f"[INFO] caching {len(case_dict)} volumes into memory...")
        for case_name, (ct_path, mask_path) in tqdm(case_dict.items(), desc="Caching volumes"):
            ct_vol = load_nifti(ct_path)
            mask_vol = load_nifti(mask_path)

            if ct_vol.ndim == 4:
                ct_vol = np.squeeze(ct_vol)
            if mask_vol.ndim == 4:
                mask_vol = np.squeeze(mask_vol)

            self.case_cache[case_name] = {
                "ct": ct_vol,
                "mask": mask_vol,
            }

    def __len__(self) -> int:
        return len(self.records)

    def _read_slice(self, vol: np.ndarray, z: int) -> np.ndarray:
        z = max(0, min(z, vol.shape[2] - 1))
        x = vol[:, :, z]
        x = minmax_norm_slice(x)
        x = resize_to(x, self.image_size, self.image_size, order=1)
        return x.astype(np.float32)

    def _read_mask(self, vol: np.ndarray, z: int) -> np.ndarray:
        z = max(0, min(z, vol.shape[2] - 1))
        x = vol[:, :, z]
        x = (x > 0).astype(np.float32)
        x = resize_to(x, self.image_size, self.image_size, order=0)
        x = (x > 0.5).astype(np.float32)
        return x.astype(np.float32)

    def _augment(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            img = np.flip(img, axis=-1).copy()
            mask = np.flip(mask, axis=-1).copy()

        if random.random() < 0.5:
            img = np.flip(img, axis=-2).copy()
            mask = np.flip(mask, axis=-2).copy()

        if random.random() < 0.3:
            k = random.choice([1, 2, 3])
            if img.ndim == 3:
                img = np.rot90(img, k, axes=(1, 2)).copy()
            else:
                img = np.rot90(img, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()

        return img, mask

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        case_name = rec["case_name"]
        z = rec["z"]

        ct_vol = self.case_cache[case_name]["ct"]
        mask_vol = self.case_cache[case_name]["mask"]

        if self.input_mode == "2d":
            img = self._read_slice(ct_vol, z)[None, :, :]
        elif self.input_mode == "2.5d":
            s1 = self._read_slice(ct_vol, z - 1)
            s2 = self._read_slice(ct_vol, z)
            s3 = self._read_slice(ct_vol, z + 1)
            img = np.stack([s1, s2, s3], axis=0)
        else:
            raise ValueError(f"Unsupported input_mode={self.input_mode}")

        mask = self._read_mask(mask_vol, z)

        if self.augment:
            img, mask = self._augment(img, mask)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        meta = {
            "case_name": case_name,
            "z": z,
        }
        return img, mask, meta


# =========================
# 4. U-Net
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_ch: int = 32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 16)

        self.up1 = Up(base_ch * 16 + base_ch * 8, base_ch * 8)
        self.up2 = Up(base_ch * 8 + base_ch * 4, base_ch * 4)
        self.up3 = Up(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.up4 = Up(base_ch * 2 + base_ch, base_ch)
        self.outc = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# =========================
# 5. Loss
# =========================
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        inter = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def compute_sdf_np(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.bool_)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.float32)

    posmask = mask
    negmask = ~posmask

    dist_out = distance_transform_edt(negmask)
    dist_in = distance_transform_edt(posmask)
    sdf = dist_out - dist_in
    return sdf.astype(np.float32)


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        b = targets.shape[0]
        loss_list = []

        for i in range(b):
            gt = targets[i, 0].detach().cpu().numpy()
            sdf = compute_sdf_np(gt)
            sdf_t = torch.from_numpy(sdf).to(logits.device).float()
            loss_i = torch.mean(probs[i, 0] * sdf_t)
            loss_list.append(loss_i)

        return torch.stack(loss_list).mean()


class CombinedLoss(nn.Module):
    def __init__(
        self,
        use_boundary_loss: bool = False,
        dice_weight: float = 0.7,
        boundary_weight: float = 0.1
    ):
        super().__init__()
        self.use_boundary_loss = use_boundary_loss
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.boundary_loss = BoundaryLoss()

    def forward(self, logits, targets):
        ld = self.dice_loss(logits, targets)
        lb = self.bce_loss(logits, targets)
        loss = self.dice_weight * ld + (1 - self.dice_weight) * lb

        if self.use_boundary_loss:
            lbd = self.boundary_loss(logits, targets)
            loss = loss + self.boundary_weight * lbd

        return loss


# =========================
# 6. 后处理与指标
# =========================
def postprocess_mask(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    labeled, num = label(mask)

    if num == 0:
        return mask.astype(np.uint8)

    areas = []
    for i in range(1, num + 1):
        area = np.sum(labeled == i)
        areas.append((i, area))

    areas = sorted(areas, key=lambda x: x[1], reverse=True)
    keep_ids = [cid for cid, area in areas[:2] if area >= min_size]

    out = np.zeros_like(mask, dtype=np.uint8)
    for cid in keep_ids:
        out[labeled == cid] = 1
    return out


def compute_metrics_from_binary(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    tp = np.sum((pred == 1) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))

    dice = (2 * tp + 1e-8) / (2 * tp + fp + fn + 1e-8)
    iou = (tp + 1e-8) / (tp + fp + fn + 1e-8)
    sens = (tp + 1e-8) / (tp + fn + 1e-8)
    spec = (tn + 1e-8) / (tn + fp + 1e-8)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "sens": float(sens),
        "spec": float(spec),
    }


# =========================
# 7. 训练与评估
# =========================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    postprocess: bool = True,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    metrics_list = []

    for imgs, masks, _ in tqdm(loader, desc="Eval", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)
        loss = criterion(logits, masks)
        total_loss += loss.item() * imgs.size(0)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        gts = masks.detach().cpu().numpy()

        for i in range(probs.shape[0]):
            pred = (probs[i, 0] > 0.5).astype(np.uint8)
            gt = (gts[i, 0] > 0.5).astype(np.uint8)

            if postprocess:
                pred = postprocess_mask(pred)

            metrics_list.append(compute_metrics_from_binary(pred, gt))

    avg_loss = total_loss / max(len(loader.dataset), 1)

    out = {"loss": avg_loss}
    for key in ["dice", "iou", "sens", "spec"]:
        out[key] = float(np.mean([m[key] for m in metrics_list])) if metrics_list else 0.0
    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0

    for imgs, masks, _ in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / max(len(loader.dataset), 1)


# =========================
# 8. 主函数
# =========================
def main():
    parser = argparse.ArgumentParser()

    # 原 week2 参数
    parser.add_argument("--ct_dir", type=str, default=r"C:\Users\13178\Documents\MATLAB\医学图像\COVID-19-CT-Seg_20cases")
    parser.add_argument("--mask_dir", type=str, default=r"C:\Users\13178\Documents\MATLAB\医学图像\Lung_Mask")
    parser.add_argument("--save_root", type=str, default="outputs/week2_runs")

    parser.add_argument("--exp_name", type=str, default="baseline_2d")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_mode", "--input-mode", dest="input_mode", type=str, default="2d")
    parser.add_argument("--use_boundary_loss", action="store_true")
    parser.add_argument("--postprocess", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dice_weight", "--dice-weight", dest="dice_weight", type=float, default=0.7)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--min_mask_pixels", type=int, default=64)
    parser.add_argument("--remove_edge_dominant", type=int, default=1)
    parser.add_argument("--edge_ratio_thr", type=float, default=0.60)

    # 第三周 HGS / QRG-HGS 兼容参数
    parser.add_argument("--run_name", "--run-name", dest="run_name", type=str, default=None)
    parser.add_argument("--output_json", "--output-json", dest="output_json", type=str, default="")
    parser.add_argument("--no_post", "--no-post", dest="no_post", action="store_true")

    args = parser.parse_args()

    if args.input_mode == "25d":
        args.input_mode = "2.5d"

    if args.input_mode not in ["2d", "2.5d"]:
        raise ValueError(f"Unsupported input_mode={args.input_mode}, only support '2d' or '2.5d'")

    # 兼容搜索器参数
    if args.run_name is not None:
        args.exp_name = args.run_name

    if args.no_post:
        args.postprocess = 0

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    save_dir = os.path.join(args.save_root, args.exp_name)
    ensure_dir(save_dir)
    ensure_dir(os.path.join(save_dir, "checkpoints"))

    pairs = find_case_pairs(args.ct_dir, args.mask_dir)
    print(f"[INFO] matched cases = {len(pairs)}")
    if len(pairs) == 0:
        raise RuntimeError("没有匹配到 CT/Mask 文件，请检查文件名是否一致。")

    train_pairs, val_pairs, test_pairs = split_by_case(pairs, seed=args.seed)
    print(f"[INFO] train/val/test cases = {len(train_pairs)}/{len(val_pairs)}/{len(test_pairs)}")

    train_records = build_slice_records(
        train_pairs,
        image_size=args.image_size,
        min_mask_pixels=args.min_mask_pixels,
        remove_edge_dominant=bool(args.remove_edge_dominant),
        edge_ratio_thr=args.edge_ratio_thr,
    )
    val_records = build_slice_records(
        val_pairs,
        image_size=args.image_size,
        min_mask_pixels=args.min_mask_pixels,
        remove_edge_dominant=bool(args.remove_edge_dominant),
        edge_ratio_thr=args.edge_ratio_thr,
    )
    test_records = build_slice_records(
        test_pairs,
        image_size=args.image_size,
        min_mask_pixels=args.min_mask_pixels,
        remove_edge_dominant=bool(args.remove_edge_dominant),
        edge_ratio_thr=args.edge_ratio_thr,
    )

    print(f"[INFO] train/val/test slices = {len(train_records)}/{len(val_records)}/{len(test_records)}")

    split_info = {
        "seed": args.seed,
        "train_cases": [x[0] for x in train_pairs],
        "val_cases": [x[0] for x in val_pairs],
        "test_cases": [x[0] for x in test_pairs],
        "num_train_slices": len(train_records),
        "num_val_slices": len(val_records),
        "num_test_slices": len(test_records),
    }
    save_json(split_info, os.path.join(save_dir, "split_info.json"))

    train_ds = LungSliceDataset(train_records, input_mode=args.input_mode, image_size=args.image_size, augment=True)
    val_ds = LungSliceDataset(val_records, input_mode=args.input_mode, image_size=args.image_size, augment=False)
    test_ds = LungSliceDataset(test_records, input_mode=args.input_mode, image_size=args.image_size, augment=False)

    pin_memory = True if device.type == "cuda" else False

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    in_channels = 1 if args.input_mode == "2d" else 3
    model = UNet(in_channels=in_channels, out_channels=1, base_ch=args.base_ch).to(device)

    criterion = CombinedLoss(
        use_boundary_loss=args.use_boundary_loss,
        dice_weight=args.dice_weight,
        boundary_weight=0.1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_dice = -1.0
    best_epoch = -1
    no_improve = 0
    history = []

    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\n========== Epoch {epoch}/{args.epochs} ==========")

        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
            postprocess=bool(args.postprocess),
        )
        epoch_time = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_dice": float(val_metrics["dice"]),
            "val_iou": float(val_metrics["iou"]),
            "val_sens": float(val_metrics["sens"]),
            "val_spec": float(val_metrics["spec"]),
            "epoch_time_sec": float(epoch_time),
        }
        history.append(row)

        print(row)

        pd.DataFrame(history).to_csv(os.path.join(save_dir, "train_history.csv"), index=False)

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            no_improve = 0

            ckpt_path = os.path.join(save_dir, "checkpoints", "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_dice": best_val_dice,
                "args": vars(args),
            }, ckpt_path)
            print(f"[INFO] saved best checkpoint to {ckpt_path}")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"[INFO] early stopping at epoch={epoch}")
            break

    total_time_sec = time.time() - total_start_time

    best_ckpt_path = os.path.join(save_dir, "checkpoints", "best.pt")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        criterion,
        postprocess=bool(args.postprocess),
    )

    summary = {
        "exp_name": args.exp_name,
        "run_name": args.exp_name,
        "seed": args.seed,
        "input_mode": args.input_mode,
        "use_boundary_loss": bool(args.use_boundary_loss),
        "postprocess": bool(args.postprocess),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": float(args.lr),
        "dice_weight": float(args.dice_weight),
        "best_epoch": best_epoch,
        "best_val_dice": float(best_val_dice),
        "val_dice": float(best_val_dice),
        "test_dice": float(test_metrics["dice"]),
        "test_iou": float(test_metrics["iou"]),
        "test_sens": float(test_metrics["sens"]),
        "test_spec": float(test_metrics["spec"]),
        "fitness": float(-best_val_dice),   # HGS 最小化目标
        "time_sec": float(total_time_sec),
        "total_time_sec": float(total_time_sec),
        "num_train_slices": len(train_records),
        "num_val_slices": len(val_records),
        "num_test_slices": len(test_records),
        "best_ckpt_path": best_ckpt_path,
        "status": "OK",
        "error_message": "",
    }

    # 默认 summary.json
    default_summary_path = os.path.join(save_dir, "summary.json")
    save_json(summary, default_summary_path)

    # 若搜索器传入 --output_json，则额外写一份到指定路径
    if args.output_json.strip():
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            ensure_dir(output_dir)
        save_json(summary, args.output_json)

    summary_csv = os.path.join(args.save_root, "week2_results_summary.csv")
    row_df = pd.DataFrame([summary])
    if os.path.exists(summary_csv):
        old_df = pd.read_csv(summary_csv)
        new_df = pd.concat([old_df, row_df], axis=0, ignore_index=True)
        new_df.to_csv(summary_csv, index=False)
    else:
        row_df.to_csv(summary_csv, index=False)

    print("\n========== FINAL SUMMARY ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
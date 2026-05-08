# -*- coding: utf-8 -*-
"""
根据旧备份 compare_raw.csv 重建被删除的 JSON 文件。

适用情况：
旧脚本删除了 trial_json/*.json 和 train_runs/*/summary.json，
但 compare_raw.csv 在 _backup_before_clean_low_qrghgs_xxx 目录中有备份。

注意：
这不是恢复原始 JSON 的所有隐藏字段，而是根据 compare_raw.csv 重建主要实验结果字段。
用于后续 compare、汇总、画图基本够用。
"""

from pathlib import Path
import json
import pandas as pd
import math


# =========================
# 1. 路径配置
# =========================

ROOT = Path(r"C:\Users\13178\Desktop\B_Project_QRG_UNet\outputs\real_compare_7methods")

BACKUP_DIR = ROOT / "_backup_before_clean_low_qrghgs_20260507_103341"

BACKUP_RAW_CSV = BACKUP_DIR / "compare_raw.csv"
CURRENT_RAW_CSV = ROOT / "compare_raw.csv"

TRIAL_JSON_DIR = ROOT / "trial_json"
TRAIN_RUNS_DIR = ROOT / "train_runs"

# True：只预览，不真正写入
# 确认没问题后改成 False
DRY_RUN = False

# True：如果 JSON 已经存在，就不覆盖
SKIP_EXISTING = True


# =========================
# 2. 工具函数
# =========================

def clean_value(v):
    """把 pandas/numpy 的 NaN 转成 None，避免 json 保存出错。"""
    if pd.isna(v):
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
    return v


def row_to_dict(row):
    """把 compare_raw.csv 的一行转成 JSON 内容。"""
    d = {}

    for col in row.index:
        d[col] = clean_value(row[col])

    # 补充常用字段，兼容你后续脚本读取
    d["method"] = clean_value(row.get("method"))
    d["trial_idx"] = clean_value(row.get("trial_idx"))
    d["run_name"] = clean_value(row.get("run_name"))
    d["seed"] = clean_value(row.get("seed"))
    d["lr"] = clean_value(row.get("lr"))
    d["dice_weight"] = clean_value(row.get("dice_weight"))
    d["batch_size"] = clean_value(row.get("batch_size"))

    d["best_val_dice"] = clean_value(row.get("best_val_dice"))
    d["test_dice"] = clean_value(row.get("test_dice"))
    d["test_iou"] = clean_value(row.get("test_iou"))
    d["test_sens"] = clean_value(row.get("test_sens"))
    d["test_spec"] = clean_value(row.get("test_spec"))
    d["time_sec"] = clean_value(row.get("time_sec"))

    d["status"] = clean_value(row.get("status"))
    d["error_message"] = clean_value(row.get("error_message"))
    d["best_ckpt_path"] = clean_value(row.get("best_ckpt_path"))
    d["resumed"] = clean_value(row.get("resumed"))

    # 额外保存一个 hparams 字段，方便后面读取
    d["hparams"] = {
        "lr": clean_value(row.get("lr")),
        "dice_weight": clean_value(row.get("dice_weight")),
        "batch_size": clean_value(row.get("batch_size")),
    }

    # 额外保存一个 metrics 字段
    d["metrics"] = {
        "best_val_dice": clean_value(row.get("best_val_dice")),
        "test_dice": clean_value(row.get("test_dice")),
        "test_iou": clean_value(row.get("test_iou")),
        "test_sens": clean_value(row.get("test_sens")),
        "test_spec": clean_value(row.get("test_spec")),
        "train_dice": clean_value(row.get("train_dice")),
        "fitness": clean_value(row.get("fitness")),
    }

    return d


def write_json(path: Path, data: dict):
    if path.exists() and SKIP_EXISTING:
        return "skip"

    if not DRY_RUN:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return "write"


# =========================
# 3. 主流程
# =========================

def main():
    if not BACKUP_RAW_CSV.exists():
        raise FileNotFoundError(f"找不到备份 compare_raw.csv：{BACKUP_RAW_CSV}")

    backup_raw = pd.read_csv(BACKUP_RAW_CSV)

    if "run_name" not in backup_raw.columns:
        raise ValueError("备份 compare_raw.csv 中没有 run_name 列，无法重建 JSON。")

    if "method" not in backup_raw.columns:
        raise ValueError("备份 compare_raw.csv 中没有 method 列。")

    backup_raw["run_name"] = backup_raw["run_name"].astype(str).str.strip()
    backup_raw["method"] = backup_raw["method"].astype(str).str.strip()

    # 如果当前 raw 存在，就优先找出被删掉的 run_name
    if CURRENT_RAW_CSV.exists():
        current_raw = pd.read_csv(CURRENT_RAW_CSV)
        current_raw["run_name"] = current_raw["run_name"].astype(str).str.strip()

        current_run_names = set(current_raw["run_name"].dropna().tolist())
        backup_run_names = set(backup_raw["run_name"].dropna().tolist())

        deleted_run_names = backup_run_names - current_run_names

        restore_df = backup_raw[backup_raw["run_name"].isin(deleted_run_names)].copy()

        print(f"[备份 raw 总行数] {len(backup_raw)}")
        print(f"[当前 raw 总行数] {len(current_raw)}")
        print(f"[根据 raw 差异识别出的被删 run 数] {len(restore_df)}")
    else:
        restore_df = backup_raw.copy()
        print("[提示] 当前 compare_raw.csv 不存在，将按备份 raw 全量重建 JSON。")

    if restore_df.empty:
        print("没有识别到需要重建的 JSON。")
        return

    print("\n[将要重建的 method 统计]")
    print(restore_df["method"].value_counts().to_string())

    n_trial_json_write = 0
    n_summary_json_write = 0
    n_skip = 0

    for _, row in restore_df.iterrows():
        run_name = str(row["run_name"]).strip()
        if not run_name or run_name.lower() == "nan":
            continue

        data = row_to_dict(row)

        trial_json_path = TRIAL_JSON_DIR / f"{run_name}.json"
        summary_json_path = TRAIN_RUNS_DIR / run_name / "summary.json"

        r1 = write_json(trial_json_path, data)
        r2 = write_json(summary_json_path, data)

        if r1 == "write":
            n_trial_json_write += 1
        else:
            n_skip += 1

        if r2 == "write":
            n_summary_json_write += 1
        else:
            n_skip += 1

        print(f"[恢复] {run_name}")
        print(f"  -> {trial_json_path}")
        print(f"  -> {summary_json_path}")

    print("\n==============================")
    print("恢复预览/执行完成")
    print(f"DRY_RUN = {DRY_RUN}")
    print(f"trial_json 写入数量：{n_trial_json_write}")
    print(f"summary.json 写入数量：{n_summary_json_write}")
    print(f"跳过数量：{n_skip}")
    print("==============================")

    if DRY_RUN:
        print("\n当前只是预览，没有真正写入。确认无误后，把 DRY_RUN = False 再运行。")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
精确删除指定的 3 个 QRGHGS 方法：
1. qrghgs_struct_v3_elite_exploit
2. qrghgs_struct_v3_elite_region
3. qrghgs_hybrid_v1

同步处理：
- compare_summary.csv 删除对应 method 行
- compare_raw.csv 删除对应 method 行
- 根据 compare_raw.csv 中的 run_name 精确移动/删除对应 JSON
  1）trial_json/{run_name}.json
  2）train_runs/{run_name}/summary.json

注意：
默认 DRY_RUN = True，只预览，不真正删除。
确认输出正确后，再改成 DRY_RUN = False。
"""

from pathlib import Path
from datetime import datetime
import shutil
import pandas as pd


# =========================================================
# 1. 路径配置
# =========================================================

ROOT = Path(r"C:\Users\13178\Desktop\B_Project_QRG_UNet\outputs\paper_main_structv3_real_benchmark")

SUMMARY_CSV = ROOT / "compare_summary.csv"
RAW_CSV = ROOT / "compare_raw.csv"

# 第一次建议 True，只预览
DRY_RUN = False

# JSON 建议移动到备份目录，不建议直接永久删除
MOVE_JSON_TO_BACKUP = False

# 是否同步处理 JSON
SYNC_JSON = True


# =========================================================
# 2. 指定要删除的 method，必须精确匹配
# =========================================================

DELETE_METHODS = {
    "qrghgs_struct_v3_elite_exploit",
    "qrghgs_struct_v3_elite_region",
    "qrghgs_hybrid_v1",
    "qrghgs_hybrid_topk_light",
}


# =========================================================
# 3. 备份与日志
# =========================================================

TIME_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_DIR = ROOT / f"_backup_before_delete_three_methods_{TIME_TAG}"
JSON_BACKUP_DIR = BACKUP_DIR / "moved_json"
LOG_PATH = ROOT / f"delete_three_methods_log_{TIME_TAG}.txt"


def log(msg=""):
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


def backup_file(path: Path):
    if not path.exists():
        log(f"[跳过备份] 文件不存在：{path}")
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dst = BACKUP_DIR / path.name

    if not DRY_RUN:
        shutil.copy2(path, dst)

    log(f"[备份] {path} -> {dst}")


def load_csvs():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"找不到 compare_summary.csv：{SUMMARY_CSV}")

    if not RAW_CSV.exists():
        raise FileNotFoundError(f"找不到 compare_raw.csv：{RAW_CSV}")

    summary_df = pd.read_csv(SUMMARY_CSV)
    raw_df = pd.read_csv(RAW_CSV)

    if "method" not in summary_df.columns:
        raise ValueError("compare_summary.csv 中找不到 method 列。")

    if "method" not in raw_df.columns:
        raise ValueError("compare_raw.csv 中找不到 method 列。")

    if "run_name" not in raw_df.columns:
        raise ValueError("compare_raw.csv 中找不到 run_name 列，无法同步定位 JSON。")

    summary_df["method"] = summary_df["method"].astype(str).str.strip()
    raw_df["method"] = raw_df["method"].astype(str).str.strip()
    raw_df["run_name"] = raw_df["run_name"].astype(str).str.strip()

    return summary_df, raw_df


def filter_summary(summary_df):
    remove_mask = summary_df["method"].isin(DELETE_METHODS)

    removed_df = summary_df[remove_mask].copy()
    kept_df = summary_df[~remove_mask].copy()

    return kept_df, removed_df


def filter_raw(raw_df):
    remove_mask = raw_df["method"].isin(DELETE_METHODS)

    removed_df = raw_df[remove_mask].copy()
    kept_df = raw_df[~remove_mask].copy()

    return kept_df, removed_df


def collect_json_paths_from_removed_raw(removed_raw_df):
    """
    根据 raw 中被删除行的 run_name 精确定位 JSON。
    不做模糊匹配。
    不扫描 JSON 内容。
    """
    json_paths = set()

    if removed_raw_df.empty:
        return []

    run_names = sorted(removed_raw_df["run_name"].dropna().unique())

    for run_name in run_names:
        run_name = str(run_name).strip()

        if not run_name or run_name.lower() == "nan":
            continue

        json_paths.add(ROOT / "trial_json" / f"{run_name}.json")
        json_paths.add(ROOT / "train_runs" / run_name / "summary.json")

    existing_paths = sorted([p for p in json_paths if p.exists()])
    missing_paths = sorted([p for p in json_paths if not p.exists()])

    log("\n[JSON 精确定位结果]")
    log(f"找到存在的 JSON 数量：{len(existing_paths)}")
    log(f"未找到的 JSON 数量：{len(missing_paths)}")

    if existing_paths:
        log("\n[将处理的 JSON 文件]")
        for p in existing_paths:
            log(str(p))

    if missing_paths:
        log("\n[未找到的 JSON 文件，仅提示，不影响 CSV 清理]")
        for p in missing_paths[:50]:
            log(str(p))
        if len(missing_paths) > 50:
            log(f"... 还有 {len(missing_paths) - 50} 个未显示")

    return existing_paths


def handle_json_files(json_paths):
    if not SYNC_JSON:
        log("\n[JSON] SYNC_JSON = False，不处理 JSON。")
        return

    if not json_paths:
        log("\n[JSON] 没有需要处理的 JSON。")
        return

    if DRY_RUN:
        log("\n[DRY_RUN] 预览模式：不会移动或删除 JSON。")
        return

    if MOVE_JSON_TO_BACKUP:
        JSON_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    for p in json_paths:
        try:
            if MOVE_JSON_TO_BACKUP:
                rel = p.relative_to(ROOT)
                dst = JSON_BACKUP_DIR / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dst))
                log(f"[JSON移动到备份] {p} -> {dst}")
            else:
                p.unlink()
                log(f"[JSON删除] {p}")
        except Exception as e:
            log(f"[JSON处理失败] {p}，原因：{e}")


def main():
    log("=" * 80)
    log("开始精确删除指定的 3 个 QRGHGS 方法")
    log(f"ROOT = {ROOT}")
    log(f"DRY_RUN = {DRY_RUN}")
    log(f"MOVE_JSON_TO_BACKUP = {MOVE_JSON_TO_BACKUP}")
    log("=" * 80)

    log("\n[指定删除 method]")
    for m in sorted(DELETE_METHODS):
        log(f"- {m}")

    summary_df, raw_df = load_csvs()

    backup_file(SUMMARY_CSV)
    backup_file(RAW_CSV)

    cleaned_summary, removed_summary = filter_summary(summary_df)
    cleaned_raw, removed_raw = filter_raw(raw_df)

    log("\n[compare_summary.csv 删除预览]")
    log(f"原始行数：{len(summary_df)}")
    log(f"删除行数：{len(removed_summary)}")
    log(f"保留行数：{len(cleaned_summary)}")

    if removed_summary.empty:
        log("summary 中没有找到这 3 个 method。")
    else:
        show_cols = [c for c in ["method", "test_dice_mean", "test_dice_std", "test_iou_mean"] if c in removed_summary.columns]
        log("\n[summary 将删除的行]")
        log(removed_summary[show_cols].to_string(index=False))

    log("\n[compare_raw.csv 删除预览]")
    log(f"原始行数：{len(raw_df)}")
    log(f"删除行数：{len(removed_raw)}")
    log(f"保留行数：{len(cleaned_raw)}")

    if removed_raw.empty:
        log("raw 中没有找到这 3 个 method。")
    else:
        log("\n[raw 中各 method 删除数量]")
        log(removed_raw["method"].value_counts().to_string())

    # 安全检查：被删的 method 必须全部属于 DELETE_METHODS
    removed_summary_methods = set(removed_summary["method"].unique())
    removed_raw_methods = set(removed_raw["method"].unique())

    unsafe_summary = removed_summary_methods - DELETE_METHODS
    unsafe_raw = removed_raw_methods - DELETE_METHODS

    if unsafe_summary:
        raise RuntimeError(f"安全检查失败：summary 误删了非目标 method：{unsafe_summary}")

    if unsafe_raw:
        raise RuntimeError(f"安全检查失败：raw 误删了非目标 method：{unsafe_raw}")

    json_paths = collect_json_paths_from_removed_raw(removed_raw)

    if DRY_RUN:
        log("\n[DRY_RUN] 当前只是预览，没有真正写回 CSV，也没有移动/删除 JSON。")
        log("确认上面的删除列表正确后，把 DRY_RUN = False 再运行。")
    else:
        cleaned_summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
        cleaned_raw.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")

        log(f"\n[已写回] {SUMMARY_CSV}")
        log(f"[已写回] {RAW_CSV}")

        handle_json_files(json_paths)

    log("\n[清理后保留的 summary method]")
    log(cleaned_summary["method"].to_string(index=False))

    log("\n" + "=" * 80)
    log("脚本结束")
    log(f"日志文件：{LOG_PATH}")
    log(f"备份目录：{BACKUP_DIR}")
    log("=" * 80)


if __name__ == "__main__":
    main()

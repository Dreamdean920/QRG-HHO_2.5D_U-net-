# -*- coding: utf-8 -*-
"""
严格安全版：
清除 compare_summary.csv 中 test_dice_mean 低于 qrghgs_baseline 的 qrghgs 衍生算法，
并同步清理 compare_raw.csv 和对应 json。

修复点：
1. 不再使用模糊匹配。
2. 只按 method 精确删除。
3. raw 只删除 method 完全等于待删 method 的行。
4. json 只根据 raw 里的 run_name 精确定位。
5. 默认不真正删除，先 DRY_RUN 预览。
6. 默认不直接删 json，而是移动到备份目录。
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

# 第一次必须先 True，只预览
DRY_RUN = False

# 如果你已经被旧脚本误删过，先从旧备份恢复 CSV
# 旧日志显示备份目录是：
# _backup_before_clean_low_qrghgs_20260507_103341
RESTORE_FROM_BACKUP_FIRST = True

# 如果你想手动指定备份目录，就填这里；不填则自动找最新的 _backup_before_clean_low_qrghgs_*
BACKUP_DIR_TO_RESTORE = None
# 示例：
# BACKUP_DIR_TO_RESTORE = ROOT / "_backup_before_clean_low_qrghgs_20260507_103341"

# JSON 建议先不要删，确认 CSV 清理正确后再打开
SYNC_JSON = True

# JSON 不建议永久删除，统一移动到本次备份目录
MOVE_JSON_TO_BACKUP = True


# =========================================================
# 2. 本次备份和日志
# =========================================================

TIME_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
THIS_BACKUP_DIR = ROOT / f"_backup_safe_clean_{TIME_TAG}"
JSON_BACKUP_DIR = THIS_BACKUP_DIR / "moved_json"
LOG_PATH = ROOT / f"safe_clean_low_qrghgs_log_{TIME_TAG}.txt"


def log(msg=""):
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


def backup_current_file(path: Path):
    if not path.exists():
        return
    THIS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dst = THIS_BACKUP_DIR / path.name
    if not DRY_RUN:
        shutil.copy2(path, dst)
    log(f"[本次备份] {path} -> {dst}")


def find_latest_old_backup():
    backups = sorted(
        ROOT.glob("_backup_before_clean_low_qrghgs_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not backups:
        return None
    return backups[0]


def restore_csv_from_backup():
    """
    用旧脚本自动生成的备份恢复 compare_summary.csv 和 compare_raw.csv。
    只恢复 CSV，不恢复 JSON。
    """
    if not RESTORE_FROM_BACKUP_FIRST:
        return

    backup_dir = BACKUP_DIR_TO_RESTORE or find_latest_old_backup()

    if backup_dir is None:
        raise FileNotFoundError(
            "没有找到旧备份目录 _backup_before_clean_low_qrghgs_*，无法自动恢复 CSV。"
        )

    backup_summary = backup_dir / "compare_summary.csv"
    backup_raw = backup_dir / "compare_raw.csv"

    if not backup_summary.exists():
        raise FileNotFoundError(f"备份中找不到 compare_summary.csv：{backup_summary}")
    if not backup_raw.exists():
        raise FileNotFoundError(f"备份中找不到 compare_raw.csv：{backup_raw}")

    log("=" * 80)
    log("[恢复阶段]")
    log(f"使用旧备份目录：{backup_dir}")
    log(f"DRY_RUN = {DRY_RUN}")

    if DRY_RUN:
        log("[DRY_RUN] 预览模式：不会真正恢复 CSV。")
    else:
        shutil.copy2(backup_summary, SUMMARY_CSV)
        shutil.copy2(backup_raw, RAW_CSV)
        log(f"[已恢复] {backup_summary} -> {SUMMARY_CSV}")
        log(f"[已恢复] {backup_raw} -> {RAW_CSV}")

    log("=" * 80)


def load_csvs():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"找不到 {SUMMARY_CSV}")
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"找不到 {RAW_CSV}")

    summary_df = pd.read_csv(SUMMARY_CSV)
    raw_df = pd.read_csv(RAW_CSV)

    required_summary_cols = {"method", "test_dice_mean"}
    required_raw_cols = {"method", "run_name"}

    missing_summary = required_summary_cols - set(summary_df.columns)
    missing_raw = required_raw_cols - set(raw_df.columns)

    if missing_summary:
        raise ValueError(f"compare_summary.csv 缺少列：{missing_summary}")
    if missing_raw:
        raise ValueError(f"compare_raw.csv 缺少列：{missing_raw}")

    summary_df["method"] = summary_df["method"].astype(str).str.strip()
    raw_df["method"] = raw_df["method"].astype(str).str.strip()
    raw_df["run_name"] = raw_df["run_name"].astype(str).str.strip()

    summary_df["test_dice_mean"] = pd.to_numeric(
        summary_df["test_dice_mean"],
        errors="coerce"
    )

    return summary_df, raw_df


def get_baseline_value(summary_df):
    baseline_rows = summary_df[summary_df["method"] == "qrghgs_baseline"].copy()

    if baseline_rows.empty:
        raise ValueError(
            "没有找到 method 完全等于 qrghgs_baseline 的行。"
            "为了避免误删，本脚本不使用模糊 baseline 匹配。"
        )

    if baseline_rows["test_dice_mean"].isna().any():
        raise ValueError("qrghgs_baseline 的 test_dice_mean 不是有效数字。")

    baseline_value = baseline_rows["test_dice_mean"].max()

    log("[baseline]")
    log(baseline_rows[["method", "test_dice_mean"]].to_string(index=False))
    log(f"baseline test_dice_mean = {baseline_value:.9f}")

    return baseline_value


def detect_bad_methods(summary_df, baseline_value):
    """
    只删除：
    1. method 以 qrghgs_ 开头
    2. method 不是 qrghgs_baseline
    3. test_dice_mean < baseline
    """
    is_qrghgs = summary_df["method"].str.startswith("qrghgs_")
    is_not_baseline = summary_df["method"] != "qrghgs_baseline"
    is_lower = summary_df["test_dice_mean"] < baseline_value

    bad_df = summary_df[is_qrghgs & is_not_baseline & is_lower].copy()

    bad_methods = set(bad_df["method"].tolist())

    if "qrghgs_baseline" in bad_methods:
        raise RuntimeError("安全检查失败：qrghgs_baseline 被加入了删除列表。")

    log("\n[本次精确识别出的待删除 method]")
    if bad_df.empty:
        log("没有需要删除的 qrghgs 衍生算法。")
    else:
        log(bad_df[["method", "test_dice_mean"]].to_string(index=False))

    return bad_df, bad_methods


def safe_filter_summary(summary_df, bad_methods, baseline_value):
    removed = summary_df[summary_df["method"].isin(bad_methods)].copy()
    kept = summary_df[~summary_df["method"].isin(bad_methods)].copy()

    # 安全检查：如果被删除的里面有 >= baseline 的，立即停止
    unsafe = removed[removed["test_dice_mean"] >= baseline_value]
    if not unsafe.empty:
        raise RuntimeError(
            "安全检查失败：有 test_dice_mean >= baseline 的方法被准备删除：\n"
            + unsafe[["method", "test_dice_mean"]].to_string(index=False)
        )

    return kept, removed


def safe_filter_raw(raw_df, bad_methods):
    removed = raw_df[raw_df["method"].isin(bad_methods)].copy()
    kept = raw_df[~raw_df["method"].isin(bad_methods)].copy()

    # 安全检查：raw 被删除的 method 必须全部在 bad_methods 里
    extra_methods = set(removed["method"].unique()) - bad_methods
    if extra_methods:
        raise RuntimeError(f"安全检查失败：raw 误删了非目标 method：{extra_methods}")

    return kept, removed


def collect_exact_json_paths(removed_raw_df):
    """
    只根据 run_name 精确定位 JSON：
    1. trial_json/{run_name}.json
    2. train_runs/{run_name}/summary.json

    不再扫描 JSON 内容。
    不再用字符串包含判断。
    """
    json_paths = set()

    if removed_raw_df.empty:
        return []

    for run_name in sorted(removed_raw_df["run_name"].dropna().unique()):
        run_name = str(run_name).strip()
        if not run_name:
            continue

        json_paths.add(ROOT / "trial_json" / f"{run_name}.json")
        json_paths.add(ROOT / "train_runs" / run_name / "summary.json")

    existing = sorted([p for p in json_paths if p.exists()])
    missing = sorted([p for p in json_paths if not p.exists()])

    log("\n[JSON 精确匹配结果]")
    log(f"找到存在的 JSON 数量：{len(existing)}")
    log(f"不存在的 JSON 数量：{len(missing)}")

    if existing:
        log("\n[将处理的 JSON]")
        for p in existing:
            log(str(p))

    if missing:
        log("\n[未找到的 JSON，仅提示，不报错]")
        for p in missing[:30]:
            log(str(p))
        if len(missing) > 30:
            log(f"... 还有 {len(missing) - 30} 个未显示")

    return existing


def move_or_delete_json(json_paths):
    if not SYNC_JSON:
        log("\n[JSON] SYNC_JSON = False，不处理 JSON。")
        return

    if not json_paths:
        log("\n[JSON] 没有需要处理的 JSON。")
        return

    if DRY_RUN:
        log("\n[DRY_RUN] 预览模式：不会移动/删除 JSON。")
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
    log("严格安全版清理脚本启动")
    log(f"ROOT = {ROOT}")
    log(f"DRY_RUN = {DRY_RUN}")
    log(f"RESTORE_FROM_BACKUP_FIRST = {RESTORE_FROM_BACKUP_FIRST}")
    log(f"SYNC_JSON = {SYNC_JSON}")
    log("=" * 80)

    # 1. 先从旧备份恢复 CSV
    restore_csv_from_backup()

    # 2. 读取 CSV
    summary_df, raw_df = load_csvs()

    # 3. 当前文件备份
    backup_current_file(SUMMARY_CSV)
    backup_current_file(RAW_CSV)

    # 4. 精确找 baseline
    baseline_value = get_baseline_value(summary_df)

    # 5. 精确找待删除方法
    bad_df, bad_methods = detect_bad_methods(summary_df, baseline_value)

    if not bad_methods:
        log("\n没有需要删除的算法，程序结束。")
        return

    # 6. 精确过滤 summary 和 raw
    cleaned_summary, removed_summary = safe_filter_summary(
        summary_df,
        bad_methods,
        baseline_value
    )

    cleaned_raw, removed_raw = safe_filter_raw(
        raw_df,
        bad_methods
    )

    log("\n[compare_summary.csv 清理统计]")
    log(f"原始行数：{len(summary_df)}")
    log(f"删除行数：{len(removed_summary)}")
    log(f"保留行数：{len(cleaned_summary)}")

    log("\n[compare_raw.csv 清理统计]")
    log(f"原始行数：{len(raw_df)}")
    log(f"删除行数：{len(removed_raw)}")
    log(f"保留行数：{len(cleaned_raw)}")

    log("\n[raw 中各待删 method 的行数]")
    log(removed_raw["method"].value_counts().to_string())

    # 7. JSON 精确定位
    json_paths = collect_exact_json_paths(removed_raw)

    # 8. 真正写回
    if DRY_RUN:
        log("\n[DRY_RUN] 预览模式：CSV 不会写回，JSON 不会移动/删除。")
        log("确认上面的删除列表完全正确后，把 DRY_RUN = False 再运行。")
    else:
        cleaned_summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
        cleaned_raw.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
        log(f"\n[已写回] {SUMMARY_CSV}")
        log(f"[已写回] {RAW_CSV}")

        move_or_delete_json(json_paths)

    log("\n[保留的 summary method]")
    log(cleaned_summary["method"].to_string(index=False))

    log("\n" + "=" * 80)
    log("脚本结束")
    log(f"日志文件：{LOG_PATH}")
    log(f"本次备份目录：{THIS_BACKUP_DIR}")
    log("=" * 80)


if __name__ == "__main__":
    main()

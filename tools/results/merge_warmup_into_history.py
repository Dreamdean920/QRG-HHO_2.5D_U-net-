import pandas as pd
from pathlib import Path

# =========================
# 路径配置
# =========================
BASE_HISTORY = Path("outputs/week4_merged/all_methods_merged.csv")
WARMUP_CSV = Path("outputs/batch_warmup/warmup_batch_results.csv")
OUT_DIR = Path("outputs/history_augmented")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "all_methods_merged_plus_warmup.csv"


def main():
    if not BASE_HISTORY.exists():
        raise FileNotFoundError(f"找不到历史文件: {BASE_HISTORY}")
    if not WARMUP_CSV.exists():
        raise FileNotFoundError(f"找不到 warmup 文件: {WARMUP_CSV}")

    base = pd.read_csv(BASE_HISTORY)
    warm = pd.read_csv(WARMUP_CSV)

    print("原始历史条数:", len(base))
    print("warmup 条数:", len(warm))

    # =========================
    # 1. 统一列结构
    # =========================
    # base 中通常有这些列：
    # method, trial_idx, run_name, seed, lr, dice_weight, batch_size,
    # best_val_dice, test_dice, test_iou, test_sens, test_spec, time_sec,
    # status, error_message, best_ckpt_path, source

    # warmup 只有少数列，需要补齐
    for col in base.columns:
        if col not in warm.columns:
            warm[col] = None

    # 指定 warmup 的 method/source
    warm["method"] = "warmup"
    warm["source"] = "batch_warmup"

    # 如果没有 trial_idx，就按顺序生成
    if "trial_idx" in warm.columns:
        warm["trial_idx"] = range(len(warm))
    else:
        warm["trial_idx"] = range(len(warm))

    # 如果没有 seed，默认 42
    if "seed" in warm.columns:
        warm["seed"] = warm["seed"].fillna(42)
    else:
        warm["seed"] = 42

    # 列顺序对齐
    warm = warm[base.columns]

    # =========================
    # 2. 只保留成功结果
    # =========================
    base = base[base["status"] == "OK"].copy()
    warm = warm[warm["status"] == "OK"].copy()

    # =========================
    # 3. 合并
    # =========================
    merged = pd.concat([base, warm], ignore_index=True)

    # 去重：按 run_name 保留最后一次
    if "run_name" in merged.columns:
        merged = (
            merged.sort_values("run_name")
                  .drop_duplicates(subset=["run_name"], keep="last")
                  .reset_index(drop=True)
        )

    # =========================
    # 4. 保存
    # =========================
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("\n✅ 已保存增强后的历史文件：", OUT_CSV)
    print("合并后条数:", len(merged))

    # =========================
    # 5. 打印 batch 覆盖情况
    # =========================
    print("\n===== batch_size 覆盖情况 =====")
    print(merged["batch_size"].value_counts().sort_index())

    if "best_val_dice" in merged.columns:
        print("\n===== 各 batch 的 best_val_dice 统计 =====")
        print(
            merged.groupby("batch_size")["best_val_dice"]
                  .agg(["count", "mean", "max", "std"])
        )


if __name__ == "__main__":
    main()
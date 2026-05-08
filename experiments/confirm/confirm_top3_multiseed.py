import sys
import json
import subprocess
from pathlib import Path

import pandas as pd


# =========================================================
# 0. 路径配置
# =========================================================
PROJECT_ROOT = Path(".")
BASE_OUT_ROOT = PROJECT_ROOT / "outputs" / "real_compare_7methods"

SOURCE_CSV = BASE_OUT_ROOT / "compare_raw.csv"

CONFIRM_ROOT = BASE_OUT_ROOT / "top3_multiseed_confirm"
CONFIRM_TRAIN_ROOT = CONFIRM_ROOT / "train_runs"
CONFIRM_JSON_ROOT = CONFIRM_ROOT / "trial_json"

CONFIRM_ROOT.mkdir(parents=True, exist_ok=True)
CONFIRM_TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
CONFIRM_JSON_ROOT.mkdir(parents=True, exist_ok=True)

CONFIRM_RAW_CSV = CONFIRM_ROOT / "confirm_raw.csv"
CANDIDATE_SUMMARY_CSV = CONFIRM_ROOT / "confirm_candidate_summary.csv"
METHOD_SUMMARY_CSV = CONFIRM_ROOT / "confirm_method_summary.csv"

PYTHON_EXE = sys.executable

# =========================================================
# 1. Confirm 配置
# =========================================================
CONFIRM_METHODS = [
    "qrghgs_baseline",
    "qrghgs_hybrid",
    "qrghgs_struct_v3",
    "ga",
    "hgs",
    "random",
    "tpe",
    "pso",
]

TOP_K = 3

CONFIRM_SEEDS = [42, 52, 62, 72, 82]

# 如果时间太贵，可以先改成 20；
# 正式论文建议 50。
CONFIRM_EPOCHS = 50

COMMON_ARGS = [
    "--input_mode", "2.5d",
    "--no_post",
    "--save_root", str(CONFIRM_TRAIN_ROOT),
    "--epochs", str(CONFIRM_EPOCHS),
]


# =========================================================
# 2. 工具函数
# =========================================================
def load_json_if_ok(json_path: Path):
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if data.get("status", "OK") == "OK":
            return data
    except Exception:
        return None
    return None


def make_confirm_run_name(source_method, rank_idx, seed):
    return f"confirm_{source_method}_rank{rank_idx}_seed{seed}"


def read_existing_confirm_rows():
    if CONFIRM_RAW_CSV.exists():
        try:
            return pd.read_csv(CONFIRM_RAW_CSV).to_dict("records")
        except Exception:
            return []
    return []


def save_confirm_results(rows):
    df = pd.DataFrame(rows)
    df.to_csv(CONFIRM_RAW_CSV, index=False, encoding="utf-8-sig")

    if len(df) == 0:
        return

    ok_df = df[df["status"] == "OK"].copy()

    if len(ok_df) == 0:
        return

    # 每个候选参数的多 seed 结果
    cand_summary = []
    group_cols = ["source_method", "candidate_rank", "source_run_name"]

    for keys, g in ok_df.groupby(group_cols):
        source_method, candidate_rank, source_run_name = keys

        cand_summary.append({
            "source_method": source_method,
            "candidate_rank": candidate_rank,
            "source_run_name": source_run_name,
            "n_seed": len(g),
            "lr": g["lr"].iloc[0],
            "dice_weight": g["dice_weight"].iloc[0],
            "batch_size": g["batch_size"].iloc[0],
            "source_best_val_dice": g["source_best_val_dice"].iloc[0],
            "source_test_dice": g["source_test_dice"].iloc[0],
            "confirm_best_val_mean": g["best_val_dice"].mean(),
            "confirm_best_val_std": g["best_val_dice"].std(ddof=0),
            "confirm_test_dice_mean": g["test_dice"].mean(),
            "confirm_test_dice_std": g["test_dice"].std(ddof=0),
            "confirm_test_dice_max": g["test_dice"].max(),
            "confirm_test_iou_mean": g["test_iou"].mean(),
            "confirm_test_sens_mean": g["test_sens"].mean(),
            "confirm_test_spec_mean": g["test_spec"].mean(),
            "time_sec_mean": g["time_sec"].mean(),
            "time_sec_sum": g["time_sec"].sum(),
        })

    cand_df = pd.DataFrame(cand_summary)
    cand_df = cand_df.sort_values(
        ["source_method", "confirm_test_dice_mean"],
        ascending=[True, False],
    )
    cand_df.to_csv(CANDIDATE_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    # 每个方法选择 confirm mean 最好的 top1 候选
    method_summary = []
    for method, g in cand_df.groupby("source_method"):
        best = g.sort_values("confirm_test_dice_mean", ascending=False).iloc[0]

        method_summary.append({
            "source_method": method,
            "selected_candidate_rank": best["candidate_rank"],
            "selected_source_run_name": best["source_run_name"],
            "n_seed": best["n_seed"],
            "lr": best["lr"],
            "dice_weight": best["dice_weight"],
            "batch_size": best["batch_size"],
            "source_best_val_dice": best["source_best_val_dice"],
            "source_test_dice": best["source_test_dice"],
            "confirm_test_dice_mean": best["confirm_test_dice_mean"],
            "confirm_test_dice_std": best["confirm_test_dice_std"],
            "confirm_test_dice_max": best["confirm_test_dice_max"],
            "confirm_best_val_mean": best["confirm_best_val_mean"],
            "confirm_best_val_std": best["confirm_best_val_std"],
            "confirm_test_iou_mean": best["confirm_test_iou_mean"],
            "confirm_test_sens_mean": best["confirm_test_sens_mean"],
            "confirm_test_spec_mean": best["confirm_test_spec_mean"],
            "time_sec_mean": best["time_sec_mean"],
            "time_sec_sum": best["time_sec_sum"],
        })

    method_df = pd.DataFrame(method_summary)
    method_df = method_df.sort_values("confirm_test_dice_mean", ascending=False)
    method_df.to_csv(METHOD_SUMMARY_CSV, index=False, encoding="utf-8-sig")


def run_confirm_one(candidate, rank_idx, seed):
    source_method = candidate["method"]
    source_run_name = candidate["run_name"]

    lr = float(candidate["lr"])
    dice_weight = float(candidate["dice_weight"])
    batch_size = int(candidate["batch_size"])

    confirm_run_name = make_confirm_run_name(source_method, rank_idx, seed)
    output_json = CONFIRM_JSON_ROOT / f"{confirm_run_name}.json"

    old = load_json_if_ok(output_json)
    if old is not None:
        print(f"[SKIP] {confirm_run_name}")
        return {
            "source_method": source_method,
            "candidate_rank": rank_idx,
            "source_run_name": source_run_name,
            "confirm_run_name": confirm_run_name,
            "seed": seed,
            "lr": lr,
            "dice_weight": dice_weight,
            "batch_size": batch_size,
            "source_best_val_dice": candidate.get("best_val_dice"),
            "source_test_dice": candidate.get("test_dice"),
            "best_val_dice": old.get("best_val_dice"),
            "test_dice": old.get("test_dice"),
            "test_iou": old.get("test_iou"),
            "test_sens": old.get("test_sens"),
            "test_spec": old.get("test_spec"),
            "time_sec": old.get("time_sec"),
            "status": old.get("status", "OK"),
            "error_message": old.get("error_message", ""),
            "best_ckpt_path": old.get("best_ckpt_path", ""),
            "resumed": True,
        }

    cmd = [
        PYTHON_EXE,
        "scripts/train_week3_unet.py",
        *COMMON_ARGS,
        "--seed", str(seed),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--dice_weight", str(dice_weight),
        "--run_name", confirm_run_name,
        "--output_json", str(output_json),
    ]

    print("\n" + "=" * 80)
    print(f"[CONFIRM] {confirm_run_name}")
    print(" ".join(cmd))
    print("=" * 80)

    proc = subprocess.run(cmd)

    if proc.returncode != 0 or not output_json.exists():
        return {
            "source_method": source_method,
            "candidate_rank": rank_idx,
            "source_run_name": source_run_name,
            "confirm_run_name": confirm_run_name,
            "seed": seed,
            "lr": lr,
            "dice_weight": dice_weight,
            "batch_size": batch_size,
            "source_best_val_dice": candidate.get("best_val_dice"),
            "source_test_dice": candidate.get("test_dice"),
            "best_val_dice": None,
            "test_dice": None,
            "test_iou": None,
            "test_sens": None,
            "test_spec": None,
            "time_sec": None,
            "status": "ERROR",
            "error_message": "subprocess_failed_or_missing_json",
            "best_ckpt_path": "",
            "resumed": False,
        }

    data = json.loads(output_json.read_text(encoding="utf-8"))

    return {
        "source_method": source_method,
        "candidate_rank": rank_idx,
        "source_run_name": source_run_name,
        "confirm_run_name": confirm_run_name,
        "seed": seed,
        "lr": lr,
        "dice_weight": dice_weight,
        "batch_size": batch_size,
        "source_best_val_dice": candidate.get("best_val_dice"),
        "source_test_dice": candidate.get("test_dice"),
        "best_val_dice": data.get("best_val_dice"),
        "test_dice": data.get("test_dice"),
        "test_iou": data.get("test_iou"),
        "test_sens": data.get("test_sens"),
        "test_spec": data.get("test_spec"),
        "time_sec": data.get("time_sec"),
        "status": data.get("status", "OK"),
        "error_message": data.get("error_message", ""),
        "best_ckpt_path": data.get("best_ckpt_path", ""),
        "resumed": False,
    }


# =========================================================
# 3. 主流程
# =========================================================
def select_topk_candidates():
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"找不到源文件: {SOURCE_CSV}")

    df = pd.read_csv(SOURCE_CSV)

    df = df[df["status"] == "OK"].copy()
    df = (
        df.sort_values(["method", "trial_idx", "resumed"])
        .drop_duplicates(subset=["method", "run_name"], keep="last")
        .reset_index(drop=True)
    )

    for col in ["best_val_dice", "test_dice", "lr", "dice_weight", "batch_size"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["best_val_dice", "lr", "dice_weight", "batch_size"])

    selected = []

    for method in CONFIRM_METHODS:
        sub = df[df["method"] == method].copy()

        if len(sub) == 0:
            print(f"[WARN] 方法 {method} 没有记录，跳过。")
            continue

        sub = sub.sort_values("best_val_dice", ascending=False).head(TOP_K)

        for rank_idx, (_, row) in enumerate(sub.iterrows(), start=1):
            selected.append(row.to_dict())

    return selected


if __name__ == "__main__":
    rows = read_existing_confirm_rows()

    candidates = select_topk_candidates()

    print("\n===== 将要 confirm 的候选参数 =====")
    for i, c in enumerate(candidates, start=1):
        print(
            i,
            c["method"],
            c["run_name"],
            "val=",
            c["best_val_dice"],
            "test=",
            c.get("test_dice"),
            "lr=",
            c["lr"],
            "dice=",
            c["dice_weight"],
            "batch=",
            c["batch_size"],
        )

    for cand in candidates:
        # candidate_rank 已经在候选列表中按每个方法单独编号
        # 这里重新计算 rank：同一 source_method 内按 best_val 排名
        source_method = cand["method"]
        same_method = [x for x in candidates if x["method"] == source_method]
        same_method_sorted = sorted(
            same_method,
            key=lambda z: float(z["best_val_dice"]),
            reverse=True,
        )
        rank_idx = same_method_sorted.index(cand) + 1

        for seed in CONFIRM_SEEDS:
            row = run_confirm_one(cand, rank_idx, seed)
            rows.append(row)
            save_confirm_results(rows)

    print("\n✅ Top3 多 seed confirm 完成。")
    print("原始 confirm 结果:", CONFIRM_RAW_CSV)
    print("候选参数汇总:", CANDIDATE_SUMMARY_CSV)
    print("方法最终汇总:", METHOD_SUMMARY_CSV)
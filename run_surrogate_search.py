import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from surrogate_model import RFEnsembleSurrogate, load_history_from_csv


# =========================
# 配置
# =========================
HISTORY_CSV = "outputs/week4_merged/all_methods_merged.csv"
OUTPUT_DIR = Path("outputs/surrogate_search")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_EXE = sys.executable

SPACE = {
    "lr": (0.0012, 0.0019),
    "dice_weight": (0.62, 0.80),
    "batch_size": [4],   # 当前建议先固定 4
}

N_CANDIDATES = 3000     # 代理模型内部候选池
N_REAL_EVALS = 3        # 每轮只真实训练 3 个点
UCB_BETA = 1.5          # 探索强度，越大越鼓励探索
SEED = 42


def sample_candidates(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        rows.append({
            "lr": float(rng.uniform(*SPACE["lr"])),
            "dice_weight": float(rng.uniform(*SPACE["dice_weight"])),
            "batch_size": int(rng.choice(SPACE["batch_size"])),
        })
    return pd.DataFrame(rows)


def build_run_name(idx: int) -> str:
    return f"surrogate_trial{idx:03d}"


def real_evaluate(row: pd.Series, run_idx: int) -> dict:
    run_name = build_run_name(run_idx)
    output_json = OUTPUT_DIR / f"{run_name}.json"

    cmd = [
        PYTHON_EXE,
        "scripts/train_week3_unet.py",
        "--input_mode", "2.5d",
        "--no_post",
        "--save_root", str(OUTPUT_DIR / "train_runs"),
        "--epochs", "20",
        "--seed", str(SEED),
        "--batch_size", str(int(row["batch_size"])),
        "--lr", str(float(row["lr"])),
        "--dice_weight", str(float(row["dice_weight"])),
        "--run_name", run_name,
        "--output_json", str(output_json),
    ]

    print("\n[REAL EVAL]", " ".join(cmd))
    proc = subprocess.run(cmd)

    if proc.returncode != 0 or not output_json.exists():
        return {
            "run_name": run_name,
            "lr": row["lr"],
            "dice_weight": row["dice_weight"],
            "batch_size": row["batch_size"],
            "status": "ERROR",
            "best_val_dice": None,
            "test_dice": None,
            "time_sec": None,
        }

    data = json.loads(output_json.read_text(encoding="utf-8"))
    return {
        "run_name": run_name,
        "lr": row["lr"],
        "dice_weight": row["dice_weight"],
        "batch_size": row["batch_size"],
        "status": data.get("status", "OK"),
        "best_val_dice": data.get("best_val_dice"),
        "test_dice": data.get("test_dice"),
        "time_sec": data.get("time_sec"),
    }


def main():
    history = load_history_from_csv(HISTORY_CSV)

    # 只保留 batch=4，和你当前公平口径一致
    history = history[history["batch_size"] == 4].copy().reset_index(drop=True)

    print("历史样本数:", len(history))
    print("历史最好 best_val_dice:", history["best_val_dice"].max())

    surrogate = RFEnsembleSurrogate(n_models=8, random_seed=SEED)
    surrogate.fit(history)

    candidates = sample_candidates(N_CANDIDATES, seed=SEED)
    Xcand = candidates[["lr", "dice_weight", "batch_size"]].values

    mean, std = surrogate.predict_mean_std(Xcand)

    # UCB: 既看高均值，也看高不确定性
    candidates["pred_mean"] = mean
    candidates["pred_std"] = std
    candidates["acq"] = candidates["pred_mean"] + UCB_BETA * candidates["pred_std"]

    # 去掉和历史中太接近的点（可选）
    # 这里简化不做复杂去重，直接按acq排序
    topk = candidates.sort_values("acq", ascending=False).head(N_REAL_EVALS).reset_index(drop=True)

    topk.to_csv(OUTPUT_DIR / "selected_candidates.csv", index=False, encoding="utf-8-sig")
    print("\n===== 代理模型挑出的候选 =====")
    print(topk)

    # 真实训练这些候选
    real_rows = []
    for i, row in topk.iterrows():
        real_rows.append(real_evaluate(row, i))

    real_df = pd.DataFrame(real_rows)
    real_df.to_csv(OUTPUT_DIR / "real_eval_results.csv", index=False, encoding="utf-8-sig")

    print("\n===== 真实评估结果 =====")
    print(real_df)


if __name__ == "__main__":
    main()
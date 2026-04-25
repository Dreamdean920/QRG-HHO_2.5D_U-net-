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

OUT_ROOT = Path("outputs/surrogate_loop")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

PYTHON_EXE = sys.executable

SPACE = {
    "lr": (0.0012, 0.0019),
    "dice_weight": (0.62, 0.80),
    "batch_size": [4],
}

N_CANDIDATES = 3000
N_REAL_PER_ROUND = 3
N_ROUNDS = 5

UCB_BETA = 1.5
SEED = 42

# =========================
# 工具函数
# =========================
def sample_candidates(n, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        rows.append({
            "lr": float(rng.uniform(*SPACE["lr"])),
            "dice_weight": float(rng.uniform(*SPACE["dice_weight"])),
            "batch_size": int(rng.choice(SPACE["batch_size"]))
        })
    return pd.DataFrame(rows)


def real_eval(row, run_name):
    output_json = OUT_ROOT / f"{run_name}.json"

    cmd = [
        PYTHON_EXE,
        "scripts/train_week3_unet.py",
        "--input_mode", "2.5d",
        "--no_post",
        "--save_root", str(OUT_ROOT / "train_runs"),
        "--epochs", "20",
        "--seed", str(SEED),
        "--batch_size", str(int(row["batch_size"])),
        "--lr", str(float(row["lr"])),
        "--dice_weight", str(float(row["dice_weight"])),
        "--run_name", run_name,
        "--output_json", str(output_json),
    ]

    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd)

    if not output_json.exists():
        return None

    data = json.loads(output_json.read_text(encoding="utf-8"))

    return {
        "run_name": run_name,
        "lr": row["lr"],
        "dice_weight": row["dice_weight"],
        "batch_size": row["batch_size"],
        "best_val_dice": data.get("best_val_dice"),
        "test_dice": data.get("test_dice"),
        "time_sec": data.get("time_sec"),
        "status": data.get("status", "OK"),
    }


# =========================
# 主流程
# =========================
def main():
    history = load_history_from_csv(HISTORY_CSV)

    # 统一口径
    history = history[history["batch_size"] == 4].copy()

    print("初始历史数量:", len(history))

    all_new = []

    for round_idx in range(N_ROUNDS):

        print(f"\n========== ROUND {round_idx} ==========")

        # ===== 1. 训练代理 =====
        surrogate = RFEnsembleSurrogate()
        surrogate.fit(history)

        # ===== 2. 采样候选 =====
        candidates = sample_candidates(N_CANDIDATES, seed=SEED + round_idx)

        X = candidates[["lr", "dice_weight", "batch_size"]].values
        mean, std = surrogate.predict_mean_std(X)

        candidates["pred_mean"] = mean
        candidates["pred_std"] = std
        candidates["acq"] = mean + UCB_BETA * std

        # ===== 3. 选最优点 =====
        topk = candidates.sort_values("acq", ascending=False).head(N_REAL_PER_ROUND)

        print("\n候选参数：")
        print(topk[["lr", "dice_weight", "pred_mean", "pred_std"]])

        # ===== 4. 真实训练 =====
        new_rows = []
        for i, row in topk.iterrows():
            run_name = f"round{round_idx}_trial{i}"
            res = real_eval(row, run_name)
            if res:
                new_rows.append(res)

        new_df = pd.DataFrame(new_rows)

        if len(new_df) == 0:
            print("⚠️ 本轮没有有效结果")
            continue

        print("\n真实结果：")
        print(new_df[["best_val_dice", "test_dice"]])

        # ===== 5. 加入历史 =====
        history = pd.concat([history, new_df], ignore_index=True)
        all_new.append(new_df)

        # 保存
        history.to_csv(OUT_ROOT / "history_updated.csv", index=False)

    print("\n===== DONE =====")

    if all_new:
        final_df = pd.concat(all_new, ignore_index=True)
        final_df.to_csv(OUT_ROOT / "all_new_results.csv", index=False)


if __name__ == "__main__":
    main()
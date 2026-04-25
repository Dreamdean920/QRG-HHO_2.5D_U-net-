import os
import sys
import json
import math
import time
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import optuna
except Exception:
    optuna = None


# =========================================================
# 0. 基础配置
# =========================================================
PROJECT_ROOT = Path(".")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "week4_compare"
TRAIN_SAVE_ROOT = OUTPUT_ROOT / "train_runs"
TRIAL_JSON_ROOT = OUTPUT_ROOT / "trial_json"
RESULT_CSV = OUTPUT_ROOT / "compare_raw.csv"
SUMMARY_CSV = OUTPUT_ROOT / "compare_summary.csv"

os.makedirs(TRAIN_SAVE_ROOT, exist_ok=True)
os.makedirs(TRIAL_JSON_ROOT, exist_ok=True)

PYTHON_EXE = sys.executable

# 统一实验协议（和你当前QRG-HGS主线对齐）
COMMON_ARGS = [
    "--input_mode", "2.5d",
    "--no_post",
    "--save_root", str(TRAIN_SAVE_ROOT),
    "--epochs", "20",
]

# 搜索空间（按你当前强区间）
SPACE = {
    "lr": ("float", 0.0012, 0.0019),
    "dice_weight": ("float", 0.62, 0.80),
    "batch_size": ("categorical", [4]),   # 你当前结果显示 batch=4 最稳，先固定公平对比
}

SEED = 42
SEARCH_BUDGET = 72   # 和你QRG-HGS当前大约评估次数对齐
HGS_POP_SIZE = 6     # 可调
HGS_MAX_ITERS = SEARCH_BUDGET // HGS_POP_SIZE  # 72/6 = 12


# =========================================================
# 1. 工具函数
# =========================================================
def sample_random_params(rng: random.Random):
    params = {}
    for k, spec in SPACE.items():
        if spec[0] == "float":
            _, low, high = spec
            params[k] = rng.uniform(low, high)
        elif spec[0] == "categorical":
            _, choices = spec
            params[k] = rng.choice(choices)
        else:
            raise ValueError(spec[0])
    return params


def clip_params(params):
    out = {}
    for k, spec in SPACE.items():
        if spec[0] == "float":
            _, low, high = spec
            out[k] = float(min(max(params[k], low), high))
        elif spec[0] == "categorical":
            _, choices = spec
            if params[k] not in choices:
                out[k] = choices[0]
            else:
                out[k] = params[k]
    return out


def params_to_vec(params):
    return np.array([params["lr"], params["dice_weight"]], dtype=np.float64)


def vec_to_params(vec):
    return {
        "lr": float(vec[0]),
        "dice_weight": float(vec[1]),
        "batch_size": 4,
    }


def make_run_name(method: str, trial_idx: int):
    return f"{method}_trial{trial_idx:03d}"


def load_existing_json(json_path: Path):
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("status", "") == "OK":
            return data
    except Exception:
        return None
    return None


def evaluate_params(method: str, trial_idx: int, params: dict, seed: int = 42):
    run_name = make_run_name(method, trial_idx)
    output_json = TRIAL_JSON_ROOT / f"{run_name}.json"

    # 断点续跑：已有成功结果直接跳过
    old = load_existing_json(output_json)
    if old is not None:
        row = {
            "method": method,
            "trial_idx": trial_idx,
            "run_name": run_name,
            "seed": seed,
            "lr": params["lr"],
            "dice_weight": params["dice_weight"],
            "batch_size": params["batch_size"],
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
        print(f"[SKIP] {run_name}")
        return row

    cmd = [
        PYTHON_EXE,
        "scripts/train_week3_unet.py",
        *COMMON_ARGS,
        "--seed", str(seed),
        "--batch_size", str(params["batch_size"]),
        "--lr", str(params["lr"]),
        "--dice_weight", str(params["dice_weight"]),
        "--run_name", run_name,
        "--output_json", str(output_json),
    ]

    print("\n" + "=" * 80)
    print(f"[RUN] {run_name}")
    print(" ".join(cmd))
    print("=" * 80)

    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        return {
            "method": method,
            "trial_idx": trial_idx,
            "run_name": run_name,
            "seed": seed,
            "lr": params["lr"],
            "dice_weight": params["dice_weight"],
            "batch_size": params["batch_size"],
            "best_val_dice": None,
            "test_dice": None,
            "test_iou": None,
            "test_sens": None,
            "test_spec": None,
            "time_sec": None,
            "status": "ERROR",
            "error_message": "subprocess_failed",
            "best_ckpt_path": "",
            "resumed": False,
        }

    data = load_existing_json(output_json)
    if data is None:
        return {
            "method": method,
            "trial_idx": trial_idx,
            "run_name": run_name,
            "seed": seed,
            "lr": params["lr"],
            "dice_weight": params["dice_weight"],
            "batch_size": params["batch_size"],
            "best_val_dice": None,
            "test_dice": None,
            "test_iou": None,
            "test_sens": None,
            "test_spec": None,
            "time_sec": None,
            "status": "ERROR",
            "error_message": "missing_output_json_or_not_ok",
            "best_ckpt_path": "",
            "resumed": False,
        }

    return {
        "method": method,
        "trial_idx": trial_idx,
        "run_name": run_name,
        "seed": seed,
        "lr": params["lr"],
        "dice_weight": params["dice_weight"],
        "batch_size": params["batch_size"],
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


def save_results(rows):
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")

    ok_df = df[df["status"] == "OK"].copy()
    summary_rows = []
    for method, g in ok_df.groupby("method"):
        summary_rows.append({
            "method": method,
            "n_ok": len(g),
            "best_val_dice": g["best_val_dice"].max(),
            "best_test_dice": g["test_dice"].max(),
            "mean_test_dice": g["test_dice"].mean(),
            "std_test_dice": g["test_dice"].std(ddof=0),
            "mean_test_iou": g["test_iou"].mean(),
            "std_test_iou": g["test_iou"].std(ddof=0),
            "mean_test_sens": g["test_sens"].mean(),
            "mean_test_spec": g["test_spec"].mean(),
            "mean_time_sec": g["time_sec"].mean(),
        })

    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df) > 0:
        summary_df = summary_df.sort_values(by="mean_test_dice", ascending=False)
        summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    return df, summary_df


# =========================================================
# 2. Random Search
# =========================================================
def run_random(rows):
    rng = random.Random(SEED)
    for trial_idx in range(SEARCH_BUDGET):
        params = sample_random_params(rng)
        row = evaluate_params("random", trial_idx, params, seed=SEED)
        rows.append(row)
        save_results(rows)
    return rows


# =========================================================
# 3. TPE
# =========================================================
def run_tpe(rows):
    if optuna is None:
        raise RuntimeError("当前环境没有安装 optuna，请先执行: pip install optuna")

    def objective(trial):
        params = {
            "lr": trial.suggest_float("lr", 0.0012, 0.0019),
            "dice_weight": trial.suggest_float("dice_weight", 0.62, 0.80),
            "batch_size": 4,
        }
        trial_idx = trial.number
        row = evaluate_params("tpe", trial_idx, params, seed=SEED)
        rows.append(row)
        save_results(rows)

        if row["status"] != "OK" or row["best_val_dice"] is None:
            return 0.0
        return float(row["best_val_dice"])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=SEARCH_BUDGET)
    return rows


# =========================================================
# 4. HGS（简化公平版）
# =========================================================
def run_hgs(rows):
    rng = np.random.default_rng(SEED)

    lb = np.array([0.0012, 0.62], dtype=np.float64)
    ub = np.array([0.0019, 0.80], dtype=np.float64)

    pop = lb + (ub - lb) * rng.random((HGS_POP_SIZE, 2))
    fitness = np.full(HGS_POP_SIZE, np.inf, dtype=np.float64)

    trial_idx = 0

    # 初始种群评估
    for i in range(HGS_POP_SIZE):
        params = vec_to_params(pop[i])
        row = evaluate_params("hgs", trial_idx, params, seed=SEED)
        rows.append(row)
        save_results(rows)

        if row["status"] == "OK" and row["best_val_dice"] is not None:
            fitness[i] = -float(row["best_val_dice"])
        else:
            fitness[i] = 1e9
        trial_idx += 1

    best_idx = int(np.argmin(fitness))
    best_pos = pop[best_idx].copy()
    best_fit = float(fitness[best_idx])

    for it in range(HGS_MAX_ITERS):
        shrink = 2.0 * (1.0 - it / max(HGS_MAX_ITERS, 1))
        worst_fit = float(np.max(fitness))
        new_pop = pop.copy()

        for i in range(HGS_POP_SIZE):
            Xi = pop[i].copy()

            if i == best_idx:
                noise = 0.01 * (ub - lb) * rng.normal(size=2)
                cand = Xi + noise
            else:
                Fi = float(fitness[i])
                BF = float(best_fit)
                WF = float(worst_fit)

                if abs(WF - BF) < 1e-12:
                    hunger_ratio = 0.0
                else:
                    hunger_ratio = (Fi - BF) / (WF - BF + 1e-12)

                E = 2.0 / (math.exp(abs(Fi - BF)) + math.exp(-abs(Fi - BF)))
                R = 2.0 * shrink * rng.random(2) - shrink
                W1 = 1.0 if rng.random() > 0.5 else (1.0 + hunger_ratio * rng.random())
                W2 = (1.0 - math.exp(-abs(hunger_ratio))) * rng.random() * 2.0

                if rng.random() < 0.3:
                    cand = Xi * (1.0 + rng.normal(0, 0.05, size=2))
                else:
                    diff = np.abs(best_pos - Xi)
                    if rng.random() > E:
                        cand = W1 * best_pos + R * W2 * diff
                    else:
                        cand = W1 * best_pos - R * W2 * diff

                cand += 0.01 * (ub - lb) * rng.normal(size=2)

            cand = np.clip(cand, lb, ub)
            new_pop[i] = cand

        new_fitness = np.full(HGS_POP_SIZE, np.inf, dtype=np.float64)
        for i in range(HGS_POP_SIZE):
            params = vec_to_params(new_pop[i])
            row = evaluate_params("hgs", trial_idx, params, seed=SEED)
            rows.append(row)
            save_results(rows)

            if row["status"] == "OK" and row["best_val_dice"] is not None:
                new_fitness[i] = -float(row["best_val_dice"])
            else:
                new_fitness[i] = 1e9
            trial_idx += 1

        improved = new_fitness < fitness
        pop[improved] = new_pop[improved]
        fitness[improved] = new_fitness[improved]

        best_idx = int(np.argmin(fitness))
        best_pos = pop[best_idx].copy()
        best_fit = float(fitness[best_idx])

        if trial_idx >= SEARCH_BUDGET:
            break

    return rows


# =========================================================
# 5. 主入口
# =========================================================
if __name__ == "__main__":
    rows = []
    if RESULT_CSV.exists():
        try:
            old_df = pd.read_csv(RESULT_CSV)
            rows = old_df.to_dict("records")
            print(f"[INFO] 读取已有结果: {len(rows)} 条")
        except Exception:
            rows = []

    # 你可以注释掉某些行，只跑某一个方法
    rows = run_random(rows)
    rows = run_tpe(rows)
    rows = run_hgs(rows)

    df, summary_df = save_results(rows)

    print("\n========== RAW RESULTS ==========")
    print(df.tail())

    print("\n========== SUMMARY ==========")
    print(summary_df)
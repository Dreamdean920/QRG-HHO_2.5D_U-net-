import os
import sys
import json
import math
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

# 建议用新目录，避免和你旧的 compare_raw.csv 混在一起
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "week4_compare_all"
TRAIN_SAVE_ROOT = OUTPUT_ROOT / "train_runs"
TRIAL_JSON_ROOT = OUTPUT_ROOT / "trial_json"
RESULT_CSV = OUTPUT_ROOT / "compare_raw.csv"
SUMMARY_CSV = OUTPUT_ROOT / "compare_summary.csv"

os.makedirs(TRAIN_SAVE_ROOT, exist_ok=True)
os.makedirs(TRIAL_JSON_ROOT, exist_ok=True)

PYTHON_EXE = sys.executable

# 统一实验协议
COMMON_ARGS = [
    "--input_mode", "2.5d",
    "--no_post",
    "--save_root", str(TRAIN_SAVE_ROOT),
    "--epochs", "20",
]

# 搜索空间（与你当前强区间一致）
SPACE = {
    "lr": ("float", 0.0012, 0.0019),
    "dice_weight": ("float", 0.62, 0.80),
    "batch_size": ("categorical", [4]),   # 当前统一固定为4，保证公平
}

SEED = 42
SEARCH_BUDGET = 72

# HGS / QRG-HGS 配置
POP_SIZE = 6
MAX_ITERS = SEARCH_BUDGET // POP_SIZE  # 72/6 = 12

# QRG-HGS 配置（接近你当前主线）
QRG_START_RATIO = 0.5
QRG_THETA_INIT = 0.3141592654
QRG_THETA_FINAL = 0.0523598776
STAGNATION_PATIENCE = 2
ELITE_KEEP_RATIO = 0.2
RESTART_RATIO = 0.2


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
            out[k] = params[k] if params[k] in choices else choices[0]
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

    # 断点续跑
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

    # 去重后再汇总，避免 resumed 重复污染
    if len(df) > 0:
        df_clean = (
            df.sort_values(["method", "trial_idx", "resumed"])
              .drop_duplicates(subset=["method", "run_name"], keep="last")
        )
    else:
        df_clean = df.copy()

    ok_df = df_clean[df_clean["status"] == "OK"].copy()
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

    return df_clean, summary_df


def eval_population(method, rows, pop, trial_idx_start):
    """
    评估一批候选点，返回：
    rows, fitness(np.array), next_trial_idx
    fitness = -best_val_dice
    """
    fitness = []
    trial_idx = trial_idx_start

    for i in range(len(pop)):
        params = vec_to_params(pop[i])
        row = evaluate_params(method, trial_idx, params, seed=SEED)
        rows.append(row)
        save_results(rows)

        if row["status"] == "OK" and row["best_val_dice"] is not None:
            fitness.append(-float(row["best_val_dice"]))
        else:
            fitness.append(1e9)

        trial_idx += 1

    return rows, np.array(fitness, dtype=np.float64), trial_idx


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

    pop = lb + (ub - lb) * rng.random((POP_SIZE, 2))
    rows, fitness, trial_idx = eval_population("hgs", rows, pop, 0)

    best_idx = int(np.argmin(fitness))
    best_pos = pop[best_idx].copy()
    best_fit = float(fitness[best_idx])

    for it in range(MAX_ITERS):
        shrink = 2.0 * (1.0 - it / max(MAX_ITERS, 1))
        worst_fit = float(np.max(fitness))
        new_pop = pop.copy()

        for i in range(POP_SIZE):
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

        rows, new_fitness, trial_idx = eval_population("hgs", rows, new_pop, trial_idx)

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
# 5. QRG-HGS（停滞触发版，内联可运行）
# =========================================================
def run_qrghgs(rows):
    rng = np.random.default_rng(SEED)

    lb = np.array([0.0012, 0.62], dtype=np.float64)
    ub = np.array([0.0019, 0.80], dtype=np.float64)

    pop = lb + (ub - lb) * rng.random((POP_SIZE, 2))
    rows, fitness, trial_idx = eval_population("qrghgs", rows, pop, 0)

    best_idx = int(np.argmin(fitness))
    best_pos = pop[best_idx].copy()
    best_fit = float(fitness[best_idx])

    stagnation_count = 0
    qrg_start_iter = int(QRG_START_RATIO * MAX_ITERS)

    for it in range(MAX_ITERS):
        print(f"\n========== QRGHGS ITER {it+1}/{MAX_ITERS} ==========")
        print(f"[ITER] current best fitness = {best_fit:.8f}")

        shrink = 2.0 * (1.0 - it / max(MAX_ITERS, 1))
        worst_fit = float(np.max(fitness))
        new_pop = pop.copy()

        # -------- HGS更新 --------
        for i in range(POP_SIZE):
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

        rows, new_fitness, trial_idx = eval_population("qrghgs", rows, new_pop, trial_idx)

        improved = new_fitness < fitness
        pop[improved] = new_pop[improved]
        fitness[improved] = new_fitness[improved]

        old_best_fit = best_fit
        best_idx = int(np.argmin(fitness))
        best_pos = pop[best_idx].copy()
        best_fit = float(fitness[best_idx])

        if best_fit < old_best_fit - 1e-12:
            stagnation_count = 0
        else:
            stagnation_count += 1

        print(f"[ITER] after HGS best fitness = {best_fit:.8f}")
        print(f"[ITER] stagnation_count = {stagnation_count}")

        # -------- 停滞触发 QRG + 局部重启 --------
        if it >= qrg_start_iter and stagnation_count >= STAGNATION_PATIENCE:
            theta = QRG_THETA_INIT - (QRG_THETA_INIT - QRG_THETA_FINAL) * (it / max(MAX_ITERS - 1, 1))
            elite_k = max(1, int(math.ceil(ELITE_KEEP_RATIO * POP_SIZE)))
            restart_k = max(1, int(math.ceil(RESTART_RATIO * POP_SIZE)))

            elite_ids = np.argsort(fitness)[:elite_k]

            # 对 elite 做 QRG 微调
            qrg_pop = []
            for idx in elite_ids:
                Xi = pop[idx].copy()
                direction = best_pos - Xi
                eps = 0.005 * (ub - lb) * rng.normal(size=2)
                cand = Xi + theta * direction + eps
                cand = np.clip(cand, lb, ub)
                qrg_pop.append(cand)

            qrg_pop = np.array(qrg_pop, dtype=np.float64)
            rows, qrg_fit, trial_idx = eval_population("qrghgs", rows, qrg_pop, trial_idx)

            # 回写 elite
            for j, idx in enumerate(elite_ids):
                if qrg_fit[j] < fitness[idx]:
                    pop[idx] = qrg_pop[j]
                    fitness[idx] = qrg_fit[j]

            # 对最差个体做局部重启
            worst_ids = np.argsort(fitness)[-restart_k:]
            sigma = 0.08 * (1.0 - it / max(MAX_ITERS, 1))
            for idx in worst_ids:
                cand = best_pos + sigma * (ub - lb) * rng.normal(size=2)
                cand = np.clip(cand, lb, ub)
                pop[idx] = cand

            rows, restart_fit, trial_idx = eval_population("qrghgs", rows, pop[worst_ids], trial_idx)
            for j, idx in enumerate(worst_ids):
                fitness[idx] = restart_fit[j]

            best_idx = int(np.argmin(fitness))
            best_pos = pop[best_idx].copy()
            best_fit = float(fitness[best_idx])

            print(f"[ITER] after QRG/restart best fitness = {best_fit:.8f}")
            stagnation_count = 0

        if trial_idx >= SEARCH_BUDGET:
            break

    return rows


# =========================================================
# 6. 主入口
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

    # 你可以临时注释某些行，只跑某个方法
    rows = run_random(rows)
    rows = run_tpe(rows)
    rows = run_hgs(rows)
    rows = run_qrghgs(rows)

    df, summary_df = save_results(rows)

    print("\n========== RAW RESULTS (DEDUP) ==========")
    print(df.tail())

    print("\n========== SUMMARY ==========")
    print(summary_df)
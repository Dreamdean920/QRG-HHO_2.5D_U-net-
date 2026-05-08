import math
import sys
import json
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
OUT_ROOT = PROJECT_ROOT / "outputs" / "real_compare_7methods"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

TRAIN_SAVE_ROOT = OUT_ROOT / "train_runs"
TRIAL_JSON_ROOT = OUT_ROOT / "trial_json"
TRAIN_SAVE_ROOT.mkdir(parents=True, exist_ok=True)
TRIAL_JSON_ROOT.mkdir(parents=True, exist_ok=True)

RAW_CSV = OUT_ROOT / "compare_raw.csv"
SUMMARY_CSV = OUT_ROOT / "compare_summary.csv"

PYTHON_EXE = sys.executable

# 真实训练统一协议
COMMON_ARGS = [
    "--input_mode", "2.5d",
    "--no_post",
    "--save_root", str(TRAIN_SAVE_ROOT),
    "--epochs", "20",
]

# 当前扩大的搜索空间
SPACE = {
    "lr": (0.0005, 0.0030),
    "dice_weight": (0.50, 0.85),
    "batch_size": [2, 4, 8],
}
BATCH_CHOICES = SPACE["batch_size"]

SEED = 42

# 建议先 12 验证，正式可改 18 或 24
SEARCH_BUDGET = 30

METHODS = [
    "qrghgs_baseline",   # 主方法
    "qrghgs_hybrid",     # 新增：混合策略版
    "random",
    "tpe",
    "pso",
    "ga",
    "hgs",
]

POP_SIZE = 6


# =========================================================
# 1. 工具函数
# =========================================================
def clip_vec(x):
    x = np.array(x, dtype=np.float64)
    x[0] = np.clip(x[0], SPACE["lr"][0], SPACE["lr"][1])
    x[1] = np.clip(x[1], SPACE["dice_weight"][0], SPACE["dice_weight"][1])
    x[2] = np.clip(x[2], 0, len(BATCH_CHOICES) - 1)
    return x


def vec_to_params(x):
    x = clip_vec(x)
    batch_idx = int(round(x[2]))
    batch_idx = max(0, min(batch_idx, len(BATCH_CHOICES) - 1))
    return {
        "lr": float(x[0]),
        "dice_weight": float(x[1]),
        "batch_size": int(BATCH_CHOICES[batch_idx]),
    }


def params_to_vec(params):
    batch_idx = BATCH_CHOICES.index(int(params["batch_size"]))
    return np.array([params["lr"], params["dice_weight"], batch_idx], dtype=np.float64)


def sample_random_vec(rng):
    batch_idx = rng.randint(0, len(BATCH_CHOICES) - 1)
    return np.array([
        rng.uniform(*SPACE["lr"]),
        rng.uniform(*SPACE["dice_weight"]),
        batch_idx,
    ], dtype=np.float64)


def make_run_name(method, trial_idx):
    return f"{method}_trial{trial_idx:03d}"


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


def real_evaluate(method, trial_idx, params):
    run_name = make_run_name(method, trial_idx)
    output_json = TRIAL_JSON_ROOT / f"{run_name}.json"

    # ===== 断点续跑：已有成功 json 直接复用 =====
    old = load_json_if_ok(output_json)
    if old is not None:
        print(f"[SKIP] {run_name}")
        return {
            "method": method,
            "trial_idx": trial_idx,
            "run_name": run_name,
            "seed": SEED,
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

    cmd = [
        PYTHON_EXE,
        "scripts/train_week3_unet.py",
        *COMMON_ARGS,
        "--seed", str(SEED),
        "--batch_size", str(int(params["batch_size"])),
        "--lr", str(float(params["lr"])),
        "--dice_weight", str(float(params["dice_weight"])),
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
            "seed": SEED,
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

    if not output_json.exists():
        return {
            "method": method,
            "trial_idx": trial_idx,
            "run_name": run_name,
            "seed": SEED,
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
            "error_message": "missing_output_json",
            "best_ckpt_path": "",
            "resumed": False,
        }

    data = json.loads(output_json.read_text(encoding="utf-8"))

    return {
        "method": method,
        "trial_idx": trial_idx,
        "run_name": run_name,
        "seed": SEED,
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
    df.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")

    if len(df) == 0:
        return df, pd.DataFrame()

    df_clean = (
        df.sort_values(["method", "trial_idx", "resumed"])
          .drop_duplicates(subset=["method", "run_name"], keep="last")
          .reset_index(drop=True)
    )

    ok_df = df_clean[df_clean["status"] == "OK"].copy()
    summary_rows = []

    for method, g in ok_df.groupby("method"):
        summary_rows.append({
            "method": method,
            "n_ok": len(g),
            "best_val_dice_mean": g["best_val_dice"].mean(),
            "best_val_dice_max": g["best_val_dice"].max(),
            "best_val_dice_std": g["best_val_dice"].std(ddof=0),
            "test_dice_mean": g["test_dice"].mean(),
            "test_dice_max": g["test_dice"].max(),
            "test_dice_std": g["test_dice"].std(ddof=0),
            "test_iou_mean": g["test_iou"].mean(),
            "test_sens_mean": g["test_sens"].mean(),
            "test_spec_mean": g["test_spec"].mean(),
            "time_sec_mean": g["time_sec"].mean(),
            "time_sec_max": g["time_sec"].max(),
        })

    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df) > 0:
        summary_df = summary_df.sort_values("test_dice_mean", ascending=False)
        summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    return df_clean, summary_df


# =========================================================
# 2. Random / TPE
# =========================================================
def run_random(rows):
    rng = random.Random(SEED)
    for trial_idx in range(SEARCH_BUDGET):
        params = vec_to_params(sample_random_vec(rng))
        row = real_evaluate("random", trial_idx, params)
        rows.append(row)
        save_results(rows)
    return rows


def run_tpe(rows):
    if optuna is None:
        raise RuntimeError("当前环境没有安装 optuna，请先执行: pip install optuna")

    def objective(trial):
        params = {
            "lr": trial.suggest_float("lr", *SPACE["lr"]),
            "dice_weight": trial.suggest_float("dice_weight", *SPACE["dice_weight"]),
            "batch_size": trial.suggest_categorical("batch_size", BATCH_CHOICES),
        }
        trial_idx = trial.number
        row = real_evaluate("tpe", trial_idx, params)
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
# 3. PSO / GA / HGS
# =========================================================
def run_pso(rows):
    rng = np.random.default_rng(SEED)
    max_iters = max(1, math.ceil(SEARCH_BUDGET / POP_SIZE))

    lb = np.array([SPACE["lr"][0], SPACE["dice_weight"][0], 0], dtype=np.float64)
    ub = np.array([SPACE["lr"][1], SPACE["dice_weight"][1], len(BATCH_CHOICES) - 1], dtype=np.float64)

    X = lb + (ub - lb) * rng.random((POP_SIZE, 3))
    V = np.zeros_like(X)
    pbest = X.copy()
    pbest_fit = np.full(POP_SIZE, -1e18)
    gbest = None
    gbest_fit = -1e18

    trial_idx = 0

    for _ in range(max_iters):
        for i in range(POP_SIZE):
            params = vec_to_params(X[i])
            row = real_evaluate("pso", trial_idx, params)
            rows.append(row)
            save_results(rows)

            fit = -1e18 if row["status"] != "OK" or row["best_val_dice"] is None else float(row["best_val_dice"])

            if fit > pbest_fit[i]:
                pbest_fit[i] = fit
                pbest[i] = X[i].copy()

            if fit > gbest_fit:
                gbest_fit = fit
                gbest = X[i].copy()

            trial_idx += 1
            if trial_idx >= SEARCH_BUDGET:
                return rows

        for i in range(POP_SIZE):
            w, c1, c2 = 0.7, 1.5, 1.5
            r1 = rng.random(3)
            r2 = rng.random(3)
            V[i] = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (gbest - X[i])
            X[i] = clip_vec(X[i] + V[i])

    return rows


def run_ga(rows):
    rng = np.random.default_rng(SEED)
    max_gens = max(1, math.ceil(SEARCH_BUDGET / POP_SIZE))

    pop = np.array([sample_random_vec(random.Random(SEED + i)) for i in range(POP_SIZE)])
    trial_idx = 0

    for _ in range(max_gens):
        fits = []

        for i in range(POP_SIZE):
            params = vec_to_params(pop[i])
            row = real_evaluate("ga", trial_idx, params)
            rows.append(row)
            save_results(rows)

            fit = -1e18 if row["status"] != "OK" or row["best_val_dice"] is None else float(row["best_val_dice"])
            fits.append(fit)

            trial_idx += 1
            if trial_idx >= SEARCH_BUDGET:
                return rows

        fits = np.array(fits)
        elite_ids = np.argsort(-fits)[:2]
        new_pop = [pop[elite_ids[0]].copy(), pop[elite_ids[1]].copy()]

        while len(new_pop) < POP_SIZE:
            p1 = pop[rng.integers(0, POP_SIZE)]
            p2 = pop[rng.integers(0, POP_SIZE)]
            alpha = rng.random()
            child = alpha * p1 + (1 - alpha) * p2
            child += rng.normal(0, [0.00008, 0.02, 0.25], size=3)
            child = clip_vec(child)
            new_pop.append(child)

        pop = np.array(new_pop)

    return rows


def run_hgs(rows, method_name="hgs"):
    rng = np.random.default_rng(SEED)
    max_iters = max(1, math.ceil(SEARCH_BUDGET / POP_SIZE))

    lb = np.array([SPACE["lr"][0], SPACE["dice_weight"][0], 0], dtype=np.float64)
    ub = np.array([SPACE["lr"][1], SPACE["dice_weight"][1], len(BATCH_CHOICES) - 1], dtype=np.float64)

    pop = lb + (ub - lb) * rng.random((POP_SIZE, 3))
    fit = np.full(POP_SIZE, -1e18)

    trial_idx = 0

    # 初始评估
    for i in range(POP_SIZE):
        params = vec_to_params(pop[i])
        row = real_evaluate(method_name, trial_idx, params)
        rows.append(row)
        save_results(rows)

        fit[i] = -1e18 if row["status"] != "OK" or row["best_val_dice"] is None else float(row["best_val_dice"])

        trial_idx += 1
        if trial_idx >= SEARCH_BUDGET:
            return rows

    best_idx = int(np.argmax(fit))
    best_pos = pop[best_idx].copy()
    best_fit = float(fit[best_idx])

    for it in range(max_iters):
        shrink = 2.0 * (1.0 - it / max(max_iters, 1))
        worst_fit = float(np.min(fit))
        new_pop = pop.copy()

        for i in range(POP_SIZE):
            Xi = pop[i].copy()

            if i == best_idx:
                cand = Xi + 0.01 * (ub - lb) * rng.normal(size=3)
            else:
                Fi, BF, WF = float(fit[i]), float(best_fit), float(worst_fit)

                if abs(BF - WF) < 1e-12:
                    hunger_ratio = 0.0
                else:
                    hunger_ratio = (BF - Fi) / (BF - WF + 1e-12)

                E = 2.0 / (math.exp(abs(BF - Fi)) + math.exp(-abs(BF - Fi)))
                R = 2.0 * shrink * rng.random(3) - shrink
                W1 = 1.0 if rng.random() > 0.5 else (1.0 + hunger_ratio * rng.random())
                W2 = (1.0 - math.exp(-abs(hunger_ratio))) * rng.random() * 2.0

                diff = np.abs(best_pos - Xi)
                if rng.random() < 0.3:
                    cand = Xi * (1.0 + rng.normal(0, 0.05, size=3))
                else:
                    cand = W1 * best_pos + R * W2 * diff if rng.random() > E else W1 * best_pos - R * W2 * diff

            new_pop[i] = clip_vec(cand)

        for i in range(POP_SIZE):
            params = vec_to_params(new_pop[i])
            row = real_evaluate(method_name, trial_idx, params)
            rows.append(row)
            save_results(rows)

            new_fit = -1e18 if row["status"] != "OK" or row["best_val_dice"] is None else float(row["best_val_dice"])
            if new_fit > fit[i]:
                fit[i] = new_fit
                pop[i] = new_pop[i].copy()

            trial_idx += 1
            if trial_idx >= SEARCH_BUDGET:
                return rows

        best_idx = int(np.argmax(fit))
        best_pos = pop[best_idx].copy()
        best_fit = float(fit[best_idx])

    return rows


# =========================================================
# 4. 三个 QRG-HGS 版本
# =========================================================
def get_qrghgs_config(method):
    # 原始版：baseline 主方法
    if method == "qrghgs_baseline":
        return {
            "qrg_start_ratio": 0.30,
            "stagnation_patience": 2,
            "elite_keep_ratio": 0.20,
            "restart_ratio": 0.20,
            "theta_init": 0.3141592654,
            "theta_final": 0.0523598776,
            "use_perturb": False,
            "noise_mode": "fixed",
            "base_noise_fixed": 0.0,
            "qrg_noise_fixed": 0.0,
            "restart_noise_fixed": 0.0,
            "use_hybrid": False,
            "hybrid_restart_ratio": 0.0,
            "hybrid_local_noise": 0.0,
        }

    # 新增：混合策略版
    if method == "qrghgs_hybrid":
        return {
            "qrg_start_ratio": 0.30,
            "stagnation_patience": 2,
            "elite_keep_ratio": 0.20,
            "restart_ratio": 0.20,
            "theta_init": 0.3141592654,
            "theta_final": 0.0523598776,
            "use_perturb": False,
            "noise_mode": "fixed",
            "base_noise_fixed": 0.0,
            "qrg_noise_fixed": 0.0,
            "restart_noise_fixed": 0.0,
            "use_hybrid": True,
            "hybrid_restart_ratio": 0.20,
            "hybrid_local_noise": 0.08,
        }

    # 改进版：作为对比对象
    if method == "adaptive_qrghgs":
        return {
            "qrg_start_ratio": 0.18,
            "stagnation_patience": 1,
            "elite_keep_ratio": 0.30,
            "restart_ratio": 0.25,
            "theta_init": 0.3141592654,
            "theta_final": 0.0523598776,
            "use_perturb": True,
            "noise_mode": "increasing",
            "noise_gamma": 1.8,
            "base_noise_min": 0.010,
            "base_noise_max": 0.015,
            "qrg_noise_min": 0.004,
            "qrg_noise_max": 0.012,
            "restart_noise_min": 0.060,
            "restart_noise_max": 0.120,
            "use_hybrid": False,
            "hybrid_restart_ratio": 0.0,
            "hybrid_local_noise": 0.0,
        }

    return None


def get_noise_values(cfg, progress):
    if not cfg["use_perturb"]:
        return 0.0, 0.0, 0.0

    if cfg["noise_mode"] == "fixed":
        return (
            cfg["base_noise_fixed"],
            cfg["qrg_noise_fixed"],
            cfg["restart_noise_fixed"],
        )

    gamma = cfg["noise_gamma"]
    alpha = progress ** gamma

    base_noise = cfg["base_noise_min"] + (cfg["base_noise_max"] - cfg["base_noise_min"]) * alpha
    qrg_noise = cfg["qrg_noise_min"] + (cfg["qrg_noise_max"] - cfg["qrg_noise_min"]) * alpha
    restart_noise = cfg["restart_noise_min"] + (cfg["restart_noise_max"] - cfg["restart_noise_min"]) * alpha
    return base_noise, qrg_noise, restart_noise


def run_qrghgs(rows, method_name):
    cfg = get_qrghgs_config(method_name)
    rng = np.random.default_rng(SEED)
    max_iters = max(1, math.ceil(SEARCH_BUDGET / POP_SIZE))

    lb = np.array([SPACE["lr"][0], SPACE["dice_weight"][0], 0], dtype=np.float64)
    ub = np.array([SPACE["lr"][1], SPACE["dice_weight"][1], len(BATCH_CHOICES) - 1], dtype=np.float64)

    pop = lb + (ub - lb) * rng.random((POP_SIZE, 3))
    fit = np.full(POP_SIZE, -1e18)

    trial_idx = 0

    # 初始评估
    for i in range(POP_SIZE):
        params = vec_to_params(pop[i])
        row = real_evaluate(method_name, trial_idx, params)
        rows.append(row)
        save_results(rows)

        fit[i] = -1e18 if row["status"] != "OK" or row["best_val_dice"] is None else float(row["best_val_dice"])

        trial_idx += 1
        if trial_idx >= SEARCH_BUDGET:
            return rows

    best_idx = int(np.argmax(fit))
    best_pos = pop[best_idx].copy()
    best_fit = float(fit[best_idx])

    stagnation = 0
    qrg_start_iter = int(cfg["qrg_start_ratio"] * max_iters)

    for it in range(max_iters):
        old_best = best_fit
        shrink = 2.0 * (1.0 - it / max(max_iters, 1))
        worst_fit = float(np.min(fit))
        new_pop = pop.copy()

        progress = it / max(max_iters - 1, 1)
        base_noise_ratio, qrg_noise_ratio, restart_noise_ratio = get_noise_values(cfg, progress)

        # HGS 主体
        for i in range(POP_SIZE):
            Xi = pop[i].copy()

            if i == best_idx:
                cand = Xi + base_noise_ratio * (ub - lb) * rng.normal(size=3)
            else:
                Fi, BF, WF = float(fit[i]), float(best_fit), float(worst_fit)

                if abs(BF - WF) < 1e-12:
                    hunger_ratio = 0.0
                else:
                    hunger_ratio = (BF - Fi) / (BF - WF + 1e-12)

                E = 2.0 / (math.exp(abs(BF - Fi)) + math.exp(-abs(BF - Fi)))
                R = 2.0 * shrink * rng.random(3) - shrink
                W1 = 1.0 if rng.random() > 0.5 else (1.0 + hunger_ratio * rng.random())
                W2 = (1.0 - math.exp(-abs(hunger_ratio))) * rng.random() * 2.0

                diff = np.abs(best_pos - Xi)
                if rng.random() < 0.3:
                    cand = Xi * (1.0 + rng.normal(0, 0.05, size=3))
                else:
                    cand = W1 * best_pos + R * W2 * diff if rng.random() > E else W1 * best_pos - R * W2 * diff

            new_pop[i] = clip_vec(cand)

        # 评估 HGS 更新
        for i in range(POP_SIZE):
            params = vec_to_params(new_pop[i])
            row = real_evaluate(method_name, trial_idx, params)
            rows.append(row)
            save_results(rows)

            new_fit = -1e18 if row["status"] != "OK" or row["best_val_dice"] is None else float(row["best_val_dice"])
            if new_fit > fit[i]:
                fit[i] = new_fit
                pop[i] = new_pop[i].copy()

            trial_idx += 1
            if trial_idx >= SEARCH_BUDGET:
                return rows

        best_idx = int(np.argmax(fit))
        best_pos = pop[best_idx].copy()
        best_fit = float(fit[best_idx])

        if best_fit > old_best + 1e-12:
            stagnation = 0
        else:
            stagnation += 1

        if it >= qrg_start_iter and stagnation >= cfg["stagnation_patience"]:
            theta = cfg["theta_init"] - (cfg["theta_init"] - cfg["theta_final"]) * progress
            elite_k = max(1, math.ceil(cfg["elite_keep_ratio"] * POP_SIZE))
            elite_ids = np.argsort(-fit)[:elite_k]

            # QRG 局部开发
            for idx in elite_ids:
                Xi = pop[idx].copy()
                direction = best_pos - Xi
                qrg_move = theta * direction
                local_noise = qrg_noise_ratio * (ub - lb) * rng.normal(size=3)

                cand = Xi + qrg_move + local_noise
                cand = clip_vec(cand)

                params = vec_to_params(cand)
                row = real_evaluate(method_name, trial_idx, params)
                rows.append(row)
                save_results(rows)

                new_fit = -1e18 if row["status"] != "OK" or row["best_val_dice"] is None else float(row["best_val_dice"])
                if new_fit > fit[idx]:
                    fit[idx] = new_fit
                    pop[idx] = cand.copy()

                trial_idx += 1
                if trial_idx >= SEARCH_BUDGET:
                    return rows

            # 自适应扰动版的 restart
            if cfg["use_perturb"] and restart_noise_ratio > 0:
                worst_ids = np.argsort(fit)[:max(1, math.ceil(cfg["restart_ratio"] * POP_SIZE))]
                for idx in worst_ids:
                    cand = best_pos + restart_noise_ratio * (ub - lb) * rng.normal(size=3)
                    cand = clip_vec(cand)

                    params = vec_to_params(cand)
                    row = real_evaluate(method_name, trial_idx, params)
                    rows.append(row)
                    save_results(rows)

                    new_fit = -1e18 if row["status"] != "OK" or row["best_val_dice"] is None else float(row["best_val_dice"])
                    if new_fit > fit[idx]:
                        fit[idx] = new_fit
                        pop[idx] = cand.copy()

                    trial_idx += 1
                    if trial_idx >= SEARCH_BUDGET:
                        return rows

            # ===== 新增：混合策略版 =====
            if cfg.get("use_hybrid", False):
                hybrid_local_noise = cfg.get("hybrid_local_noise", 0.08)
                hybrid_restart_ratio = cfg.get("hybrid_restart_ratio", 0.20)

                # 围绕当前最优再做一轮保守局部细搜
                extra_elite_ids = np.argsort(-fit)[:max(1, math.ceil(0.2 * POP_SIZE))]
                for idx in extra_elite_ids:
                    Xi = pop[idx].copy()
                    direction = best_pos - Xi
                    cand = Xi + 0.25 * direction + hybrid_local_noise * (ub - lb) * rng.normal(size=3)
                    cand = clip_vec(cand)

                    params = vec_to_params(cand)
                    row = real_evaluate(method_name, trial_idx, params)
                    rows.append(row)
                    save_results(rows)

                    new_fit = -1e18 if row["status"] != "OK" or row["best_val_dice"] is None else float(row["best_val_dice"])
                    if new_fit > fit[idx]:
                        fit[idx] = new_fit
                        pop[idx] = cand.copy()

                    trial_idx += 1
                    if trial_idx >= SEARCH_BUDGET:
                        return rows

                # 对最差个体做小比例随机重启
                restart_k = max(1, math.ceil(hybrid_restart_ratio * POP_SIZE))
                worst_ids = np.argsort(fit)[:restart_k]
                for idx in worst_ids:
                    cand = lb + (ub - lb) * rng.random(3)
                    cand = clip_vec(cand)

                    params = vec_to_params(cand)
                    row = real_evaluate(method_name, trial_idx, params)
                    rows.append(row)
                    save_results(rows)

                    new_fit = -1e18 if row["status"] != "OK" or row["best_val_dice"] is None else float(row["best_val_dice"])
                    if new_fit > fit[idx]:
                        fit[idx] = new_fit
                        pop[idx] = cand.copy()

                    trial_idx += 1
                    if trial_idx >= SEARCH_BUDGET:
                        return rows

            stagnation = 0
            best_idx = int(np.argmax(fit))
            best_pos = pop[best_idx].copy()
            best_fit = float(fit[best_idx])

    return rows


# =========================================================
# 5. 主入口
# =========================================================
if __name__ == "__main__":
    rows = []
    if RAW_CSV.exists():
        try:
            old_df = pd.read_csv(RAW_CSV)
            rows = old_df.to_dict("records")
            print(f"[INFO] 已读取历史记录 {len(rows)} 条")
        except Exception:
            rows = []

    for method in METHODS:
        print(f"\n\n########## RUN METHOD: {method} ##########")

        if method == "qrghgs_baseline":
            rows = run_qrghgs(rows, "qrghgs_baseline")
        elif method == "qrghgs_hybrid":
            rows = run_qrghgs(rows, "qrghgs_hybrid")
        elif method == "adaptive_qrghgs":
            rows = run_qrghgs(rows, "adaptive_qrghgs")
        elif method == "random":
            rows = run_random(rows)
        elif method == "tpe":
            rows = run_tpe(rows)
        elif method == "pso":
            rows = run_pso(rows)
        elif method == "ga":
            rows = run_ga(rows)
        elif method == "hgs":
            rows = run_hgs(rows, "hgs")

        _, summary_df = save_results(rows)
        print("\n===== CURRENT SUMMARY =====")
        print(summary_df)

    print("\n✅ 全部方法真实训练对比完成。")
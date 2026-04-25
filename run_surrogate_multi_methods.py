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

from surrogate_model_checked import RFEnsembleSurrogate, load_history_from_csv


# =========================================================
# 0. 基础配置
# =========================================================
HISTORY_CSV = "outputs/history_augmented/all_methods_merged_plus_warmup.csv"
OUT_ROOT = Path("outputs/surrogate_multi_methods")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

PYTHON_EXE = sys.executable

SPACE = {
    "lr": (0.0005, 0.0030),
    "dice_weight": (0.50, 0.85),
    "batch_size": [2, 4, 8],
}
BATCH_CHOICES = SPACE["batch_size"]

SEED = 42
BETA = 1.5

METHODS = ["random", "tpe", "pso", "ga", "hgs", "qrghgs"]

# 每种方法在代理空间里的预算
SURROGATE_BUDGET = 30

# 每种方法最终真实评估几个点
TOPK_REAL_EVAL = 2


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


def real_evaluate(method, idx, row):
    run_name = f"sur_{method}_trial{idx:03d}"
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

    print("\n[REAL EVAL]", " ".join(cmd))
    proc = subprocess.run(cmd)

    if proc.returncode != 0 or not output_json.exists():
        return {
            "method": method,
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
        "method": method,
        "run_name": run_name,
        "lr": row["lr"],
        "dice_weight": row["dice_weight"],
        "batch_size": row["batch_size"],
        "status": data.get("status", "OK"),
        "best_val_dice": data.get("best_val_dice"),
        "test_dice": data.get("test_dice"),
        "time_sec": data.get("time_sec"),
    }


# =========================================================
# 2. 构建代理目标函数
# =========================================================
class SurrogateObjective:
    def __init__(self, surrogate, beta=1.5):
        self.surrogate = surrogate
        self.beta = beta

    def eval_vec(self, x):
        x = clip_vec(x)
        params = vec_to_params(x)
        X = np.array([[params["lr"], params["dice_weight"], params["batch_size"]]], dtype=np.float64)
        mean, std = self.surrogate.predict_mean_std(X)
        mean = float(mean[0])
        std = float(std[0])
        acq = mean + self.beta * std
        return acq, mean, std


# =========================================================
# 3. Random
# =========================================================
def run_random_surrogate(obj, budget, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(budget):
        x = sample_random_vec(rng)
        acq, mean, std = obj.eval_vec(x)
        p = vec_to_params(x)
        rows.append({
            "method": "random",
            "trial_idx": i,
            **p,
            "pred_mean": mean,
            "pred_std": std,
            "acq": acq,
        })
    return pd.DataFrame(rows)


# =========================================================
# 4. TPE
# =========================================================
def run_tpe_surrogate(obj, budget, seed):
    if optuna is None:
        print("[WARN] optuna 未安装，跳过 tpe")
        return pd.DataFrame()

    rows = []

    def objective(trial):
        lr = trial.suggest_float("lr", *SPACE["lr"])
        dice_weight = trial.suggest_float("dice_weight", *SPACE["dice_weight"])
        batch_size = trial.suggest_categorical("batch_size", BATCH_CHOICES)

        x = params_to_vec({
            "lr": lr,
            "dice_weight": dice_weight,
            "batch_size": batch_size,
        })
        acq, mean, std = obj.eval_vec(x)
        p = vec_to_params(x)
        rows.append({
            "method": "tpe",
            "trial_idx": trial.number,
            **p,
            "pred_mean": mean,
            "pred_std": std,
            "acq": acq,
        })
        return acq

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=budget)

    return pd.DataFrame(rows)


# =========================================================
# 5. PSO
# =========================================================
def run_pso_surrogate(obj, budget, seed, pop_size=6):
    rng = np.random.default_rng(seed)
    max_iters = max(1, math.ceil(budget / pop_size))

    lb = np.array([SPACE["lr"][0], SPACE["dice_weight"][0], 0], dtype=np.float64)
    ub = np.array([SPACE["lr"][1], SPACE["dice_weight"][1], len(BATCH_CHOICES) - 1], dtype=np.float64)

    X = lb + (ub - lb) * rng.random((pop_size, 3))
    V = np.zeros_like(X)

    pbest = X.copy()
    pbest_fit = np.full(pop_size, -1e18)
    gbest = None
    gbest_fit = -1e18

    rows = []
    trial_idx = 0

    for _ in range(max_iters):
        for i in range(pop_size):
            acq, mean, std = obj.eval_vec(X[i])
            p = vec_to_params(X[i])
            rows.append({
                "method": "pso",
                "trial_idx": trial_idx,
                **p,
                "pred_mean": mean,
                "pred_std": std,
                "acq": acq,
            })

            if acq > pbest_fit[i]:
                pbest_fit[i] = acq
                pbest[i] = X[i].copy()

            if acq > gbest_fit:
                gbest_fit = acq
                gbest = X[i].copy()

            trial_idx += 1
            if trial_idx >= budget:
                return pd.DataFrame(rows)

        for i in range(pop_size):
            w, c1, c2 = 0.7, 1.5, 1.5
            r1 = rng.random(3)
            r2 = rng.random(3)
            V[i] = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (gbest - X[i])
            X[i] = clip_vec(X[i] + V[i])

    return pd.DataFrame(rows)


# =========================================================
# 6. GA
# =========================================================
def run_ga_surrogate(obj, budget, seed, pop_size=6):
    rng = np.random.default_rng(seed)
    max_gens = max(1, math.ceil(budget / pop_size))

    pop = np.array([sample_random_vec(random.Random(seed + i)) for i in range(pop_size)])
    rows = []
    trial_idx = 0

    for _ in range(max_gens):
        fits = []
        for i in range(pop_size):
            acq, mean, std = obj.eval_vec(pop[i])
            p = vec_to_params(pop[i])
            rows.append({
                "method": "ga",
                "trial_idx": trial_idx,
                **p,
                "pred_mean": mean,
                "pred_std": std,
                "acq": acq,
            })
            fits.append(acq)
            trial_idx += 1
            if trial_idx >= budget:
                return pd.DataFrame(rows)

        fits = np.array(fits)
        elite_ids = np.argsort(-fits)[:2]
        new_pop = [pop[elite_ids[0]].copy(), pop[elite_ids[1]].copy()]

        while len(new_pop) < pop_size:
            p1 = pop[rng.integers(0, pop_size)]
            p2 = pop[rng.integers(0, pop_size)]
            alpha = rng.random()
            child = alpha * p1 + (1 - alpha) * p2
            child += rng.normal(0, [0.00008, 0.02, 0.25], size=3)
            child = clip_vec(child)
            new_pop.append(child)

        pop = np.array(new_pop)

    return pd.DataFrame(rows)


# =========================================================
# 7. HGS
# =========================================================
def run_hgs_surrogate(obj, budget, seed, pop_size=6):
    rng = np.random.default_rng(seed)
    max_iters = max(1, math.ceil(budget / pop_size))

    lb = np.array([SPACE["lr"][0], SPACE["dice_weight"][0], 0], dtype=np.float64)
    ub = np.array([SPACE["lr"][1], SPACE["dice_weight"][1], len(BATCH_CHOICES) - 1], dtype=np.float64)

    pop = lb + (ub - lb) * rng.random((pop_size, 3))
    fit = np.full(pop_size, -1e18)

    rows = []
    trial_idx = 0

    for i in range(pop_size):
        acq, mean, std = obj.eval_vec(pop[i])
        p = vec_to_params(pop[i])
        rows.append({
            "method": "hgs",
            "trial_idx": trial_idx,
            **p,
            "pred_mean": mean,
            "pred_std": std,
            "acq": acq,
        })
        fit[i] = acq
        trial_idx += 1
        if trial_idx >= budget:
            return pd.DataFrame(rows)

    best_idx = int(np.argmax(fit))
    best_pos = pop[best_idx].copy()
    best_fit = float(fit[best_idx])

    for it in range(max_iters):
        shrink = 2.0 * (1.0 - it / max(max_iters, 1))
        worst_fit = float(np.min(fit))
        new_pop = pop.copy()

        for i in range(pop_size):
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
                    if rng.random() > E:
                        cand = W1 * best_pos + R * W2 * diff
                    else:
                        cand = W1 * best_pos - R * W2 * diff

            cand = clip_vec(cand)
            new_pop[i] = cand

        for i in range(pop_size):
            acq, mean, std = obj.eval_vec(new_pop[i])
            p = vec_to_params(new_pop[i])
            rows.append({
                "method": "hgs",
                "trial_idx": trial_idx,
                **p,
                "pred_mean": mean,
                "pred_std": std,
                "acq": acq,
            })

            if acq > fit[i]:
                fit[i] = acq
                pop[i] = new_pop[i].copy()

            trial_idx += 1
            if trial_idx >= budget:
                return pd.DataFrame(rows)

        best_idx = int(np.argmax(fit))
        best_pos = pop[best_idx].copy()
        best_fit = float(fit[best_idx])

    return pd.DataFrame(rows)


# =========================================================
# 8. QRG-HGS
# =========================================================
def run_qrghgs_surrogate(obj, budget, seed, pop_size=6):
    rng = np.random.default_rng(seed)
    max_iters = max(1, math.ceil(budget / pop_size))

    lb = np.array([SPACE["lr"][0], SPACE["dice_weight"][0], 0], dtype=np.float64)
    ub = np.array([SPACE["lr"][1], SPACE["dice_weight"][1], len(BATCH_CHOICES) - 1], dtype=np.float64)

    pop = lb + (ub - lb) * rng.random((pop_size, 3))
    fit = np.full(pop_size, -1e18)

    rows = []
    trial_idx = 0

    for i in range(pop_size):
        acq, mean, std = obj.eval_vec(pop[i])
        p = vec_to_params(pop[i])
        rows.append({
            "method": "qrghgs",
            "trial_idx": trial_idx,
            **p,
            "pred_mean": mean,
            "pred_std": std,
            "acq": acq,
        })
        fit[i] = acq
        trial_idx += 1
        if trial_idx >= budget:
            return pd.DataFrame(rows)

    best_idx = int(np.argmax(fit))
    best_pos = pop[best_idx].copy()
    best_fit = float(fit[best_idx])

    stagnation = 0
    qrg_start_iter = int(0.3 * max_iters)

    for it in range(max_iters):
        old_best = best_fit
        shrink = 2.0 * (1.0 - it / max(max_iters, 1))
        worst_fit = float(np.min(fit))
        new_pop = pop.copy()

        for i in range(pop_size):
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
                    if rng.random() > E:
                        cand = W1 * best_pos + R * W2 * diff
                    else:
                        cand = W1 * best_pos - R * W2 * diff

            cand = clip_vec(cand)
            new_pop[i] = cand

        for i in range(pop_size):
            acq, mean, std = obj.eval_vec(new_pop[i])
            p = vec_to_params(new_pop[i])
            rows.append({
                "method": "qrghgs",
                "trial_idx": trial_idx,
                **p,
                "pred_mean": mean,
                "pred_std": std,
                "acq": acq,
            })

            if acq > fit[i]:
                fit[i] = acq
                pop[i] = new_pop[i].copy()

            trial_idx += 1
            if trial_idx >= budget:
                return pd.DataFrame(rows)

        best_idx = int(np.argmax(fit))
        best_pos = pop[best_idx].copy()
        best_fit = float(fit[best_idx])

        if best_fit > old_best + 1e-12:
            stagnation = 0
        else:
            stagnation += 1

        if it >= qrg_start_iter and stagnation >= 2:
            theta = 0.3141592654 - (0.3141592654 - 0.0523598776) * (it / max(max_iters - 1, 1))
            elite_ids = np.argsort(-fit)[:max(1, math.ceil(0.2 * pop_size))]

            for idx in elite_ids:
                Xi = pop[idx].copy()
                cand = Xi + theta * (best_pos - Xi) + 0.005 * (ub - lb) * rng.normal(size=3)
                cand = clip_vec(cand)

                acq, mean, std = obj.eval_vec(cand)
                p = vec_to_params(cand)
                rows.append({
                    "method": "qrghgs",
                    "trial_idx": trial_idx,
                    **p,
                    "pred_mean": mean,
                    "pred_std": std,
                    "acq": acq,
                })

                if acq > fit[idx]:
                    fit[idx] = acq
                    pop[idx] = cand.copy()

                trial_idx += 1
                if trial_idx >= budget:
                    return pd.DataFrame(rows)

            stagnation = 0
            best_idx = int(np.argmax(fit))
            best_pos = pop[best_idx].copy()
            best_fit = float(fit[best_idx])

    return pd.DataFrame(rows)


# =========================================================
# 9. 主流程
# =========================================================
def main():
    history = load_history_from_csv(HISTORY_CSV)
    history = history[history["status"] == "OK"].copy().reset_index(drop=True)

    # 不再固定 batch=4
    history = history.dropna(subset=["lr", "dice_weight", "batch_size", "best_val_dice"]).reset_index(drop=True)

    surrogate = RFEnsembleSurrogate(n_models=8, random_seed=SEED)
    surrogate.fit(history)

    obj = SurrogateObjective(surrogate, beta=BETA)

    all_df = []

    for method in METHODS:
        print(f"\n========== RUN {method.upper()} IN SURROGATE SPACE ==========")

        if method == "random":
            df = run_random_surrogate(obj, SURROGATE_BUDGET, SEED)
        elif method == "tpe":
            df = run_tpe_surrogate(obj, SURROGATE_BUDGET, SEED)
        elif method == "pso":
            df = run_pso_surrogate(obj, SURROGATE_BUDGET, SEED)
        elif method == "ga":
            df = run_ga_surrogate(obj, SURROGATE_BUDGET, SEED)
        elif method == "hgs":
            df = run_hgs_surrogate(obj, SURROGATE_BUDGET, SEED)
        elif method == "qrghgs":
            df = run_qrghgs_surrogate(obj, SURROGATE_BUDGET, SEED)
        else:
            continue

        if len(df) == 0:
            continue

        df.to_csv(OUT_ROOT / f"{method}_surrogate_search.csv", index=False, encoding="utf-8-sig")

        # 这里不要只看 acq，也可以看 pred_mean 排序
        topk = df.sort_values(["acq", "pred_mean"], ascending=False).head(TOPK_REAL_EVAL).copy()
        topk.to_csv(OUT_ROOT / f"{method}_topk_candidates.csv", index=False, encoding="utf-8-sig")

        real_rows = []
        for i, (_, row) in enumerate(topk.iterrows()):
            real_rows.append(real_evaluate(method, i, row))

        real_df = pd.DataFrame(real_rows)
        real_df.to_csv(OUT_ROOT / f"{method}_real_eval.csv", index=False, encoding="utf-8-sig")

        print(f"\n[{method}] surrogate top-k:")
        print(topk[["lr", "dice_weight", "batch_size", "pred_mean", "pred_std", "acq"]])

        print(f"\n[{method}] real eval:")
        print(real_df)

        real_df["method"] = method
        all_df.append(real_df)

    if len(all_df) > 0:
        merged = pd.concat(all_df, ignore_index=True)
        merged.to_csv(OUT_ROOT / "all_methods_real_eval.csv", index=False, encoding="utf-8-sig")
        print("\n===== FINAL REAL EVAL SUMMARY =====")
        print(merged.groupby("method")[["best_val_dice", "test_dice", "time_sec"]].mean())


if __name__ == "__main__":
    main()
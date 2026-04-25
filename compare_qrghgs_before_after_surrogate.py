import math
import sys
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from surrogate_model_checked import RFEnsembleSurrogate, load_history_from_csv


# =========================================================
# 0. 基础配置
# =========================================================
HISTORY_CSV = "outputs/history_augmented/all_methods_merged_plus_warmup.csv"
OUT_ROOT = Path("outputs/qrghgs_before_after")
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

SURROGATE_BUDGET = 40
TOPK_REAL_EVAL = 3
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


def real_evaluate(tag, idx, row):
    run_name = f"{tag}_trial{idx:03d}"
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
            "variant": tag,
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
        "variant": tag,
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
# 2. 代理目标
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
# 3. QRG-HGS before / after
# =========================================================
def run_qrghgs_surrogate(obj, budget, seed, variant_name, cfg):
    rng = np.random.default_rng(seed)
    max_iters = max(1, math.ceil(budget / POP_SIZE))

    lb = np.array([SPACE["lr"][0], SPACE["dice_weight"][0], 0], dtype=np.float64)
    ub = np.array([SPACE["lr"][1], SPACE["dice_weight"][1], len(BATCH_CHOICES) - 1], dtype=np.float64)

    pop = lb + (ub - lb) * rng.random((POP_SIZE, 3))
    fit = np.full(POP_SIZE, -1e18)

    rows = []
    trial_idx = 0

    # 初始评估
    for i in range(POP_SIZE):
        acq, mean, std = obj.eval_vec(pop[i])
        p = vec_to_params(pop[i])
        rows.append({
            "variant": variant_name,
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
    qrg_start_iter = int(cfg["qrg_start_ratio"] * max_iters)

    for it in range(max_iters):
        old_best = best_fit
        shrink = 2.0 * (1.0 - it / max(max_iters, 1))
        worst_fit = float(np.min(fit))
        new_pop = pop.copy()

        # -------- HGS主体更新 --------
        for i in range(POP_SIZE):
            Xi = pop[i].copy()

            if i == best_idx:
                cand = Xi + cfg["base_noise_ratio"] * (ub - lb) * rng.normal(size=3)
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

        # 评估 HGS 更新后种群
        for i in range(POP_SIZE):
            acq, mean, std = obj.eval_vec(new_pop[i])
            p = vec_to_params(new_pop[i])
            rows.append({
                "variant": variant_name,
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

        # -------- 提前启动的 QRG + 自适应停滞扰动 --------
        if it >= qrg_start_iter and stagnation >= cfg["stagnation_patience"]:
            theta = cfg["theta_init"] - (cfg["theta_init"] - cfg["theta_final"]) * (
                it / max(max_iters - 1, 1)
            )

            elite_k = max(1, math.ceil(cfg["elite_keep_ratio"] * POP_SIZE))
            elite_ids = np.argsort(-fit)[:elite_k]

            for idx in elite_ids:
                Xi = pop[idx].copy()

                # 更积极的量子扰动 + 局部探索
                direction = best_pos - Xi
                qrg_move = theta * direction
                local_noise = cfg["qrg_noise_ratio"] * (ub - lb) * rng.normal(size=3)

                cand = Xi + qrg_move + local_noise
                cand = clip_vec(cand)

                acq, mean, std = obj.eval_vec(cand)
                p = vec_to_params(cand)
                rows.append({
                    "variant": variant_name,
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

            # 对最差点做一次更强的自适应扰动
            worst_ids = np.argsort(fit)[:max(1, math.ceil(0.2 * POP_SIZE))]
            sigma = cfg["restart_noise_ratio"] * (1.0 - it / max(max_iters, 1))
            for idx in worst_ids:
                cand = best_pos + sigma * (ub - lb) * rng.normal(size=3)
                cand = clip_vec(cand)

                acq, mean, std = obj.eval_vec(cand)
                p = vec_to_params(cand)
                rows.append({
                    "variant": variant_name,
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
# 4. 主流程
# =========================================================
def main():
    # 读取增强后的历史，允许 batch=2/4/8
    history = load_history_from_csv(HISTORY_CSV)
    history = history[history["status"] == "OK"].copy().reset_index(drop=True)
    history = history.dropna(subset=["lr", "dice_weight", "batch_size", "best_val_dice"]).reset_index(drop=True)

    surrogate = RFEnsembleSurrogate(n_models=8, random_seed=SEED)
    surrogate.fit(history)

    obj = SurrogateObjective(surrogate, beta=BETA)

    configs = {
        "before": {
            "qrg_start_ratio": 0.30,
            "stagnation_patience": 2,
            "elite_keep_ratio": 0.20,
            "theta_init": 0.3141592654,
            "theta_final": 0.0523598776,
            "base_noise_ratio": 0.010,
            "qrg_noise_ratio": 0.005,
            "restart_noise_ratio": 0.080,
        },
        "after": {
            "qrg_start_ratio": 0.15,
            "stagnation_patience": 1,
            "elite_keep_ratio": 0.30,
            "theta_init": 0.3141592654,
            "theta_final": 0.0523598776,
            "base_noise_ratio": 0.015,
            "qrg_noise_ratio": 0.010,
            "restart_noise_ratio": 0.120,
        }
    }

    all_search = []
    all_real = []

    for variant_name, cfg in configs.items():
        print(f"\n========== RUN QRGHGS-{variant_name.upper()} IN SURROGATE SPACE ==========")

        df = run_qrghgs_surrogate(
            obj=obj,
            budget=SURROGATE_BUDGET,
            seed=SEED,
            variant_name=variant_name,
            cfg=cfg,
        )

        df.to_csv(OUT_ROOT / f"qrghgs_{variant_name}_surrogate_search.csv", index=False, encoding="utf-8-sig")

        topk = df.sort_values(["acq", "pred_mean"], ascending=False).head(TOPK_REAL_EVAL).copy()
        topk.to_csv(OUT_ROOT / f"qrghgs_{variant_name}_topk_candidates.csv", index=False, encoding="utf-8-sig")

        print(f"\n[{variant_name}] surrogate top-k:")
        print(topk[["lr", "dice_weight", "batch_size", "pred_mean", "pred_std", "acq"]])

        real_rows = []
        for i, (_, row) in enumerate(topk.iterrows()):
            real_rows.append(real_evaluate(f"qrghgs_{variant_name}", i, row))

        real_df = pd.DataFrame(real_rows)
        real_df.to_csv(OUT_ROOT / f"qrghgs_{variant_name}_real_eval.csv", index=False, encoding="utf-8-sig")

        print(f"\n[{variant_name}] real eval:")
        print(real_df)

        df["variant"] = variant_name
        all_search.append(df)

        real_df["variant"] = variant_name
        all_real.append(real_df)

    # 合并保存
    search_merged = pd.concat(all_search, ignore_index=True)
    real_merged = pd.concat(all_real, ignore_index=True)

    search_merged.to_csv(OUT_ROOT / "qrghgs_before_after_surrogate_all.csv", index=False, encoding="utf-8-sig")
    real_merged.to_csv(OUT_ROOT / "qrghgs_before_after_real_eval_all.csv", index=False, encoding="utf-8-sig")

    print("\n===== REAL EVAL SUMMARY =====")
    summary = real_merged.groupby("variant")[["best_val_dice", "test_dice", "time_sec"]].agg(["mean", "max", "std"])
    print(summary)

    summary.to_csv(OUT_ROOT / "qrghgs_before_after_summary.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
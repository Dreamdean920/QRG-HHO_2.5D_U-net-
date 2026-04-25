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
OUT_ROOT = Path("outputs/qrghgs_10_versions")
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

SURROGATE_BUDGET = 50
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


def real_evaluate(tag, idx, row):
    run_name = f"{tag}_trial{idx:03d}"
    output_json = OUT_ROOT / f"{run_name}.json"

    # ===== 断点续跑：已有成功 json 就直接复用 =====
    old = load_json_if_ok(output_json)
    if old is not None:
        print(f"[SKIP REAL] {run_name}")
        return {
            "variant": tag,
            "run_name": run_name,
            "lr": row["lr"],
            "dice_weight": row["dice_weight"],
            "batch_size": row["batch_size"],
            "status": old.get("status", "OK"),
            "best_val_dice": old.get("best_val_dice"),
            "test_dice": old.get("test_dice"),
            "time_sec": old.get("time_sec"),
        }

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


def save_summary_from_real_eval(all_real_path: Path, summary_path: Path):
    if not all_real_path.exists():
        return
    real_merged = pd.read_csv(all_real_path)
    if len(real_merged) == 0:
        return
    summary = real_merged.groupby("variant")[["best_val_dice", "test_dice", "time_sec"]].agg(["mean", "max", "std"])
    summary.to_csv(summary_path, encoding="utf-8-sig")
    print("\n===== REAL EVAL SUMMARY =====")
    print(summary)


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
# 3. 扰动调度
# =========================================================
def get_noise_values(cfg, progress):
    if not cfg["use_perturb"]:
        return 0.0, 0.0, 0.0

    mode = cfg.get("noise_mode", "fixed")

    if mode == "fixed":
        return (
            cfg["base_noise_fixed"],
            cfg["qrg_noise_fixed"],
            cfg["restart_noise_fixed"],
        )

    gamma = cfg.get("noise_gamma", 1.5)
    alpha = progress ** gamma

    base_noise = cfg["base_noise_min"] + (cfg["base_noise_max"] - cfg["base_noise_min"]) * alpha
    qrg_noise = cfg["qrg_noise_min"] + (cfg["qrg_noise_max"] - cfg["qrg_noise_min"]) * alpha
    restart_noise = cfg["restart_noise_min"] + (cfg["restart_noise_max"] - cfg["restart_noise_min"]) * alpha

    return base_noise, qrg_noise, restart_noise


# =========================================================
# 4. 通用 QRG-HGS 版本
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
                    if rng.random() > E:
                        cand = W1 * best_pos + R * W2 * diff
                    else:
                        cand = W1 * best_pos - R * W2 * diff

            cand = clip_vec(cand)
            new_pop[i] = cand

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

        if it >= qrg_start_iter and stagnation >= cfg["stagnation_patience"]:
            theta = cfg["theta_init"] - (cfg["theta_init"] - cfg["theta_final"]) * progress

            elite_k = max(1, math.ceil(cfg["elite_keep_ratio"] * POP_SIZE))
            elite_ids = np.argsort(-fit)[:elite_k]

            for idx in elite_ids:
                Xi = pop[idx].copy()
                direction = best_pos - Xi
                qrg_move = theta * direction
                local_noise = qrg_noise_ratio * (ub - lb) * rng.normal(size=3)

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

            if cfg["use_perturb"] and restart_noise_ratio > 0:
                worst_ids = np.argsort(fit)[:max(1, math.ceil(cfg["restart_ratio"] * POP_SIZE))]
                for idx in worst_ids:
                    cand = best_pos + restart_noise_ratio * (ub - lb) * rng.normal(size=3)
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
# 5. 版本配置
# =========================================================
def build_configs():
    base_versions = {
        "before": {
            "qrg_start_ratio": 0.30,
            "stagnation_patience": 2,
            "elite_keep_ratio": 0.20,
            "restart_ratio": 0.20,
        },
        "mid_v1": {
            "qrg_start_ratio": 0.25,
            "stagnation_patience": 1,
            "elite_keep_ratio": 0.25,
            "restart_ratio": 0.20,
        },
        "mid_v2": {
            "qrg_start_ratio": 0.20,
            "stagnation_patience": 1,
            "elite_keep_ratio": 0.25,
            "restart_ratio": 0.20,
        },
        "mid_v3": {
            "qrg_start_ratio": 0.18,
            "stagnation_patience": 1,
            "elite_keep_ratio": 0.30,
            "restart_ratio": 0.25,
        },
        "after": {
            "qrg_start_ratio": 0.15,
            "stagnation_patience": 1,
            "elite_keep_ratio": 0.30,
            "restart_ratio": 0.25,
        },
    }

    common = {
        "theta_init": 0.3141592654,
        "theta_final": 0.0523598776,
    }

    configs = {}

    for name, cfg in base_versions.items():
        configs[f"{name}_no_perturb"] = {
            **common,
            **cfg,
            "use_perturb": False,
            "noise_mode": "fixed",
            "base_noise_fixed": 0.0,
            "qrg_noise_fixed": 0.0,
            "restart_noise_fixed": 0.0,
        }

    for name, cfg in base_versions.items():
        if name == "before":
            noise_cfg = {
                "use_perturb": True,
                "noise_mode": "fixed",
                "base_noise_fixed": 0.010,
                "qrg_noise_fixed": 0.005,
                "restart_noise_fixed": 0.080,
            }
        elif name == "mid_v1":
            noise_cfg = {
                "use_perturb": True,
                "noise_mode": "increasing",
                "noise_gamma": 1.2,
                "base_noise_min": 0.008,
                "base_noise_max": 0.012,
                "qrg_noise_min": 0.003,
                "qrg_noise_max": 0.008,
                "restart_noise_min": 0.050,
                "restart_noise_max": 0.090,
            }
        elif name == "mid_v2":
            noise_cfg = {
                "use_perturb": True,
                "noise_mode": "increasing",
                "noise_gamma": 1.5,
                "base_noise_min": 0.008,
                "base_noise_max": 0.014,
                "qrg_noise_min": 0.003,
                "qrg_noise_max": 0.010,
                "restart_noise_min": 0.050,
                "restart_noise_max": 0.110,
            }
        elif name == "mid_v3":
            noise_cfg = {
                "use_perturb": True,
                "noise_mode": "increasing",
                "noise_gamma": 1.8,
                "base_noise_min": 0.010,
                "base_noise_max": 0.015,
                "qrg_noise_min": 0.004,
                "qrg_noise_max": 0.012,
                "restart_noise_min": 0.060,
                "restart_noise_max": 0.120,
            }
        else:
            noise_cfg = {
                "use_perturb": True,
                "noise_mode": "fixed",
                "base_noise_fixed": 0.015,
                "qrg_noise_fixed": 0.010,
                "restart_noise_fixed": 0.120,
            }

        configs[f"{name}_perturb"] = {
            **common,
            **cfg,
            **noise_cfg,
        }

    return configs


# =========================================================
# 6. 主流程（支持断点续跑）
# =========================================================
def main():
    history = load_history_from_csv(HISTORY_CSV)
    history = history[history["status"] == "OK"].copy().reset_index(drop=True)
    history = history.dropna(subset=["lr", "dice_weight", "batch_size", "best_val_dice"]).reset_index(drop=True)

    surrogate = RFEnsembleSurrogate(n_models=8, random_seed=SEED)
    surrogate.fit(history)

    obj = SurrogateObjective(surrogate, beta=BETA)
    configs = build_configs()

    all_search_path = OUT_ROOT / "all_surrogate_search.csv"
    all_real_path = OUT_ROOT / "all_real_eval.csv"
    summary_path = OUT_ROOT / "summary.csv"

    if all_search_path.exists():
        all_search = pd.read_csv(all_search_path)
    else:
        all_search = pd.DataFrame()

    if all_real_path.exists():
        all_real = pd.read_csv(all_real_path)
    else:
        all_real = pd.DataFrame()

    for variant_name, cfg in configs.items():
        print(f"\n========== RUN {variant_name.upper()} ==========")

        search_csv = OUT_ROOT / f"{variant_name}_surrogate_search.csv"
        topk_csv = OUT_ROOT / f"{variant_name}_topk_candidates.csv"
        real_csv = OUT_ROOT / f"{variant_name}_real_eval.csv"

        # ===== 1) 代理搜索：已有则跳过 =====
        if search_csv.exists():
            df = pd.read_csv(search_csv)
            print(f"[SKIP SEARCH] 已存在 {search_csv.name}")
        else:
            df = run_qrghgs_surrogate(
                obj=obj,
                budget=SURROGATE_BUDGET,
                seed=SEED,
                variant_name=variant_name,
                cfg=cfg,
            )
            df.to_csv(search_csv, index=False, encoding="utf-8-sig")

        # ===== 2) topk：已有则跳过 =====
        if topk_csv.exists():
            topk = pd.read_csv(topk_csv)
            print(f"[SKIP TOPK] 已存在 {topk_csv.name}")
        else:
            topk = df.sort_values(["acq", "pred_mean"], ascending=False).head(TOPK_REAL_EVAL).copy()
            topk.to_csv(topk_csv, index=False, encoding="utf-8-sig")

        print(f"\n[{variant_name}] surrogate top-k:")
        print(topk[["lr", "dice_weight", "batch_size", "pred_mean", "pred_std", "acq"]])

        # ===== 3) 真实评估：支持部分续跑 =====
        if real_csv.exists():
            real_df = pd.read_csv(real_csv)
        else:
            real_df = pd.DataFrame()

        done_run_names = set(real_df["run_name"].tolist()) if len(real_df) > 0 and "run_name" in real_df.columns else set()

        new_rows = []
        for i, (_, row) in enumerate(topk.iterrows()):
            run_name = f"{variant_name}_trial{i:03d}"
            if run_name in done_run_names:
                print(f"[SKIP REAL ROW] {run_name}")
                continue
            new_rows.append(real_evaluate(variant_name, i, row))

            # 每完成一个就立刻保存，防止再次 Ctrl+C 丢失
            tmp_df = pd.concat([real_df, pd.DataFrame(new_rows)], ignore_index=True)
            tmp_df.to_csv(real_csv, index=False, encoding="utf-8-sig")

        if len(new_rows) > 0:
            real_df = pd.concat([real_df, pd.DataFrame(new_rows)], ignore_index=True)

        print(f"\n[{variant_name}] real eval:")
        print(real_df)

        # ===== 4) 更新总表（去重覆盖） =====
        df["variant"] = variant_name
        if len(all_search) > 0:
            all_search = all_search[all_search["variant"] != variant_name]
        all_search = pd.concat([all_search, df], ignore_index=True)
        all_search.to_csv(all_search_path, index=False, encoding="utf-8-sig")

        real_df["variant"] = variant_name
        if len(all_real) > 0:
            all_real = all_real[all_real["variant"] != variant_name]
        all_real = pd.concat([all_real, real_df], ignore_index=True)
        all_real.to_csv(all_real_path, index=False, encoding="utf-8-sig")

        # 每轮都更新 summary
        save_summary_from_real_eval(all_real_path, summary_path)

    print("\n✅ 全部版本处理完成。")


if __name__ == "__main__":
    main()
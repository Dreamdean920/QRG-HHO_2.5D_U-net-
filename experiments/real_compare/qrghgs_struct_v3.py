import math
import sys
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 0. 基础配置：同目录续跑
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

COMMON_ARGS = [
    "--input_mode", "2.5d",
    "--no_post",
    "--save_root", str(TRAIN_SAVE_ROOT),
    "--epochs", "20",
]

SPACE = {
    "lr": (0.0005, 0.0030),
    "dice_weight": (0.50, 0.85),
    "batch_size": [2, 4, 8],
}
BATCH_CHOICES = SPACE["batch_size"]

SEED = 42
SEARCH_BUDGET = 30
POP_SIZE = 6

# 只跑原始 V2 + 原始 V3
METHODS = [
    "qrghgs_struct_v3",
]


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

    # 断点续跑：如果 json 已经存在且 OK，直接复用
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

    if proc.returncode != 0 or not output_json.exists():
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
            "error_message": "subprocess_failed_or_missing_json",
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


def row_to_fit(row):
    if row["status"] != "OK" or row["best_val_dice"] is None:
        return -1e18
    try:
        return float(row["best_val_dice"])
    except Exception:
        return -1e18


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
# 2. 原始 V2 / V3 配置
# =========================================================
def get_qrghgs_config(method):
    if method == "qrghgs_struct_v2":
        return {
            "qrg_start_ratio": 0.30,
            "stagnation_patience": 2,
            "elite_keep_ratio": 0.20,
            "theta_init": 0.3141592654,
            "theta_final": 0.0523598776,

            # V2：更强探索
            "local_ratio": 0.50,
            "topk_ratio": 0.25,
            "global_restart_ratio": 0.35,

            "local_noise_min": 0.04,
            "local_noise_max": 0.08,
            "topk_noise_min": 0.08,
            "topk_noise_max": 0.16,

            "random_mix_ratio": 0.50,
        }

    if method == "qrghgs_struct_v3":
        return {
            "qrg_start_ratio": 0.30,
            "stagnation_patience": 2,
            "elite_keep_ratio": 0.20,
            "theta_init": 0.3141592654,
            "theta_final": 0.0523598776,

            # V3：降低扰动强度，偏泛化稳定
            "local_ratio": 0.50,
            "topk_ratio": 0.25,
            "global_restart_ratio": 0.25,

            "local_noise_min": 0.03,
            "local_noise_max": 0.06,
            "topk_noise_min": 0.06,
            "topk_noise_max": 0.12,

            "random_mix_ratio": 0.40,
        }

    raise ValueError(f"Unknown method: {method}")


# =========================================================
# 3. 结构化扰动
# =========================================================
def structured_perturbation(
    rows,
    method_name,
    cfg,
    rng,
    pop,
    fit,
    best_pos,
    lb,
    ub,
    trial_idx,
    progress,
):
    pop_size = len(pop)

    local_noise = cfg["local_noise_min"] + (
        cfg["local_noise_max"] - cfg["local_noise_min"]
    ) * progress

    topk_noise = cfg["topk_noise_min"] + (
        cfg["topk_noise_max"] - cfg["topk_noise_min"]
    ) * progress

    # 1) Local exploitation
    local_k = max(1, math.ceil(cfg["local_ratio"] * pop_size))
    elite_ids = np.argsort(-fit)[:local_k]

    for idx in elite_ids:
        Xi = pop[idx].copy()
        direction = best_pos - Xi

        cand = Xi + 0.25 * direction + local_noise * (ub - lb) * rng.normal(size=3)
        cand = clip_vec(cand)

        params = vec_to_params(cand)
        row = real_evaluate(method_name, trial_idx, params)
        rows.append(row)
        save_results(rows)

        new_fit = row_to_fit(row)
        if new_fit > fit[idx]:
            fit[idx] = new_fit
            pop[idx] = cand.copy()

        trial_idx += 1
        if trial_idx >= SEARCH_BUDGET:
            return rows, pop, fit, trial_idx, True

    # 2) Top-k migration + random mix
    topk_k = max(1, math.ceil(cfg["topk_ratio"] * pop_size))
    top_ids = list(np.argsort(-fit)[:max(2, topk_k)])

    random_k = max(1, math.ceil(cfg["random_mix_ratio"] * topk_k))
    all_ids = list(range(pop_size))
    random_ids = list(rng.choice(all_ids, size=random_k, replace=True))

    mix_ids = top_ids + random_ids

    for _ in range(topk_k):
        center_idx = int(rng.choice(mix_ids))
        center = pop[center_idx].copy()

        cand = center + topk_noise * (ub - lb) * rng.normal(size=3)
        cand = clip_vec(cand)

        worst_idx = int(np.argmin(fit))

        params = vec_to_params(cand)
        row = real_evaluate(method_name, trial_idx, params)
        rows.append(row)
        save_results(rows)

        new_fit = row_to_fit(row)
        if new_fit > fit[worst_idx]:
            fit[worst_idx] = new_fit
            pop[worst_idx] = cand.copy()

        trial_idx += 1
        if trial_idx >= SEARCH_BUDGET:
            return rows, pop, fit, trial_idx, True

    # 3) Global restart
    global_k = max(1, math.ceil(cfg["global_restart_ratio"] * pop_size))
    worst_ids = np.argsort(fit)[:global_k]

    for idx in worst_ids:
        cand = lb + (ub - lb) * rng.random(3)
        cand = clip_vec(cand)

        params = vec_to_params(cand)
        row = real_evaluate(method_name, trial_idx, params)
        rows.append(row)
        save_results(rows)

        new_fit = row_to_fit(row)
        if new_fit > fit[idx]:
            fit[idx] = new_fit
            pop[idx] = cand.copy()

        trial_idx += 1
        if trial_idx >= SEARCH_BUDGET:
            return rows, pop, fit, trial_idx, True

    return rows, pop, fit, trial_idx, False


# =========================================================
# 4. QRG-HGS 主体
# =========================================================
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

        fit[i] = row_to_fit(row)

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

        # HGS 主体更新
        for i in range(POP_SIZE):
            Xi = pop[i].copy()

            if i == best_idx:
                cand = Xi
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

            new_pop[i] = clip_vec(cand)

        # 评估 HGS 更新
        for i in range(POP_SIZE):
            params = vec_to_params(new_pop[i])
            row = real_evaluate(method_name, trial_idx, params)
            rows.append(row)
            save_results(rows)

            new_fit = row_to_fit(row)
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

        # QRG + 结构化扰动
        if it >= qrg_start_iter and stagnation >= cfg["stagnation_patience"]:
            theta = cfg["theta_init"] - (cfg["theta_init"] - cfg["theta_final"]) * progress

            elite_k = max(1, math.ceil(cfg["elite_keep_ratio"] * POP_SIZE))
            elite_ids = np.argsort(-fit)[:elite_k]

            # QRG 局部开发
            for idx in elite_ids:
                Xi = pop[idx].copy()
                direction = best_pos - Xi
                cand = Xi + theta * direction
                cand = clip_vec(cand)

                params = vec_to_params(cand)
                row = real_evaluate(method_name, trial_idx, params)
                rows.append(row)
                save_results(rows)

                new_fit = row_to_fit(row)
                if new_fit > fit[idx]:
                    fit[idx] = new_fit
                    pop[idx] = cand.copy()

                trial_idx += 1
                if trial_idx >= SEARCH_BUDGET:
                    return rows

            rows, pop, fit, trial_idx, done = structured_perturbation(
                rows=rows,
                method_name=method_name,
                cfg=cfg,
                rng=rng,
                pop=pop,
                fit=fit,
                best_pos=best_pos,
                lb=lb,
                ub=ub,
                trial_idx=trial_idx,
                progress=progress,
            )

            if done:
                return rows

            stagnation = 0
            best_idx = int(np.argmax(fit))
            best_pos = pop[best_idx].copy()
            best_fit = float(fit[best_idx])

    return rows


# =========================================================
# 5. 主入口：同目录续跑
# =========================================================
if __name__ == "__main__":
    rows = []

    if RAW_CSV.exists():
        try:
            old_df = pd.read_csv(RAW_CSV)
            rows = old_df.to_dict("records")
            print(f"[INFO] 已读取历史记录 {len(rows)} 条，将在同目录继续续跑。")
        except Exception:
            rows = []

    for method in METHODS:
        print(f"\n\n########## RUN METHOD: {method} ##########")
        rows = run_qrghgs(rows, method)

        _, summary_df = save_results(rows)
        print("\n===== CURRENT SUMMARY =====")
        print(summary_df)

    print("\n✅ 原始 V2 + 原始 V3 已跑到 SEARCH_BUDGET = 30。")
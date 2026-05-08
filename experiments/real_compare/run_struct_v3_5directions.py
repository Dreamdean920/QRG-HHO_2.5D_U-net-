import math
import sys
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 0. 全局配置：统一写入原目录
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

# =========================================================
# 默认跑 4 个 Struct_v3 改进搜索版本
# =========================================================
RUN_SEARCH = True

METHODS = [
    "qrghgs_struct_v3_elite_exploit",
    "qrghgs_struct_v3_perf_trigger",
    "qrghgs_struct_v3_elite_region",
    "qrghgs_struct_v3_dual_branch",
]

# =========================================================
# 第 5 个方向：Top3 多 seed confirm
# 默认关闭，等搜索跑完后再打开
# =========================================================
RUN_CONFIRM = False

CONFIRM_METHODS = [
    "qrghgs_struct_v3",
    "qrghgs_struct_v3_elite_exploit",
    "qrghgs_struct_v3_perf_trigger",
    "qrghgs_struct_v3_elite_region",
    "qrghgs_struct_v3_dual_branch",
    "qrghgs_hybrid",
    "ga",
]

TOP_K = 3
CONFIRM_SEEDS = [42, 52, 62, 72, 82]
CONFIRM_EPOCHS = 20


# =========================================================
# 1. 基础工具函数
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
    return np.array(
        [
            float(params["lr"]),
            float(params["dice_weight"]),
            float(batch_idx),
        ],
        dtype=np.float64,
    )


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


def row_to_fit(row):
    """
    搜索阶段保持公平口径：
    只用 best_val_dice 做 fitness。
    不使用 test_dice。
    """
    if row.get("status") != "OK" or row.get("best_val_dice") is None:
        return -1e18

    try:
        return float(row["best_val_dice"])
    except Exception:
        return -1e18


def load_existing_rows_clean():
    if not RAW_CSV.exists():
        return []

    try:
        df = pd.read_csv(RAW_CSV)
    except Exception:
        return []

    if len(df) == 0:
        return []

    if "resumed" not in df.columns:
        df["resumed"] = False

    df = (
        df.sort_values(["method", "trial_idx", "resumed"])
        .drop_duplicates(subset=["method", "run_name"], keep="last")
        .reset_index(drop=True)
    )

    return df.to_dict("records")


def save_results(rows):
    df = pd.DataFrame(rows)

    if len(df) == 0:
        df.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
        return df, pd.DataFrame()

    if "resumed" not in df.columns:
        df["resumed"] = False

    df_clean = (
        df.sort_values(["method", "trial_idx", "resumed"])
        .drop_duplicates(subset=["method", "run_name"], keep="last")
        .reset_index(drop=True)
    )

    df_clean.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")

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
# 2. 从历史 Struct_v3 中读取高潜力中心
# =========================================================
def load_historical_elite_centers(top_k=4):
    """
    从已有 compare_raw.csv 中读取 qrghgs_struct_v3 的 best_val top-k 参数作为高潜力区域中心。
    注意：只按 best_val_dice 选，不按 test_dice 选，避免测试集泄漏。
    """
    fallback = [
        np.array([0.002066, 0.500, 2.0], dtype=np.float64),
        np.array([0.002243, 0.533, 2.0], dtype=np.float64),
        np.array([0.002527, 0.531, 2.0], dtype=np.float64),
        np.array([0.002567, 0.732, 2.0], dtype=np.float64),
    ]

    if not RAW_CSV.exists():
        return fallback

    try:
        df = pd.read_csv(RAW_CSV)
    except Exception:
        return fallback

    if len(df) == 0 or "method" not in df.columns:
        return fallback

    sub = df[(df["method"] == "qrghgs_struct_v3") & (df["status"] == "OK")].copy()

    if len(sub) == 0:
        return fallback

    for col in ["best_val_dice", "lr", "dice_weight", "batch_size"]:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")

    sub = sub.dropna(subset=["best_val_dice", "lr", "dice_weight", "batch_size"])

    if len(sub) == 0:
        return fallback

    sub = sub.sort_values("best_val_dice", ascending=False).head(top_k)

    centers = []
    for _, r in sub.iterrows():
        try:
            batch_size = int(r["batch_size"])
            if batch_size not in BATCH_CHOICES:
                continue
            batch_idx = BATCH_CHOICES.index(batch_size)

            centers.append(
                clip_vec(
                    np.array(
                        [float(r["lr"]), float(r["dice_weight"]), float(batch_idx)],
                        dtype=np.float64,
                    )
                )
            )
        except Exception:
            continue

    if len(centers) == 0:
        return fallback

    return centers


def update_elite_archive(row, pos, elite_archive, archive_size=6):
    fit = row_to_fit(row)

    if fit <= -1e17:
        return elite_archive

    item = {
        "pos": pos.copy(),
        "fit": fit,
        "best_val_dice": row.get("best_val_dice"),
        "test_dice": row.get("test_dice"),
        "lr": row.get("lr"),
        "dice_weight": row.get("dice_weight"),
        "batch_size": row.get("batch_size"),
    }

    elite_archive.append(item)
    elite_archive = sorted(elite_archive, key=lambda z: z["fit"], reverse=True)
    elite_archive = elite_archive[:archive_size]

    return elite_archive


def get_all_centers(historical_centers, elite_archive):
    centers = []

    for item in elite_archive:
        centers.append(item["pos"].copy())

    for c in historical_centers:
        centers.append(c.copy())

    if len(centers) == 0:
        centers = load_historical_elite_centers(top_k=4)

    return centers


def sample_elite_region(rng, historical_centers, elite_archive, small=True):
    """
    围绕高潜力中心采样。
    small=True：小范围精修
    small=False：中等范围回访
    """
    centers = get_all_centers(historical_centers, elite_archive)
    center = centers[int(rng.integers(0, len(centers)))].copy()

    if small:
        lr_noise = rng.normal(0, 0.00018)
        dice_noise = rng.normal(0, 0.025)
        batch_noise = rng.normal(0, 0.08)
    else:
        lr_noise = rng.normal(0, 0.00028)
        dice_noise = rng.normal(0, 0.040)
        batch_noise = rng.normal(0, 0.15)

    cand = center.copy()
    cand[0] += lr_noise
    cand[1] += dice_noise
    cand[2] += batch_noise

    return clip_vec(cand)


def sample_dual_branch(rng):
    """
    双分支高潜力区域：
    1. low-dice branch：batch=8, lr≈0.0020~0.0026, dice≈0.50~0.56
    2. high-dice branch：batch=8, lr≈0.0023~0.0027, dice≈0.70~0.78
    3. small stable branch：batch=4, lr≈0.0012~0.0018, dice≈0.62~0.72
    """
    branch = rng.choice(["low", "high", "stable"], p=[0.50, 0.30, 0.20])

    if branch == "low":
        lr = rng.uniform(0.0020, 0.0026)
        dice = rng.uniform(0.50, 0.56)
        batch_idx = 2

    elif branch == "high":
        lr = rng.uniform(0.0023, 0.0027)
        dice = rng.uniform(0.70, 0.78)
        batch_idx = 2

    else:
        lr = rng.uniform(0.0012, 0.0018)
        dice = rng.uniform(0.62, 0.72)
        batch_idx = 1

    return clip_vec(np.array([lr, dice, batch_idx], dtype=np.float64))


# =========================================================
# 3. 真实训练评估
# =========================================================
def real_evaluate(method, trial_idx, params):
    run_name = make_run_name(method, trial_idx)
    output_json = TRIAL_JSON_ROOT / f"{run_name}.json"

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


# =========================================================
# 4. 四个 Struct_v3 改进版本配置
# =========================================================
def get_config(method):
    base = {
        "qrg_start_ratio": 0.30,
        "stagnation_patience": 2,
        "elite_keep_ratio": 0.20,
        "theta_init": 0.3141592654,
        "theta_final": 0.0523598776,

        # 原始 Struct_v3
        "local_ratio": 0.50,
        "topk_ratio": 0.25,
        "global_restart_ratio": 0.25,

        "local_noise_min": 0.03,
        "local_noise_max": 0.06,
        "topk_noise_min": 0.06,
        "topk_noise_max": 0.12,

        "random_mix_ratio": 0.40,

        "elite_archive_size": 6,
    }

    if method == "qrghgs_struct_v3_elite_exploit":
        cfg = dict(base)
        cfg.update({
            "mode": "elite_exploit",

            # 前 20 次保持原始 V3，后 10 次围绕高潜力区域细搜
            "exploit_start_trial": 20,
            "exploit_keep_batch8": True,
        })
        return cfg

    if method == "qrghgs_struct_v3_perf_trigger":
        cfg = dict(base)
        cfg.update({
            "mode": "perf_trigger",

            # 达到高 Val 后自动收缩
            "trigger_val": 0.965,
            "trigger_global_restart_ratio": 0.05,
            "trigger_random_mix_ratio": 0.10,
            "trigger_local_noise_max": 0.03,
            "trigger_topk_noise_max": 0.05,
        })
        return cfg

    if method == "qrghgs_struct_v3_elite_region":
        cfg = dict(base)
        cfg.update({
            "mode": "elite_region",

            # 每次结构化扰动中，30% 候选回访高潜力区域
            "elite_region_ratio": 0.30,
        })
        return cfg

    if method == "qrghgs_struct_v3_dual_branch":
        cfg = dict(base)
        cfg.update({
            "mode": "dual_branch",

            # 40% 候选来自双分支区域
            "dual_branch_ratio": 0.40,
        })
        return cfg

    raise ValueError(f"Unknown method: {method}")


def get_dynamic_params(cfg, trial_idx, best_fit):
    """
    针对性能触发式版本动态调整扰动强度。
    """
    params = {
        "global_restart_ratio": cfg["global_restart_ratio"],
        "local_noise_max": cfg["local_noise_max"],
        "topk_noise_max": cfg["topk_noise_max"],
        "random_mix_ratio": cfg["random_mix_ratio"],
        "triggered": False,
    }

    if cfg["mode"] == "perf_trigger" and best_fit >= cfg["trigger_val"]:
        params["global_restart_ratio"] = cfg["trigger_global_restart_ratio"]
        params["local_noise_max"] = cfg["trigger_local_noise_max"]
        params["topk_noise_max"] = cfg["trigger_topk_noise_max"]
        params["random_mix_ratio"] = cfg["trigger_random_mix_ratio"]
        params["triggered"] = True

    return params


def strategy_candidate(
    base_cand,
    cfg,
    rng,
    trial_idx,
    historical_centers,
    elite_archive,
    best_fit,
):
    """
    根据不同版本，对候选进行轻微替换或增强。
    """
    mode = cfg["mode"]

    if mode == "elite_exploit":
        if trial_idx >= cfg["exploit_start_trial"]:
            # 后期只围绕高潜力中心小范围细搜
            return sample_elite_region(
                rng,
                historical_centers=historical_centers,
                elite_archive=elite_archive,
                small=True,
            )

    elif mode == "elite_region":
        if rng.random() < cfg["elite_region_ratio"]:
            return sample_elite_region(
                rng,
                historical_centers=historical_centers,
                elite_archive=elite_archive,
                small=False,
            )

    elif mode == "dual_branch":
        if rng.random() < cfg["dual_branch_ratio"]:
            return sample_dual_branch(rng)

    return clip_vec(base_cand)


# =========================================================
# 5. 结构化扰动
# =========================================================
def structured_perturbation(
    rows,
    method_name,
    cfg,
    rng,
    pop,
    fit,
    best_pos,
    best_fit,
    lb,
    ub,
    trial_idx,
    progress,
    historical_centers,
    elite_archive,
):
    pop_size = len(pop)

    dyn = get_dynamic_params(cfg, trial_idx, best_fit)

    if dyn["triggered"]:
        print("[PERF-TRIGGER] best_val 已达到阈值，进入收缩搜索阶段。")

    local_noise = cfg["local_noise_min"] + (
        dyn["local_noise_max"] - cfg["local_noise_min"]
    ) * progress

    topk_noise = cfg["topk_noise_min"] + (
        dyn["topk_noise_max"] - cfg["topk_noise_min"]
    ) * progress

    # -----------------------------------------------------
    # 1) Local exploitation
    # -----------------------------------------------------
    local_k = max(1, math.ceil(cfg["local_ratio"] * pop_size))
    elite_ids = np.argsort(-fit)[:local_k]

    for idx in elite_ids:
        Xi = pop[idx].copy()
        direction = best_pos - Xi

        base_cand = Xi + 0.25 * direction + local_noise * (ub - lb) * rng.normal(size=3)

        cand = strategy_candidate(
            base_cand=base_cand,
            cfg=cfg,
            rng=rng,
            trial_idx=trial_idx,
            historical_centers=historical_centers,
            elite_archive=elite_archive,
            best_fit=best_fit,
        )

        params = vec_to_params(cand)
        row = real_evaluate(method_name, trial_idx, params)
        rows.append(row)
        save_results(rows)

        elite_archive = update_elite_archive(
            row,
            cand,
            elite_archive,
            archive_size=cfg["elite_archive_size"],
        )

        new_fit = row_to_fit(row)
        if new_fit > fit[idx]:
            fit[idx] = new_fit
            pop[idx] = cand.copy()

        trial_idx += 1
        if trial_idx >= SEARCH_BUDGET:
            return rows, pop, fit, trial_idx, True, elite_archive

    # -----------------------------------------------------
    # 2) Top-k migration
    # -----------------------------------------------------
    topk_k = max(1, math.ceil(cfg["topk_ratio"] * pop_size))
    top_ids = list(np.argsort(-fit)[:max(2, topk_k)])

    random_k = max(1, math.ceil(dyn["random_mix_ratio"] * topk_k))
    all_ids = list(range(pop_size))
    random_ids = list(rng.choice(all_ids, size=random_k, replace=True))

    mix_ids = top_ids + random_ids

    for _ in range(topk_k):
        center_idx = int(rng.choice(mix_ids))
        center = pop[center_idx].copy()

        base_cand = center + topk_noise * (ub - lb) * rng.normal(size=3)

        cand = strategy_candidate(
            base_cand=base_cand,
            cfg=cfg,
            rng=rng,
            trial_idx=trial_idx,
            historical_centers=historical_centers,
            elite_archive=elite_archive,
            best_fit=best_fit,
        )

        worst_idx = int(np.argmin(fit))

        params = vec_to_params(cand)
        row = real_evaluate(method_name, trial_idx, params)
        rows.append(row)
        save_results(rows)

        elite_archive = update_elite_archive(
            row,
            cand,
            elite_archive,
            archive_size=cfg["elite_archive_size"],
        )

        new_fit = row_to_fit(row)
        if new_fit > fit[worst_idx]:
            fit[worst_idx] = new_fit
            pop[worst_idx] = cand.copy()

        trial_idx += 1
        if trial_idx >= SEARCH_BUDGET:
            return rows, pop, fit, trial_idx, True, elite_archive

    # -----------------------------------------------------
    # 3) Global restart
    # -----------------------------------------------------
    global_k = max(1, math.ceil(dyn["global_restart_ratio"] * pop_size))
    worst_ids = np.argsort(fit)[:global_k]

    for idx in worst_ids:
        if cfg["mode"] == "dual_branch" and rng.random() < cfg["dual_branch_ratio"]:
            cand = sample_dual_branch(rng)
        elif cfg["mode"] == "elite_region" and rng.random() < cfg["elite_region_ratio"]:
            cand = sample_elite_region(
                rng,
                historical_centers=historical_centers,
                elite_archive=elite_archive,
                small=False,
            )
        elif cfg["mode"] == "elite_exploit" and trial_idx >= cfg["exploit_start_trial"]:
            cand = sample_elite_region(
                rng,
                historical_centers=historical_centers,
                elite_archive=elite_archive,
                small=True,
            )
        else:
            cand = lb + (ub - lb) * rng.random(3)
            cand = clip_vec(cand)

        params = vec_to_params(cand)
        row = real_evaluate(method_name, trial_idx, params)
        rows.append(row)
        save_results(rows)

        elite_archive = update_elite_archive(
            row,
            cand,
            elite_archive,
            archive_size=cfg["elite_archive_size"],
        )

        new_fit = row_to_fit(row)
        if new_fit > fit[idx]:
            fit[idx] = new_fit
            pop[idx] = cand.copy()

        trial_idx += 1
        if trial_idx >= SEARCH_BUDGET:
            return rows, pop, fit, trial_idx, True, elite_archive

    return rows, pop, fit, trial_idx, False, elite_archive


# =========================================================
# 6. QRG-HGS 主体
# =========================================================
def run_qrghgs(rows, method_name):
    cfg = get_config(method_name)
    rng = np.random.default_rng(SEED)

    historical_centers = load_historical_elite_centers(top_k=4)
    print(f"[INFO] {method_name} 使用历史高潜力中心 {len(historical_centers)} 个。")

    max_iters = max(1, math.ceil(SEARCH_BUDGET / POP_SIZE))

    lb = np.array(
        [SPACE["lr"][0], SPACE["dice_weight"][0], 0],
        dtype=np.float64,
    )
    ub = np.array(
        [SPACE["lr"][1], SPACE["dice_weight"][1], len(BATCH_CHOICES) - 1],
        dtype=np.float64,
    )

    pop = lb + (ub - lb) * rng.random((POP_SIZE, 3))
    fit = np.full(POP_SIZE, -1e18)

    elite_archive = []

    trial_idx = 0

    # -----------------------------------------------------
    # 初始化评估
    # -----------------------------------------------------
    for i in range(POP_SIZE):
        params = vec_to_params(pop[i])
        row = real_evaluate(method_name, trial_idx, params)
        rows.append(row)
        save_results(rows)

        fit[i] = row_to_fit(row)
        elite_archive = update_elite_archive(
            row,
            pop[i],
            elite_archive,
            archive_size=cfg["elite_archive_size"],
        )

        trial_idx += 1
        if trial_idx >= SEARCH_BUDGET:
            return rows

    best_idx = int(np.argmax(fit))
    best_pos = pop[best_idx].copy()
    best_fit = float(fit[best_idx])

    stagnation = 0
    qrg_start_iter = int(cfg["qrg_start_ratio"] * max_iters)

    # -----------------------------------------------------
    # 主循环
    # -----------------------------------------------------
    for it in range(max_iters):
        old_best = best_fit
        shrink = 2.0 * (1.0 - it / max(max_iters, 1))
        worst_fit = float(np.min(fit))
        new_pop = pop.copy()
        progress = it / max(max_iters - 1, 1)

        # -------------------------------------------------
        # HGS 主体更新
        # -------------------------------------------------
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
                    base_cand = Xi * (1.0 + rng.normal(0, 0.05, size=3))
                else:
                    if rng.random() > E:
                        base_cand = W1 * best_pos + R * W2 * diff
                    else:
                        base_cand = W1 * best_pos - R * W2 * diff

                cand = strategy_candidate(
                    base_cand=base_cand,
                    cfg=cfg,
                    rng=rng,
                    trial_idx=trial_idx,
                    historical_centers=historical_centers,
                    elite_archive=elite_archive,
                    best_fit=best_fit,
                )

            new_pop[i] = clip_vec(cand)

        # -------------------------------------------------
        # 评估 HGS 更新
        # -------------------------------------------------
        for i in range(POP_SIZE):
            params = vec_to_params(new_pop[i])
            row = real_evaluate(method_name, trial_idx, params)
            rows.append(row)
            save_results(rows)

            elite_archive = update_elite_archive(
                row,
                new_pop[i],
                elite_archive,
                archive_size=cfg["elite_archive_size"],
            )

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

        # -------------------------------------------------
        # QRG + 结构化扰动
        # -------------------------------------------------
        if it >= qrg_start_iter and stagnation >= cfg["stagnation_patience"]:
            theta = cfg["theta_init"] - (cfg["theta_init"] - cfg["theta_final"]) * progress

            elite_k = max(1, math.ceil(cfg["elite_keep_ratio"] * POP_SIZE))
            elite_ids = np.argsort(-fit)[:elite_k]

            # QRG 局部开发
            for idx in elite_ids:
                Xi = pop[idx].copy()
                direction = best_pos - Xi
                base_cand = Xi + theta * direction

                cand = strategy_candidate(
                    base_cand=base_cand,
                    cfg=cfg,
                    rng=rng,
                    trial_idx=trial_idx,
                    historical_centers=historical_centers,
                    elite_archive=elite_archive,
                    best_fit=best_fit,
                )

                params = vec_to_params(cand)
                row = real_evaluate(method_name, trial_idx, params)
                rows.append(row)
                save_results(rows)

                elite_archive = update_elite_archive(
                    row,
                    cand,
                    elite_archive,
                    archive_size=cfg["elite_archive_size"],
                )

                new_fit = row_to_fit(row)
                if new_fit > fit[idx]:
                    fit[idx] = new_fit
                    pop[idx] = cand.copy()

                trial_idx += 1
                if trial_idx >= SEARCH_BUDGET:
                    return rows

            rows, pop, fit, trial_idx, done, elite_archive = structured_perturbation(
                rows=rows,
                method_name=method_name,
                cfg=cfg,
                rng=rng,
                pop=pop,
                fit=fit,
                best_pos=best_pos,
                best_fit=best_fit,
                lb=lb,
                ub=ub,
                trial_idx=trial_idx,
                progress=progress,
                historical_centers=historical_centers,
                elite_archive=elite_archive,
            )

            if done:
                return rows

            stagnation = 0
            best_idx = int(np.argmax(fit))
            best_pos = pop[best_idx].copy()
            best_fit = float(fit[best_idx])

    return rows


# =========================================================
# 7. 第五方向：Top3 多 seed confirm
# =========================================================
def run_top3_multiseed_confirm():
    confirm_root = OUT_ROOT / "top3_multiseed_confirm"
    confirm_train_root = confirm_root / "train_runs"
    confirm_json_root = confirm_root / "trial_json"

    confirm_root.mkdir(parents=True, exist_ok=True)
    confirm_train_root.mkdir(parents=True, exist_ok=True)
    confirm_json_root.mkdir(parents=True, exist_ok=True)

    confirm_raw_csv = confirm_root / "confirm_raw.csv"
    candidate_summary_csv = confirm_root / "confirm_candidate_summary.csv"
    method_summary_csv = confirm_root / "confirm_method_summary.csv"

    if not RAW_CSV.exists():
        raise FileNotFoundError(f"找不到搜索结果文件: {RAW_CSV}")

    df = pd.read_csv(RAW_CSV)
    df = df[df["status"] == "OK"].copy()

    if "resumed" not in df.columns:
        df["resumed"] = False

    df = (
        df.sort_values(["method", "trial_idx", "resumed"])
        .drop_duplicates(subset=["method", "run_name"], keep="last")
        .reset_index(drop=True)
    )

    for col in ["best_val_dice", "lr", "dice_weight", "batch_size"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["best_val_dice", "lr", "dice_weight", "batch_size"])

    candidates = []
    for method in CONFIRM_METHODS:
        sub = df[df["method"] == method].copy()

        if len(sub) == 0:
            print(f"[WARN] confirm 跳过 {method}，因为没有搜索记录。")
            continue

        sub = sub.sort_values("best_val_dice", ascending=False).head(TOP_K)

        for rank_idx, (_, row) in enumerate(sub.iterrows(), start=1):
            cand = row.to_dict()
            cand["candidate_rank"] = rank_idx
            candidates.append(cand)

    if confirm_raw_csv.exists():
        try:
            confirm_rows = pd.read_csv(confirm_raw_csv).to_dict("records")
        except Exception:
            confirm_rows = []
    else:
        confirm_rows = []

    def make_confirm_run_name(source_method, rank_idx, seed):
        return f"confirm_{source_method}_rank{rank_idx}_seed{seed}"

    def load_confirm_json_if_ok(path):
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("status", "OK") == "OK":
                return data
        except Exception:
            return None
        return None

    def save_confirm_results(rows):
        cdf = pd.DataFrame(rows)
        cdf.to_csv(confirm_raw_csv, index=False, encoding="utf-8-sig")

        if len(cdf) == 0:
            return

        ok_df = cdf[cdf["status"] == "OK"].copy()

        if len(ok_df) == 0:
            return

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

        if len(cand_df) > 0:
            cand_df = cand_df.sort_values(
                ["source_method", "confirm_test_dice_mean"],
                ascending=[True, False],
            )
            cand_df.to_csv(candidate_summary_csv, index=False, encoding="utf-8-sig")

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

        if len(method_df) > 0:
            method_df = method_df.sort_values("confirm_test_dice_mean", ascending=False)
            method_df.to_csv(method_summary_csv, index=False, encoding="utf-8-sig")

    for cand in candidates:
        source_method = cand["method"]
        rank_idx = int(cand["candidate_rank"])
        source_run_name = cand["run_name"]

        lr = float(cand["lr"])
        dice_weight = float(cand["dice_weight"])
        batch_size = int(cand["batch_size"])

        for seed in CONFIRM_SEEDS:
            confirm_run_name = make_confirm_run_name(source_method, rank_idx, seed)
            output_json = confirm_json_root / f"{confirm_run_name}.json"

            old = load_confirm_json_if_ok(output_json)

            if old is not None:
                print(f"[SKIP CONFIRM] {confirm_run_name}")

                row = {
                    "source_method": source_method,
                    "candidate_rank": rank_idx,
                    "source_run_name": source_run_name,
                    "confirm_run_name": confirm_run_name,
                    "seed": seed,
                    "lr": lr,
                    "dice_weight": dice_weight,
                    "batch_size": batch_size,
                    "source_best_val_dice": cand.get("best_val_dice"),
                    "source_test_dice": cand.get("test_dice"),
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

                confirm_rows.append(row)
                save_confirm_results(confirm_rows)
                continue

            cmd = [
                PYTHON_EXE,
                "scripts/train_week3_unet.py",
                "--input_mode", "2.5d",
                "--no_post",
                "--save_root", str(confirm_train_root),
                "--epochs", str(CONFIRM_EPOCHS),
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
                row = {
                    "source_method": source_method,
                    "candidate_rank": rank_idx,
                    "source_run_name": source_run_name,
                    "confirm_run_name": confirm_run_name,
                    "seed": seed,
                    "lr": lr,
                    "dice_weight": dice_weight,
                    "batch_size": batch_size,
                    "source_best_val_dice": cand.get("best_val_dice"),
                    "source_test_dice": cand.get("test_dice"),
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

                confirm_rows.append(row)
                save_confirm_results(confirm_rows)
                continue

            data = json.loads(output_json.read_text(encoding="utf-8"))

            row = {
                "source_method": source_method,
                "candidate_rank": rank_idx,
                "source_run_name": source_run_name,
                "confirm_run_name": confirm_run_name,
                "seed": seed,
                "lr": lr,
                "dice_weight": dice_weight,
                "batch_size": batch_size,
                "source_best_val_dice": cand.get("best_val_dice"),
                "source_test_dice": cand.get("test_dice"),
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

            confirm_rows.append(row)
            save_confirm_results(confirm_rows)

    print("\n✅ Top3 多 seed confirm 完成。")
    print("confirm_raw:", confirm_raw_csv)
    print("candidate_summary:", candidate_summary_csv)
    print("method_summary:", method_summary_csv)


# =========================================================
# 8. 主入口
# =========================================================
if __name__ == "__main__":
    if RUN_SEARCH:
        rows = load_existing_rows_clean()
        print(f"[INFO] 已读取并清理历史记录 {len(rows)} 条，将在同目录继续续跑。")

        for method in METHODS:
            print(f"\n\n########## RUN METHOD: {method} ##########")
            rows = run_qrghgs(rows, method)

            _, summary_df = save_results(rows)
            print("\n===== CURRENT SUMMARY =====")
            print(summary_df)

        print("\n✅ 4 个 Struct_v3 改进搜索版本已完成。")

    if RUN_CONFIRM:
        run_top3_multiseed_confirm()
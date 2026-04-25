import json
import re
from pathlib import Path
import pandas as pd

# =========================
# 路径配置
# =========================
QRGHGS_JSON_ROOT = Path("outputs/week4/trial_json")
BASELINE_JSON_ROOT = Path("outputs/week4_compare/trial_json")

OUT_ROOT = Path("outputs/week4_merged")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

QRGHGS_OUT = OUT_ROOT / "qrghgs_from_week4.csv"
BASELINE_OUT = OUT_ROOT / "baseline_from_week4_compare.csv"
MERGED_OUT = OUT_ROOT / "all_methods_merged.csv"
SUMMARY_OUT = OUT_ROOT / "all_methods_summary.csv"


def try_load_json(json_path: Path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_trial_idx(name: str):
    m = re.search(r"trial(\d+)", name)
    if m:
        return int(m.group(1))
    return None


# =========================
# 1) 提取 week4 里的 qrghgs
# =========================
qrghgs_rows = []

for jf in QRGHGS_JSON_ROOT.glob("*.json"):
    run_name = jf.stem

    if not run_name.startswith("qrghgs_trial"):
        continue

    data = try_load_json(jf)
    if data is None:
        continue

    if data.get("status", "OK") != "OK":
        continue

    # 统一协议筛选
    if data.get("input_mode") != "2.5d":
        continue
    if data.get("postprocess") is not False:
        continue
    if int(data.get("epochs", -1)) != 20:
        continue

    row = {
        "method": "qrghgs",
        "trial_idx": extract_trial_idx(run_name),
        "run_name": data.get("run_name", run_name),
        "seed": data.get("seed", 42),
        "lr": data.get("lr"),
        "dice_weight": data.get("dice_weight"),
        "batch_size": data.get("batch_size"),
        "best_val_dice": data.get("best_val_dice"),
        "test_dice": data.get("test_dice"),
        "test_iou": data.get("test_iou"),
        "test_sens": data.get("test_sens"),
        "test_spec": data.get("test_spec"),
        "time_sec": data.get("time_sec"),
        "status": data.get("status", "OK"),
        "error_message": data.get("error_message", ""),
        "best_ckpt_path": data.get("best_ckpt_path", ""),
        "source": "week4",
    }
    qrghgs_rows.append(row)

qrghgs_df = pd.DataFrame(qrghgs_rows)
qrghgs_df = qrghgs_df.sort_values("trial_idx").reset_index(drop=True)
qrghgs_df.to_csv(QRGHGS_OUT, index=False, encoding="utf-8-sig")

print(f"[OK] qrghgs rows = {len(qrghgs_df)}")
print(f"[SAVE] {QRGHGS_OUT}")


# =========================
# 2) 提取 week4_compare 的 baseline
# =========================
baseline_rows = []

for jf in BASELINE_JSON_ROOT.glob("*.json"):
    run_name = jf.stem
    data = try_load_json(jf)
    if data is None:
        continue

    if run_name.startswith("random_trial"):
        method = "random"
    elif run_name.startswith("tpe_trial"):
        method = "tpe"
    elif run_name.startswith("hgs_trial"):
        method = "hgs"
    else:
        continue

    if data.get("status", "OK") != "OK":
        continue

    row = {
        "method": method,
        "trial_idx": extract_trial_idx(run_name),
        "run_name": data.get("run_name", run_name),
        "seed": data.get("seed", 42),
        "lr": data.get("lr"),
        "dice_weight": data.get("dice_weight"),
        "batch_size": data.get("batch_size"),
        "best_val_dice": data.get("best_val_dice"),
        "test_dice": data.get("test_dice"),
        "test_iou": data.get("test_iou"),
        "test_sens": data.get("test_sens"),
        "test_spec": data.get("test_spec"),
        "time_sec": data.get("time_sec"),
        "status": data.get("status", "OK"),
        "error_message": data.get("error_message", ""),
        "best_ckpt_path": data.get("best_ckpt_path", ""),
        "source": "week4_compare",
    }
    baseline_rows.append(row)

baseline_df = pd.DataFrame(baseline_rows)
baseline_df = baseline_df.sort_values(["method", "trial_idx"]).reset_index(drop=True)
baseline_df.to_csv(BASELINE_OUT, index=False, encoding="utf-8-sig")

print(f"[OK] baseline rows = {len(baseline_df)}")
print(f"[SAVE] {BASELINE_OUT}")


# =========================
# 3) 合并
# =========================
all_df = pd.concat([baseline_df, qrghgs_df], ignore_index=True)

all_df = all_df[all_df["status"] == "OK"].copy()

# 去重
all_df = (
    all_df.sort_values(["method", "trial_idx"])
          .drop_duplicates(subset=["method", "run_name"], keep="last")
          .reset_index(drop=True)
)

all_df.to_csv(MERGED_OUT, index=False, encoding="utf-8-sig")
print(f"[SAVE] {MERGED_OUT}")


# =========================
# 4) 汇总表
# =========================
summary_rows = []
for method, g in all_df.groupby("method"):
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
summary_df = summary_df.sort_values("mean_test_dice", ascending=False).reset_index(drop=True)
summary_df.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")

print("\n===== SUMMARY =====")
print(summary_df)
print(f"[SAVE] {SUMMARY_OUT}")
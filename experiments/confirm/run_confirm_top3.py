import os
import json
import sys
import subprocess
import pandas as pd

SEEDS = [42, 52, 62, 72, 82]

CANDIDATES = [
    {
        "candidate_name": "QRGHGS_topA_trial003",
        "lr": 0.0012030274048531,
        "dice_weight": 0.732031863277994,
        "batch_size": 4,
    },
    {
        "candidate_name": "QRGHGS_topB_trial047",
        "lr": 0.00151021969539,
        "dice_weight": 0.8,
        "batch_size": 4,
    },
    {
        "candidate_name": "QRGHGS_topC_trial044",
        "lr": 0.0012740085973707,
        "dice_weight": 0.62,
        "batch_size": 4,
    },
]

SAVE_ROOT = "outputs/week4/confirm_runs"
JSON_ROOT = "outputs/week4/confirm_json"
RAW_CSV = "outputs/week4/confirm_top3_raw.csv"
SUMMARY_CSV = "outputs/week4/confirm_top3_summary.csv"

os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(JSON_ROOT, exist_ok=True)


def load_ok_json(output_json: str):
    """如果 json 存在且 status=OK，则返回内容，否则返回 None"""
    if not os.path.exists(output_json):
        return None
    try:
        with open(output_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("status", "") == "OK":
            return data
    except Exception:
        return None
    return None


def build_row(cname, run_name, seed, cand, data=None, status="ERROR", error_message=""):
    row = {
        "candidate_name": cname,
        "run_name": run_name,
        "seed": seed,
        "lr": cand["lr"],
        "dice_weight": cand["dice_weight"],
        "batch_size": cand["batch_size"],
        "best_val_dice": None,
        "test_dice": None,
        "test_iou": None,
        "test_sens": None,
        "test_spec": None,
        "time_sec": None,
        "status": status,
        "error_message": error_message,
        "best_ckpt_path": "",
    }
    if data is not None:
        row.update({
            "best_val_dice": data.get("best_val_dice"),
            "test_dice": data.get("test_dice"),
            "test_iou": data.get("test_iou"),
            "test_sens": data.get("test_sens"),
            "test_spec": data.get("test_spec"),
            "time_sec": data.get("time_sec"),
            "status": data.get("status", "OK"),
            "error_message": data.get("error_message", ""),
            "best_ckpt_path": data.get("best_ckpt_path", ""),
        })
    return row


all_rows = []

for cand in CANDIDATES:
    cname = cand["candidate_name"]

    for seed in SEEDS:
        run_name = f"{cname}_s{seed}"
        output_json = os.path.join(JSON_ROOT, f"{run_name}.json")

        # 先检查是否已经成功生成，成功就直接跳过训练
        old = load_ok_json(output_json)
        if old is not None:
            print(f"[SKIP] 已存在成功结果: {run_name}")
            all_rows.append(build_row(cname, run_name, seed, cand, data=old, status="OK"))
            continue

        cmd = [
            sys.executable,
            "scripts/train_week3_unet.py",
            "--input_mode", "2.5d",
            "--no_post",
            "--save_root", SAVE_ROOT,
            "--epochs", "20",
            "--seed", str(seed),
            "--batch_size", str(cand["batch_size"]),
            "--lr", str(cand["lr"]),
            "--dice_weight", str(cand["dice_weight"]),
            "--run_name", run_name,
            "--output_json", output_json,
        ]

        print("\n" + "=" * 80)
        print(f"[RUN] {run_name}")
        print(" ".join(cmd))
        print("=" * 80)

        proc = subprocess.run(cmd)

        # 训练进程失败
        if proc.returncode != 0:
            all_rows.append(build_row(
                cname, run_name, seed, cand,
                data=None,
                status="ERROR",
                error_message="subprocess_failed"
            ))
            continue

        # 训练进程成功，但没写 output_json
        data = load_ok_json(output_json)
        if data is None:
            err_msg = "missing_output_json_or_not_ok"
            # 如果 json 存在但不是 OK，也尝试读取具体错误
            if os.path.exists(output_json):
                try:
                    with open(output_json, "r", encoding="utf-8") as f:
                        temp = json.load(f)
                    err_msg = temp.get("error_message", err_msg)
                    all_rows.append(build_row(
                        cname, run_name, seed, cand,
                        data=temp,
                        status=temp.get("status", "ERROR"),
                        error_message=err_msg
                    ))
                    continue
                except Exception:
                    pass

            all_rows.append(build_row(
                cname, run_name, seed, cand,
                data=None,
                status="ERROR",
                error_message=err_msg
            ))
            continue

        # 成功
        all_rows.append(build_row(cname, run_name, seed, cand, data=data, status="OK"))

# 保存原始表
raw_df = pd.DataFrame(all_rows)
raw_df = raw_df.sort_values(by=["candidate_name", "seed"]).reset_index(drop=True)
raw_df.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")

print("\n========== STATUS COUNTS ==========")
print(raw_df["status"].value_counts(dropna=False))

ok_df = raw_df[raw_df["status"] == "OK"].copy()

summary_rows = []
for cname, g in ok_df.groupby("candidate_name"):
    summary_rows.append({
        "candidate_name": cname,
        "lr": g["lr"].iloc[0],
        "dice_weight": g["dice_weight"].iloc[0],
        "batch_size": g["batch_size"].iloc[0],
        "n_ok": len(g),
        "mean_best_val_dice": g["best_val_dice"].mean(),
        "std_best_val_dice": g["best_val_dice"].std(ddof=0),
        "mean_test_dice": g["test_dice"].mean(),
        "std_test_dice": g["test_dice"].std(ddof=0),
        "mean_test_iou": g["test_iou"].mean(),
        "std_test_iou": g["test_iou"].std(ddof=0),
        "mean_test_sens": g["test_sens"].mean(),
        "std_test_sens": g["test_sens"].std(ddof=0),
        "mean_test_spec": g["test_spec"].mean(),
        "std_test_spec": g["test_spec"].std(ddof=0),
        "mean_time_sec": g["time_sec"].mean(),
    })

summary_df = pd.DataFrame(summary_rows)

if len(summary_df) == 0:
    print("\n[WARN] 当前没有任何成功的 confirm 结果。")
else:
    summary_df = summary_df.sort_values(by="mean_test_dice", ascending=False).reset_index(drop=True)
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\n========== SUMMARY ==========")
    print(summary_df)
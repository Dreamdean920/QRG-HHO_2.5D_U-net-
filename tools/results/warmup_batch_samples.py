import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PYTHON_EXE = sys.executable
OUT_ROOT = Path("outputs/batch_warmup")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_PER_BATCH = 6   # 每个 batch 先补 6 个点
EPOCHS = 5        # 先短训，降低成本

LR_RANGE = (0.0008, 0.0022)
DICE_RANGE = (0.58, 0.82)
BATCHES = [2, 8]


def main():
    rng = np.random.default_rng(SEED)
    rows = []

    idx = 0
    for b in BATCHES:
        for _ in range(N_PER_BATCH):
            lr = float(rng.uniform(*LR_RANGE))
            dice_weight = float(rng.uniform(*DICE_RANGE))

            run_name = f"warmup_b{b}_{idx:03d}"
            output_json = OUT_ROOT / f"{run_name}.json"

            cmd = [
                PYTHON_EXE,
                "scripts/train_week3_unet.py",
                "--input_mode", "2.5d",
                "--no_post",
                "--save_root", str(OUT_ROOT / "train_runs"),
                "--epochs", str(EPOCHS),
                "--seed", str(SEED),
                "--batch_size", str(b),
                "--lr", str(lr),
                "--dice_weight", str(dice_weight),
                "--run_name", run_name,
                "--output_json", str(output_json),
            ]

            print("\n[RUN]", " ".join(cmd))
            proc = subprocess.run(cmd)

            if proc.returncode != 0 or not output_json.exists():
                rows.append({
                    "run_name": run_name,
                    "batch_size": b,
                    "lr": lr,
                    "dice_weight": dice_weight,
                    "status": "ERROR",
                    "best_val_dice": None,
                    "test_dice": None,
                    "time_sec": None,
                })
            else:
                data = json.loads(output_json.read_text(encoding="utf-8"))
                rows.append({
                    "run_name": run_name,
                    "batch_size": b,
                    "lr": lr,
                    "dice_weight": dice_weight,
                    "status": data.get("status", "OK"),
                    "best_val_dice": data.get("best_val_dice"),
                    "test_dice": data.get("test_dice"),
                    "time_sec": data.get("time_sec"),
                })

            idx += 1

    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "warmup_batch_results.csv", index=False, encoding="utf-8-sig")
    print("\n✅ 已保存:", OUT_ROOT / "warmup_batch_results.csv")
    print(df)


if __name__ == "__main__":
    main()
"""
Realtime-print HGS / QRG-HGS hyperparameter search for a week2/week3-style U-Net pipeline.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from hgs_core import hgs_optimize
from qrghgs_dimtheta import qrghgs_optimize


def decode_position(pos: np.ndarray) -> Dict[str, Any]:
    lr = float(pos[0])
    dice_weight = float(pos[1])
    batch_candidates = [2, 4, 8]
    batch_idx = int(np.clip(np.round(pos[2]), 0, len(batch_candidates) - 1))
    batch_size = int(batch_candidates[batch_idx])
    return {
        "lr": lr,
        "dice_weight": dice_weight,
        "batch_size": batch_size,
        "batch_idx": batch_idx,
    }


def build_train_command(args, hp: Dict[str, Any], run_name: str, out_json: Path) -> List[str]:
    cmd = [
        sys.executable,
        args.train_script,
        "--lr", str(hp["lr"]),
        "--dice-weight", str(hp["dice_weight"]),
        "--batch-size", str(hp["batch_size"]),
        "--run-name", run_name,
        "--output-json", str(out_json),
    ]
    if args.input_mode:
        cmd += ["--input-mode", args.input_mode]
    if args.no_post:
        cmd += ["--no-post"]
    if args.extra_args:
        cmd += args.extra_args
    return cmd


def read_metrics(json_path: Path) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def get_first(*keys, default=None):
        for k in keys:
            if k in data:
                return data[k]
        return default

    return {
        "val_dice": float(get_first("best_val_dice", "val_dice", "best_dice", default=0.0)),
        "test_dice": float(get_first("test_dice", default=0.0)),
        "test_iou": float(get_first("test_iou", default=0.0)),
        "test_sens": float(get_first("test_sens", "test_sensitivity", default=0.0)),
        "test_spec": float(get_first("test_spec", "test_specificity", default=0.0)),
        "raw": data,
    }


def write_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_hp(hp: Dict[str, Any]) -> str:
    return f"lr={hp['lr']:.6f}, dice_weight={hp['dice_weight']:.4f}, batch_size={hp['batch_size']}"


def run_and_stream(cmd: List[str], log_file: Path) -> None:
    """
    实时把子进程输出同时写到终端和日志文件。
    """
    with open(log_file, "w", encoding="utf-8") as logf:
        logf.write("COMMAND: " + " ".join(cmd) + "\n\n")
        logf.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            logf.write(line)
            logf.flush()

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-script", type=str, required=True)
    parser.add_argument("--algo", type=str, default="hgs", choices=["hgs", "qrghgs"])
    parser.add_argument("--save-root", type=str, default="outputs/search_hgs")
    parser.add_argument("--pop-size", type=int, default=6)
    parser.add_argument("--max-iter", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-mode", type=str, default="25d")
    parser.add_argument("--no-post", action="store_true", default=True)
    parser.add_argument("--extra-args", nargs="*", default=[])

    parser.add_argument("--lr-min", type=float, default=8e-4)
    parser.add_argument("--lr-max", type=float, default=2e-3)
    parser.add_argument("--dw-min", type=float, default=0.60)
    parser.add_argument("--dw-max", type=float, default=0.80)

    parser.add_argument("--qrg-start-ratio", type=float, default=0.5)
    parser.add_argument("--qrg-interval", type=int, default=1)
    parser.add_argument("--qrg-top-ratio", type=float, default=0.34)
    parser.add_argument("--theta-max", type=float, default=math.pi / 10)
    parser.add_argument("--theta-min", type=float, default=math.pi / 60)

    args = parser.parse_args()

    save_root = Path(args.save_root) / args.algo
    trial_dir = save_root / "trial_jsons"
    log_dir = save_root / "logs"
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    lb = np.array([args.lr_min, args.dw_min, 0.0], dtype=np.float64)
    ub = np.array([args.lr_max, args.dw_max, 2.0], dtype=np.float64)

    print("\n===== SEARCH CONFIG =====")
    print(json.dumps({
        "algo": args.algo,
        "train_script": args.train_script,
        "save_root": str(save_root),
        "pop_size": args.pop_size,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "input_mode": args.input_mode,
        "no_post": args.no_post,
        "lr_range": [args.lr_min, args.lr_max],
        "dice_weight_range": [args.dw_min, args.dw_max],
        "batch_candidates": [2, 4, 8],
        "extra_args": args.extra_args,
    }, indent=2, ensure_ascii=False))

    best_so_far_fitness = float("inf")
    best_so_far_trial = None

    def objective_fn(pos: np.ndarray, trial_idx: int) -> Dict[str, Any]:
        nonlocal best_so_far_fitness, best_so_far_trial

        hp = decode_position(pos)
        run_name = f"{args.algo}_trial{trial_idx:03d}_bs{hp['batch_size']}_lr{hp['lr']:.6f}_dw{hp['dice_weight']:.4f}"
        out_json = trial_dir / f"{run_name}.json"
        log_file = log_dir / f"{run_name}.log"
        cmd = build_train_command(args, hp, run_name, out_json)

        print(f"\n===== TRIAL {trial_idx:03d} START =====")
        print(f"[TRIAL] run_name = {run_name}")
        print(f"[TRIAL] hparams  = {format_hp(hp)}")
        print(f"[TRIAL] log_file = {log_file}")
        print(f"[TRIAL] out_json = {out_json}")
        print("[TRIAL] command  = " + " ".join(cmd))
        print("=" * 60)

        t0 = time.time()
        status = "OK"
        error_message = ""
        val_dice = 0.0
        test_dice = 0.0
        test_iou = 0.0
        test_sens = 0.0
        test_spec = 0.0

        try:
            run_and_stream(cmd, log_file)
            metrics = read_metrics(out_json)
            val_dice = metrics["val_dice"]
            test_dice = metrics["test_dice"]
            test_iou = metrics["test_iou"]
            test_sens = metrics["test_sens"]
            test_spec = metrics["test_spec"]
        except Exception as e:
            status = "FAIL"
            error_message = str(e)

        time_sec = time.time() - t0
        fitness = -val_dice if status == "OK" else 1e6

        if fitness < best_so_far_fitness:
            best_so_far_fitness = fitness
            best_so_far_trial = run_name

        trial_json = {
            "trial_idx": trial_idx,
            "run_name": run_name,
            "algo": args.algo,
            "lr": hp["lr"],
            "dice_weight": hp["dice_weight"],
            "batch_size": hp["batch_size"],
            "fitness": fitness,
            "val_dice": val_dice,
            "test_dice": test_dice,
            "test_iou": test_iou,
            "test_sens": test_sens,
            "test_spec": test_spec,
            "time_sec": time_sec,
            "status": status,
            "error_message": error_message,
            "log_path": str(log_file),
            "result_json": str(out_json),
            "best_so_far_fitness": best_so_far_fitness,
            "best_so_far_trial": best_so_far_trial,
        }

        with open(trial_dir / f"{run_name}_summary.json", "w", encoding="utf-8") as f:
            json.dump(trial_json, f, indent=2, ensure_ascii=False)

        print("\n----- TRIAL RESULT -----")
        print(json.dumps(trial_json, indent=2, ensure_ascii=False))
        print("------------------------")

        return trial_json

    search_start_time = time.time()

    if args.algo == "hgs":
        result = hgs_optimize(
            objective_fn=objective_fn,
            dim=3,
            lb=lb,
            ub=ub,
            pop_size=args.pop_size,
            max_iter=args.max_iter,
            seed=args.seed,
            verbose=True,
        )
    else:
        result = qrghgs_optimize(
            objective_fn=objective_fn,
            dim=3,
            lb=lb,
            ub=ub,
            pop_size=args.pop_size,
            max_iter=args.max_iter,
            seed=args.seed,
            qrg_start_ratio=args.qrg_start_ratio,
            qrg_interval=args.qrg_interval,
            qrg_top_ratio=args.qrg_top_ratio,
            theta_max=args.theta_max,
            theta_min=args.theta_min,
            verbose=True,
        )

    search_time_sec = time.time() - search_start_time

    best_hp = decode_position(result.best_position)
    summary = {
        "algo": args.algo,
        "best_position": result.best_position.tolist(),
        "best_hparams": best_hp,
        "best_fitness": float(result.best_fitness),
        "history_best_fitness": [float(x) for x in result.history_best_fitness],
        "history_best_position": result.history_best_position,
        "search_time_sec": search_time_sec,
    }

    write_csv(result.trial_records, save_root / "trial_records.csv")
    with open(save_root / "search_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== SEARCH FINISHED =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
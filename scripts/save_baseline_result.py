from pathlib import Path
import yaml
import pandas as pd


def load_config():
    with open("configs/baseline.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    train_log_path = Path("outputs/logs/baseline_train_log.csv")
    test_metrics_path = Path("outputs/metrics/baseline_test_metrics.csv")
    save_path = Path("results/baseline_results.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not train_log_path.exists():
        raise FileNotFoundError(f"找不到训练日志: {train_log_path}")
    if not test_metrics_path.exists():
        raise FileNotFoundError(f"找不到测试指标: {test_metrics_path}")

    train_df = pd.read_csv(train_log_path)
    test_df = pd.read_csv(test_metrics_path)

    best_idx = train_df["val_dice"].idxmax()
    best_row = train_df.iloc[best_idx]

    row = {
        "version_name": "Baseline-Strict-A",
        "input_size": cfg["train"]["input_size"],
        "batch_size": cfg["train"]["batch_size"],
        "lr": cfg["train"]["lr"],
        "use_augmentation": cfg["train"].get("use_augmentation", False),
        "best_epoch": int(best_row["epoch"]),
        "best_val_dice": float(best_row["val_dice"]),
        "best_val_iou": float(best_row["val_iou"]),
        "test_dice": float(test_df.iloc[0]["test_dice"]),
        "test_iou": float(test_df.iloc[0]["test_iou"])
    }

    out_df = pd.DataFrame([row])
    out_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"Baseline结果已保存到: {save_path}")
    print(out_df)


if __name__ == "__main__":
    main()
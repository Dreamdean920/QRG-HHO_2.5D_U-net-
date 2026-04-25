import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

FEATURES = ["lr", "dice_weight", "batch_size"]
TARGET = "best_val_dice"
BATCH_CHOICES = [2, 4, 8]


def load_history_from_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "OK"].copy()
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    return df


def load_history_from_json_dir(json_dir: str) -> pd.DataFrame:
    rows = []
    for jf in Path(json_dir).glob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue

        if data.get("status", "OK") != "OK":
            continue

        row = {
            "run_name": data.get("run_name", jf.stem),
            "lr": data.get("lr"),
            "dice_weight": data.get("dice_weight"),
            "batch_size": data.get("batch_size"),
            "best_val_dice": data.get("best_val_dice"),
            "test_dice": data.get("test_dice"),
            "time_sec": data.get("time_sec"),
            "status": data.get("status", "OK"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    return df


def inspect_history_coverage(df: pd.DataFrame) -> None:
    print("\n===== 历史数据覆盖检查 =====")
    print("总样本数:", len(df))

    if len(df) == 0:
        print("⚠️ 没有可用历史数据")
        return

    counts = df["batch_size"].value_counts().sort_index()
    for b in BATCH_CHOICES:
        n = int(counts.get(b, 0))
        print(f"batch_size={b}: {n} 条")

    missing = [b for b in BATCH_CHOICES if int(counts.get(b, 0)) == 0]
    low = [b for b in BATCH_CHOICES if 0 < int(counts.get(b, 0)) < 5]

    if missing:
        print(f"⚠️ 以下 batch_size 完全没有样本: {missing}")
        print("⚠️ 代理模型对这些 batch 的预测不可靠，建议先补少量真实样本。")

    if low:
        print(f"⚠️ 以下 batch_size 样本过少(<5): {low}")
        print("⚠️ 建议至少补到每个 batch 6~10 个样本。")

    print("\n各 batch 的 best_val_dice 统计：")
    print(df.groupby("batch_size")[TARGET].agg(["count", "mean", "max", "std"]))


class RFEnsembleSurrogate:
    def __init__(self, n_models: int = 8, random_seed: int = 42):
        self.n_models = n_models
        self.random_seed = random_seed
        self.models = []

    def fit(self, df: pd.DataFrame) -> None:
        inspect_history_coverage(df)

        X = df[FEATURES].values
        y = df[TARGET].values

        self.models = []
        rng = np.random.default_rng(self.random_seed)

        for i in range(self.n_models):
            idx = rng.integers(0, len(df), size=len(df))
            Xb, yb = X[idx], y[idx]

            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                random_state=self.random_seed + i,
                n_jobs=-1,
            )
            model.fit(Xb, yb)
            self.models.append(model)

    def predict_mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        preds = []
        for m in self.models:
            preds.append(m.predict(X))
        preds = np.stack(preds, axis=0)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return mean, std
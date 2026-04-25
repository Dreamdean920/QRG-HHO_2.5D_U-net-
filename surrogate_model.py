import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


FEATURES = ["lr", "dice_weight", "batch_size"]
TARGET = "best_val_dice"


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


class RFEnsembleSurrogate:
    def __init__(self, n_models: int = 8, random_seed: int = 42):
        self.n_models = n_models
        self.random_seed = random_seed
        self.models = []

    def fit(self, df: pd.DataFrame) -> None:
        X = df[FEATURES].values
        y = df[TARGET].values

        self.models = []
        rng = np.random.default_rng(self.random_seed)

        for i in range(self.n_models):
            # bootstrap 采样，构造“伪不确定性”
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
        preds = np.stack(preds, axis=0)  # [n_models, n_samples]
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return mean, std
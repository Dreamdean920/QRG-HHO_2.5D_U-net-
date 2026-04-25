from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

from utils import load_yaml, ensure_dir
from objective_adapter import ObjectiveAdapter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source_csv", required=True, help="主搜索结果 CSV，例如 outputs/master_results.csv")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    adapter = ObjectiveAdapter(cfg)
    out_dir = ensure_dir(Path(cfg["output_root"]) / "multiseed_confirm")

    df = pd.read_csv(args.source_csv)
    if "method" not in df.columns:
        raise ValueError("source_csv 里缺少 method 列")
    if "score" not in df.columns:
        raise ValueError("source_csv 里缺少 score 列")

    best_rows = df.sort_values("score", ascending=False).groupby("method").head(1)
    all_rows = []
    for _, row in best_rows.iterrows():
        method = row["method"]
        params = {k: row[k] for k in cfg["space"].keys()}
        for seed in cfg["search"]["confirm_seeds"]:
            run_name = f"confirm_{method}_s{seed}"
            res = adapter.evaluate(params=params, seed=int(seed), run_name=run_name, mode="confirm_eval")
            res["method"] = method
            res["confirm_seed"] = int(seed)
            all_rows.append(res)

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(out_dir / "multiseed_raw.csv", index=False)

    agg = out_df.groupby("method").agg(
        mean_dice=("test_dice", "mean"),
        std_dice=("test_dice", "std"),
        mean_iou=("test_iou", "mean"),
        std_iou=("test_iou", "std"),
        mean_sens=("test_sens", "mean"),
        mean_spec=("test_spec", "mean"),
    ).reset_index()
    agg.to_csv(out_dir / "multiseed_summary.csv", index=False)
    print(agg)

if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from utils import load_yaml, ensure_dir, save_json
from objective_adapter import ObjectiveAdapter
from qrghgs import AdaptiveQRGHGS

def clone_cfg(cfg):
    import copy
    return copy.deepcopy(cfg)

def run_variant(name, cfg, adapter):
    searcher = AdaptiveQRGHGS(cfg=cfg, evaluate_fn=adapter.evaluate, rng_seed=int(cfg["search"]["random_seed"]))
    best, history = searcher.optimize(eval_seed=cfg["search"]["eval_seeds"][0], method_name=name)
    df = pd.DataFrame(history)
    df["variant"] = name
    return best, df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    adapter = ObjectiveAdapter(cfg)
    out_dir = ensure_dir(Path(cfg["output_root"]) / "ablations")

    variants = {}

    # HGS：关掉 QRG 影响，等价做法是把 qrg_start_ratio 设到 1.1，并放宽停滞
    c = clone_cfg(cfg)
    c["qrg_hgs"]["qrg_start_ratio"] = 1.1
    c["qrg_hgs"]["stagnation_patience"] = 999
    variants["hgs"] = c

    # QRG-HGS 固定触发：从后期固定开始，不用停滞触发
    c = clone_cfg(cfg)
    c["qrg_hgs"]["qrg_start_ratio"] = 0.4
    c["qrg_hgs"]["stagnation_patience"] = 999
    variants["qrg_hgs_fixed"] = c

    # Adaptive 主方法
    variants["adaptive_qrghgs"] = clone_cfg(cfg)

    # w/o QRG
    c = clone_cfg(cfg)
    c["qrg_hgs"]["qrg_start_ratio"] = 1.1
    c["qrg_hgs"]["stagnation_patience"] = 999
    variants["wo_qrg"] = c

    # w/o trigger：QRG 一直可触发
    c = clone_cfg(cfg)
    c["qrg_hgs"]["qrg_start_ratio"] = 0.0
    c["qrg_hgs"]["stagnation_patience"] = 999
    variants["wo_trigger"] = c

    frames = []
    best_map = {}
    for name, subcfg in variants.items():
        best, df = run_variant(name, subcfg, adapter)
        frames.append(df)
        best_map[name] = best

    master = pd.concat(frames, ignore_index=True)
    master.to_csv(out_dir / "ablation_raw.csv", index=False)
    save_json(out_dir / "ablation_best.json", best_map)

    summary = master.groupby("variant")["score"].max().reset_index().sort_values("score", ascending=False)
    summary.to_csv(out_dir / "ablation_summary.csv", index=False)
    print(summary)

if __name__ == "__main__":
    main()

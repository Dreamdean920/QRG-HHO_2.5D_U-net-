from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from utils import load_yaml, ensure_dir
from objective_adapter import ObjectiveAdapter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=["asha", "bohb"], required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    adapter = ObjectiveAdapter(cfg)
    out_dir = ensure_dir(Path(cfg["output_root"]) / "ray_tune")

    try:
        import ray
        from ray import tune
        from ray.tune import Tuner
        from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB
    except Exception as e:
        raise RuntimeError("请先安装 ray[tune]。") from e

    def trainable(config):
        run_name = f"{args.method}_{tune.get_context().get_trial_id()}"
        result = adapter.evaluate(
            params=config,
            seed=cfg["search"]["eval_seeds"][0],
            run_name=run_name,
            mode="short_eval",
        )
        score = float(result["score"]) if result["status"] == "OK" else -1e18
        tune.report(score=score, val_dice=result.get("val_dice", -1.0))

    space = {}
    for k, spec in cfg["space"].items():
        if spec["type"] == "float":
            if spec.get("log", False):
                space[k] = tune.loguniform(float(spec["low"]), float(spec["high"]))
            else:
                space[k] = tune.uniform(float(spec["low"]), float(spec["high"]))
        elif spec["type"] == "int":
            space[k] = tune.randint(int(spec["low"]), int(spec["high"]) + 1)
        elif spec["type"] == "categorical":
            space[k] = tune.choice(list(spec["choices"]))
        else:
            raise ValueError(spec["type"])

    if args.method == "asha":
        scheduler = ASHAScheduler(metric="score", mode="max")
        tuner = Tuner(
            trainable,
            param_space=space,
            tune_config=tune.TuneConfig(
                metric="score",
                mode="max",
                scheduler=scheduler,
                num_samples=int(cfg["search"]["n_trials"]),
            ),
            run_config=tune.RunConfig(name="asha_search", storage_path=str(out_dir)),
        )
    else:
        try:
            from ray.tune.search.bohb import TuneBOHB
            from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
        except Exception as e:
            raise RuntimeError("BOHB 需要安装 hpbandster 和 ConfigSpace。") from e

        cs = ConfigurationSpace()
        for k, spec in cfg["space"].items():
            if spec["type"] == "float":
                cs.add_hyperparameter(
                    UniformFloatHyperparameter(k, lower=float(spec["low"]), upper=float(spec["high"]), log=bool(spec.get("log", False)))
                )
            elif spec["type"] == "int":
                cs.add_hyperparameter(
                    UniformIntegerHyperparameter(k, lower=int(spec["low"]), upper=int(spec["high"]), log=False)
                )
            elif spec["type"] == "categorical":
                cs.add_hyperparameter(CategoricalHyperparameter(k, choices=list(spec["choices"])))

        bohb = TuneBOHB(space=cs)
        scheduler = HyperBandForBOHB(time_attr="training_iteration", metric="score", mode="max")
        tuner = Tuner(
            trainable,
            param_space=space,
            tune_config=tune.TuneConfig(
                metric="score",
                mode="max",
                search_alg=bohb,
                scheduler=scheduler,
                num_samples=int(cfg["search"]["n_trials"]),
            ),
            run_config=tune.RunConfig(name="bohb_search", storage_path=str(out_dir)),
        )

    results = tuner.fit()
    rows = []
    for r in results:
        row = dict(r.config)
        metrics = dict(r.metrics)
        row.update(metrics)
        row["method"] = args.method
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{args.method}_results.csv", index=False)
    print(df.sort_values("score", ascending=False).head(5))

if __name__ == "__main__":
    main()

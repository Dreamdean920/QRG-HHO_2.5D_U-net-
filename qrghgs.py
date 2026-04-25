from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np

# ⭐ 核心算法（你的最优版本）
from search.qrg_hgs_core import qrghgs_optimize


class AdaptiveQRGHGS:
    def __init__(self, cfg: Dict[str, Any], evaluate_fn, rng_seed: int = 3407) -> None:
        self.cfg = cfg
        self.space = cfg["space"]
        self.qcfg = cfg["qrg_hgs"]
        self.evaluate_fn = evaluate_fn
        self.rng_seed = rng_seed

        self.keys = list(self.space.keys())

    # ===============================
    # 1. 搜索空间 → 向量边界
    # ===============================
    def _space_to_bounds(self):
        lb, ub = [], []
        for k in self.keys:
            spec = self.space[k]

            if spec["type"] == "float":
                lb.append(float(spec["low"]))
                ub.append(float(spec["high"]))

            elif spec["type"] == "int":
                lb.append(float(spec["low"]))
                ub.append(float(spec["high"]))

            elif spec["type"] == "categorical":
                choices = list(spec["choices"])
                lb.append(0.0)
                ub.append(float(len(choices) - 1))

            else:
                raise ValueError(f"Unsupported space type: {spec['type']}")

        return np.array(lb, dtype=np.float64), np.array(ub, dtype=np.float64)

    # ===============================
    # 2. 向量 → 参数
    # ===============================
    def _vector_to_params(self, x: np.ndarray) -> Dict[str, Any]:
        params = {}

        for i, k in enumerate(self.keys):
            spec = self.space[k]
            v = float(x[i])

            if spec["type"] == "float":
                params[k] = v

            elif spec["type"] == "int":
                params[k] = int(round(v))

            elif spec["type"] == "categorical":
                choices = list(spec["choices"])
                idx = int(round(v))
                idx = max(0, min(len(choices) - 1, idx))
                params[k] = choices[idx]

            else:
                raise ValueError(f"Unsupported space type: {spec['type']}")

        return params

    # ===============================
    # 3. 主优化入口
    # ===============================
    def optimize(self, eval_seed: int = 42, method_name: str = "qrghgs") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:

        lb, ub = self._space_to_bounds()

        # ===== 包装目标函数 =====
        def objective_fn(x: np.ndarray, trial_counter: int) -> Dict[str, Any]:
            params = self._vector_to_params(x)

            run_name = f"{method_name}_trial{trial_counter:03d}"

            res = self.evaluate_fn(
                params=params,
                seed=eval_seed,
                run_name=run_name,
            )

            # ⭐ 防崩保护
            if res["status"] != "OK":
                res["fitness"] = 1e18

            # ⭐ 记录参数（用于后续分析）
            res["params"] = params
            res["trial_id"] = trial_counter

            return res

        # ===== 调用你的核心算法 =====
        result = qrghgs_optimize(
            objective_fn=objective_fn,
            dim=len(self.keys),
            lb=lb,
            ub=ub,

            pop_size=int(self.qcfg.get("population_size", 10)),
            max_iter=int(self.qcfg.get("max_iters", 10)),

            seed=int(self.cfg["search"].get("random_seed", self.rng_seed)),

            qrg_start_ratio=float(self.qcfg.get("qrg_start_ratio", 0.4)),
            theta_max=float(self.qcfg.get("qrg_theta_init", 0.3141592654)),
            theta_min=float(self.qcfg.get("qrg_theta_final", 0.0523598776)),

            qrg_on_stagnation_only=True,
            stag_patience=int(self.qcfg.get("stagnation_patience", 3)),

            verbose=True,
        )

        # ===== 输出结果 =====
        best_params = self._vector_to_params(result.best_position)

        return {
            "best_params": best_params,
            "best_score": -float(result.best_fitness),  # 转回“越大越好”
            "best_fitness": float(result.best_fitness),
            "history_best_fitness": result.history_best_fitness,
            "history_best_position": result.history_best_position,
        }, result.trial_records
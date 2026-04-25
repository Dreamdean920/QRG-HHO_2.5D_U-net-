import math
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Any
from hgs_core import _ensure_bounds, _compute_hunger, _compute_weights


@dataclass
class QRGHGSResult:
    best_position: np.ndarray
    best_fitness: float
    history_best_fitness: List[float]
    history_best_position: List[List[float]]
    trial_records: List[Dict[str, Any]]


def _qrg_update(x: np.ndarray, best: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                theta: float, rng: np.random.Generator) -> np.ndarray:
    ratio = (x - lb) / (ub - lb + 1e-12)
    ratio = np.clip(ratio, 0.0, 1.0)
    alpha = np.sqrt(ratio)
    beta = np.sqrt(1.0 - ratio)

    direction = np.sign(best - x)
    signed_theta = theta * direction

    alpha_new = alpha * np.cos(signed_theta) - beta * np.sin(signed_theta)
    beta_new = alpha * np.sin(signed_theta) + beta * np.cos(signed_theta)
    p = np.clip(alpha_new ** 2, 0.0, 1.0)

    candidate = lb + p * (ub - lb)
    mix = 0.7 + 0.3 * rng.random(len(x))
    out = mix * candidate + (1.0 - mix) * x
    return _ensure_bounds(out, lb, ub)


def qrghgs_optimize(
    objective_fn: Callable[[np.ndarray, int], Dict[str, Any]],
    dim: int,
    lb: np.ndarray,
    ub: np.ndarray,
    pop_size: int = 6,
    max_iter: int = 8,
    seed: int = 42,
    l_param: float = 0.08,
    qrg_start_ratio: float = 0.5,
    qrg_interval: int = 1,
    qrg_top_ratio: float = 0.34,
    theta_max: float = math.pi / 10,
    theta_min: float = math.pi / 60,
    verbose: bool = True,
) -> QRGHGSResult:
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)

    X = rng.uniform(lb, ub, size=(pop_size, dim))
    hunger = np.zeros(pop_size, dtype=np.float64)

    trial_records: List[Dict[str, Any]] = []
    history_best_fitness: List[float] = []
    history_best_position: List[List[float]] = []

    global_best_fit = float('inf')
    global_best_pos = X[0].copy()
    trial_counter = 0

    for t in range(max_iter):
        fitness = np.zeros(pop_size, dtype=np.float64)
        for i in range(pop_size):
            X[i] = _ensure_bounds(X[i], lb, ub)
            details = objective_fn(X[i].copy(), trial_counter)
            fit = float(details['fitness'])
            fitness[i] = fit
            rec = {'iter': t, 'pop_idx': i, 'trial_idx': trial_counter}
            rec.update(details)
            trial_records.append(rec)

            if fit < global_best_fit:
                global_best_fit = fit
                global_best_pos = X[i].copy()

            trial_counter += 1

        history_best_fitness.append(global_best_fit)
        history_best_position.append(global_best_pos.tolist())
        if verbose:
            print(f"[QRG-HGS] iter={t+1}/{max_iter} global_best_fitness={global_best_fit:.6f}")

        order = np.argsort(fitness)
        best_idx = int(order[0])
        best_fit = float(fitness[best_idx])
        Xb = X[best_idx].copy()

        hunger = _compute_hunger(fitness, hunger, lb, ub, rng)
        w1, w2 = _compute_weights(hunger, rng, l_param=l_param)
        shrink = 2.0 * (1.0 - (t / max(max_iter, 1)))

        X_new = X.copy()
        for i in range(pop_size):
            E = 2.0 / (math.exp(abs(float(fitness[i]) - best_fit)) + math.exp(-abs(float(fitness[i]) - best_fit)))
            R = 2.0 * shrink * rng.random(dim) - shrink
            r1 = rng.random()
            r2 = rng.random()
            if r1 < l_param:
                X_new[i] = X[i] * (1.0 + rng.normal(size=dim))
            else:
                delta = np.abs(Xb - X[i])
                if r2 > E:
                    X_new[i] = w1[i] * Xb + R * w2[i] * delta
                else:
                    X_new[i] = w1[i] * Xb - R * w2[i] * delta

            if i == best_idx:
                X_new[i] = Xb.copy()

        X = _ensure_bounds(X_new, lb, ub)

        start_iter = max(1, int(max_iter * qrg_start_ratio))
        if (t + 1) >= start_iter and ((t + 1 - start_iter) % max(qrg_interval, 1) == 0):
            theta = theta_max - (theta_max - theta_min) * ((t + 1) / max(max_iter, 1))
            topk = max(1, int(math.ceil(pop_size * qrg_top_ratio)))
            elite_ids = order[:topk]
            for idx in elite_ids:
                if idx == best_idx:
                    continue
                X[idx] = _qrg_update(X[idx], Xb, lb, ub, theta=theta, rng=rng)

    return QRGHGSResult(
        best_position=global_best_pos,
        best_fitness=global_best_fit,
        history_best_fitness=history_best_fitness,
        history_best_position=history_best_position,
        trial_records=trial_records,
    )

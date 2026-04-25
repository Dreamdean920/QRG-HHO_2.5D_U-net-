import math
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Any


@dataclass
class HGSResult:
    best_position: np.ndarray
    best_fitness: float
    history_best_fitness: List[float]
    history_best_position: List[List[float]]
    trial_records: List[Dict[str, Any]]


def _ensure_bounds(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lb), ub)


def _compute_hunger(fitness: np.ndarray, hunger: np.ndarray, lb: np.ndarray, ub: np.ndarray, rng: np.random.Generator,
                    hunger_floor_ratio: float = 0.03) -> np.ndarray:
    best = float(np.min(fitness))
    worst = float(np.max(fitness))
    span = float(np.mean(ub - lb))
    lower_hunger = hunger_floor_ratio * span
    new_hunger = hunger.copy()

    for i in range(len(fitness)):
        if abs(float(fitness[i]) - best) < 1e-12:
            new_hunger[i] = 0.0
            continue

        denom = max(worst - best, 1e-12)
        th = ((float(fitness[i]) - best) / denom) * rng.random() * 2.0 * span
        h = th if th >= lower_hunger else lower_hunger * (1.0 + rng.random())
        new_hunger[i] += h
    return new_hunger


def _compute_weights(hunger: np.ndarray, rng: np.random.Generator, l_param: float) -> Tuple[np.ndarray, np.ndarray]:
    n = len(hunger)
    sum_hunger = float(np.sum(hunger)) + 1e-12
    w1 = np.ones(n, dtype=np.float64)
    w2 = np.zeros(n, dtype=np.float64)

    for i in range(n):
        r3 = rng.random()
        r4 = rng.random()
        r5 = rng.random()
        if r3 < l_param:
            w1[i] = hunger[i] * n / sum_hunger * r4
        else:
            w1[i] = 1.0
        w2[i] = (1.0 - math.exp(-abs(float(hunger[i]) - sum_hunger))) * r5 * 2.0
    return w1, w2


def hgs_optimize(
    objective_fn: Callable[[np.ndarray, int], Dict[str, Any]],
    dim: int,
    lb: np.ndarray,
    ub: np.ndarray,
    pop_size: int = 6,
    max_iter: int = 8,
    seed: int = 42,
    l_param: float = 0.08,
    verbose: bool = True,
) -> HGSResult:
    """
    Pure HGS for minimization.

    objective_fn(pos, trial_idx) must return dict containing at least:
        {
            'fitness': float,
            'val_dice': float,
            'test_dice': float,
            'time_sec': float,
            ...
        }
    """
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
        details_list: List[Dict[str, Any]] = []

        for i in range(pop_size):
            X[i] = _ensure_bounds(X[i], lb, ub)
            details = objective_fn(X[i].copy(), trial_counter)
            fit = float(details['fitness'])
            fitness[i] = fit
            details_list.append(details)

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
            print(f"[HGS] iter={t+1}/{max_iter} global_best_fitness={global_best_fit:.6f}")

        best_idx = int(np.argmin(fitness))
        best_fit = float(fitness[best_idx])
        worst_fit = float(np.max(fitness))
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

    return HGSResult(
        best_position=global_best_pos,
        best_fitness=global_best_fit,
        history_best_fitness=history_best_fitness,
        history_best_position=history_best_position,
        trial_records=trial_records,
    )

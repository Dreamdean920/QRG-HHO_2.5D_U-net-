import math
from dataclasses import dataclass
from typing import Callable, Dict, Any, List

import numpy as np


@dataclass
class SearchResult:
    best_position: np.ndarray
    best_fitness: float
    history_best_fitness: List[float]
    history_best_position: List[List[float]]
    trial_records: List[Dict[str, Any]]


def _ensure_bounds(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return np.clip(x, lb, ub)


def _initialize_population(pop_size: int, dim: int, lb: np.ndarray, ub: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return lb + (ub - lb) * rng.random((pop_size, dim))


def _evaluate_population(
    pop: np.ndarray,
    objective_fn: Callable[[np.ndarray, int], Dict[str, Any]],
    trial_counter: int,
) -> (np.ndarray, List[Dict[str, Any]], int):
    fitness = np.zeros(pop.shape[0], dtype=np.float64)
    records = []
    for i in range(pop.shape[0]):
        trial_counter += 1
        rec = objective_fn(pop[i].copy(), trial_counter)
        fitness[i] = float(rec["fitness"])
        records.append(rec)
    return fitness, records, trial_counter


def qrghgs_optimize(
    objective_fn: Callable[[np.ndarray, int], Dict[str, Any]],
    dim: int,
    lb: np.ndarray,
    ub: np.ndarray,
    pop_size: int = 6,
    max_iter: int = 2,
    seed: int = 42,
    qrg_start_ratio: float = 0.5,
    qrg_interval: int = 1,
    qrg_top_ratio: float = 0.34,
    theta_max: float = math.pi / 10,
    theta_min: float = math.pi / 60,
    verbose: bool = True,
    # 新增：停滞触发式 QRG
    qrg_on_stagnation_only: bool = True,
    stag_patience: int = 2,
    stag_tol: float = 1e-4,
    # 新增：只精修连续维度
    qrg_optimize_dims: List[int] = None,
    freeze_batch_dim: bool = True,
) -> SearchResult:
    """
    改进版 QRG-HGS：
    1) HGS 负责主体全局搜索
    2) QRG 在后期 + 停滞时触发
    3) QRG 默认只优化前两维（lr, dice_weight）
    """

    rng = np.random.default_rng(seed)

    if qrg_optimize_dims is None:
        if freeze_batch_dim and dim >= 3:
            qrg_optimize_dims = [0, 1]
        else:
            qrg_optimize_dims = list(range(dim))

    pop = _initialize_population(pop_size, dim, lb, ub, rng)

    trial_counter = 0
    trial_records: List[Dict[str, Any]] = []

    # 初始评估
    fitness, records, trial_counter = _evaluate_population(pop, objective_fn, trial_counter)
    trial_records.extend(records)

    best_idx = int(np.argmin(fitness))
    best_position = pop[best_idx].copy()
    best_fitness = float(fitness[best_idx])

    history_best_fitness = [best_fitness]
    history_best_position = [best_position.tolist()]

    prev_best_fitness = best_fitness
    stagnation_count = 0

    qrg_start_iter = max(1, int(np.ceil(max_iter * qrg_start_ratio)))

    if verbose:
        print("\n===== QRG-HGS INITIALIZED =====")
        print(f"pop_size={pop_size}, max_iter={max_iter}, dim={dim}")
        print(f"best_init_fitness={best_fitness:.8f}")
        print(f"qrg_start_iter={qrg_start_iter}")
        print(f"qrg_on_stagnation_only={qrg_on_stagnation_only}")
        print(f"stag_patience={stag_patience}, stag_tol={stag_tol}")
        print(f"qrg_optimize_dims={qrg_optimize_dims}")
        print("=" * 60)

    # 主循环
    for it in range(1, max_iter + 1):
        if verbose:
            print(f"\n========== ITER {it}/{max_iter} ==========")
            print(f"[ITER] current best fitness = {best_fitness:.8f}")

        # 根据 HGS 风格做主体更新
        new_pop = pop.copy()

        worst_fitness = float(np.max(fitness))
        shrink = 2.0 * (1.0 - it / max(max_iter, 1))

        for i in range(pop_size):
            Xi = pop[i].copy()

            # 如果是当前 best，可小幅扰动避免完全不动
            if i == best_idx:
                noise = 0.01 * (ub - lb) * rng.normal(size=dim)
                cand = Xi + noise
                cand = _ensure_bounds(cand, lb, ub)
                new_pop[i] = cand
                continue

            # HGS 的核心量：E, R, W1, W2
            # 这里是简化、工程友好的实现
            Fi = float(fitness[i])
            BF = float(best_fitness)
            WF = float(worst_fitness)

            # E: 与当前最优差距决定 exploitation / exploration 倾向
            E = 2.0 / (math.exp(abs(Fi - BF)) + math.exp(-abs(Fi - BF)))

            # R: 随迭代缩小
            R = 2.0 * shrink * rng.random(dim) - shrink

            # hunger ratio / weight
            if WF - BF < 1e-12:
                hunger_ratio = 0.0
            else:
                hunger_ratio = (Fi - BF) / (WF - BF + 1e-12)

            W1 = 1.0 if rng.random() > 0.5 else (1.0 + hunger_ratio * rng.random())
            W2 = (1.0 - math.exp(-abs(hunger_ratio))) * rng.random() * 2.0

            r1 = rng.random()
            r2 = rng.random()

            # HGS 三种更新模式（工程简化版）
            if r1 < 0.3:
                # game1: 自身附近随机搜索
                cand = Xi * (1.0 + rng.normal(0, 0.05, size=dim))
            else:
                diff = np.abs(best_position - Xi)
                if r2 > E:
                    # game2
                    cand = W1 * best_position + R * W2 * diff
                else:
                    # game3
                    cand = W1 * best_position - R * W2 * diff

            # 加一点小扰动，增强多样性
            cand += 0.01 * (ub - lb) * rng.normal(size=dim)
            cand = _ensure_bounds(cand, lb, ub)
            new_pop[i] = cand

        # 评估 HGS 更新后的群体
        new_fitness, records, trial_counter = _evaluate_population(new_pop, objective_fn, trial_counter)
        trial_records.extend(records)

        # 贪心保留
        improved_mask = new_fitness < fitness
        pop[improved_mask] = new_pop[improved_mask]
        fitness[improved_mask] = new_fitness[improved_mask]

        # 更新全局最优
        curr_best_idx = int(np.argmin(fitness))
        curr_best_fitness = float(fitness[curr_best_idx])
        curr_best_position = pop[curr_best_idx].copy()

        improvement = prev_best_fitness - curr_best_fitness
        if improvement < stag_tol:
            stagnation_count += 1
        else:
            stagnation_count = 0

        prev_best_fitness = curr_best_fitness

        if curr_best_fitness < best_fitness:
            best_fitness = curr_best_fitness
            best_position = curr_best_position.copy()
            best_idx = curr_best_idx

        if verbose:
            print(f"[ITER] after HGS best fitness = {best_fitness:.8f}")
            print(f"[ITER] stagnation_count = {stagnation_count}")

        # ========= QRG 触发逻辑 =========
        do_qrg = False
        if it >= qrg_start_iter:
            if qrg_on_stagnation_only:
                if stagnation_count >= stag_patience:
                    do_qrg = True
            else:
                if it % qrg_interval == 0:
                    do_qrg = True

        if do_qrg:
            if verbose:
                print("[QRG] triggered.")
            top_k = max(1, int(np.ceil(pop_size * qrg_top_ratio)))
            elite_idx = np.argsort(fitness)[:top_k]

            progress = (it - 1) / max(max_iter - 1, 1)
            theta_t = theta_max - (theta_max - theta_min) * progress

            qrg_pop = pop.copy()

            for idx in elite_idx:
                Xi = pop[idx].copy()
                Xnew = Xi.copy()

                for d in qrg_optimize_dims:
                # 论文增强版：维度自适应随机旋转强度
                # lr 更敏感，不再固定压死，而是在较小范围内自适应浮动
                # dice_weight 保持正常旋转强度
                # batch_size 维度继续冻结不动
                    if d == 0:      # lr
                        theta_use = theta_t * (0.75 + 0.20 * rng.random())   # [0.75, 0.95] * theta_t
                        step_ratio = 0.12 + 0.02 * rng.random()              # [0.12, 0.14]
                    elif d == 1:    # dice_weight
                        theta_use = theta_t * (0.95 + 0.10 * rng.random())   # [0.95, 1.05] * theta_t
                        step_ratio = 0.14 + 0.02 * rng.random()              # [0.14, 0.16]
                    else:
                        continue

                    direction = 1.0 if best_position[d] >= Xi[d] else -1.0
                    step_scale = math.sin(theta_use) * (ub[d] - lb[d]) * step_ratio
                    perturb = direction * step_scale * (0.5 + rng.random())

                    Xnew[d] = Xi[d] + perturb
                qrg_pop[idx] = Xnew

            # 评估 QRG 候选
            qrg_fitness, records, trial_counter = _evaluate_population(qrg_pop[elite_idx], objective_fn, trial_counter)
            trial_records.extend(records)

            # 仅替换 elite 对应位置
            for j, idx in enumerate(elite_idx):
                if qrg_fitness[j] < fitness[idx]:
                    pop[idx] = qrg_pop[idx]
                    fitness[idx] = qrg_fitness[j]

            # QRG 后重新更新全局最优
            curr_best_idx = int(np.argmin(fitness))
            curr_best_fitness = float(fitness[curr_best_idx])
            curr_best_position = pop[curr_best_idx].copy()

            if curr_best_fitness < best_fitness:
                best_fitness = curr_best_fitness
                best_position = curr_best_position.copy()
                best_idx = curr_best_idx

            if verbose:
                print(f"[QRG] after QRG best fitness = {best_fitness:.8f}")

            # 触发过一次后清零停滞计数
            stagnation_count = 0

        history_best_fitness.append(best_fitness)
        history_best_position.append(best_position.tolist())

        if verbose:
            print(f"[ITER] final best fitness = {best_fitness:.8f}")

    return SearchResult(
        best_position=best_position.copy(),
        best_fitness=float(best_fitness),
        history_best_fitness=history_best_fitness,
        history_best_position=history_best_position,
        trial_records=trial_records,
    )
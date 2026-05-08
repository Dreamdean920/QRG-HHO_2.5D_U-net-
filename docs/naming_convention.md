# 主线与失败线命名约定

本文件根据 `QRG_HGS_UNet_代码与实验结果整合说明` 记录当前项目命名。

## 主线

- 结果目录：`outputs/paper_main_structv3_real_benchmark`
- 真实对比入口：`experiments/real_compare/run_paper_main_real_benchmark.py`
- Struct-v3 主方法：`experiments/real_compare/run_main_qrghgs_struct_v3.py`
- Top3 confirm：`experiments/confirm/confirm_paper_main_top3_multiseed.py`

主方法仍使用结果表中的 method 名 `qrghgs_struct_v3`，这样可以继续复用已有 `trial_json`、`train_runs` 和 `compare_raw.csv`。

## 分析

- Trial 预算：`tools/analysis/analyze_completed_methods_convergence_budget.py`
- 时间预算：`tools/analysis/analyze_time_completed_methods.py`
- Trial 输出目录：`analysis_trial_budget_completed_methods`
- 时间输出目录：`analysis_time_budget_completed_methods`

## 失败线

失败线不进入主结果表，只作为备份、消融或历史探索保留。

| 原始方向 | 标志性定位 |
|---|---|
| `qrghgs_struct_v2` | `failed_struct_v2_explore_heavy`，探索过强 |
| `qrghgs_struct_cons` | `failed_struct_cons_over_conservative`，过度保守 |
| `qrghgs_struct_bal` | `failed_struct_bal_unstable_balance`，平衡不稳定 |
| `qrghgs_struct_v3_fitness` | `failed_structv3_fitness_bias`，fitness 改动未改善泛化 |
| `qrghgs_struct_v4_memory` | `failed_v4_memory_archive_overfit`，档案记忆未超过主线 |
| `qrghgs_hybrid_topk_light` | `failed_hybrid_topk_light_underpowered`，轻量 top-k 不足 |
| tuned / hybrid tuned | `failed_hybrid_tuned_no_gain`，调参未超过 Struct-v3 |

历史代理模型脚本已改名为 `experiments/surrogate/failed_surrogate_qrghgs_*`，表示它们是归档线而不是当前可运行主线。

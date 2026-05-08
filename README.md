# B_Project_QRG_UNet

本项目围绕 `2.5D U-Net` 肺部感染分割实验，核心任务是比较不同超参数搜索方法在真实训练下的表现。当前最应该作为论文和汇报主线的结果目录是：

`outputs/paper_main_structv3_real_benchmark`

该目录是按整合文档重新命名后的主线结果归档，实际包含 8 个方法/版本：`random`、`tpe`、`pso`、`ga`、`hgs`、`qrghgs_baseline`、`qrghgs_hybrid`、`qrghgs_struct_v3`。每个方法均有 30 次成功 trial。

## 当前主线结果

主结果文件：

- `outputs/paper_main_structv3_real_benchmark/compare_summary.csv`：方法级汇总表
- `outputs/paper_main_structv3_real_benchmark/compare_raw.csv`：每个 trial 的原始记录
- `outputs/paper_main_structv3_real_benchmark/analysis_trial_budget_completed_methods/`：收敛、小预算、排名图表
- `outputs/paper_main_structv3_real_benchmark/analysis_time_budget_completed_methods/`：时间效率分析图表
- `outputs/paper_main_structv3_real_benchmark/train_runs/`：真实训练输出与 checkpoint
- `outputs/paper_main_structv3_real_benchmark/trial_json/`：每次 trial 的 JSON 结果

从当前 `compare_summary.csv` 和排名表看：

| 方法 | 成功 trial | best val Dice | best test Dice | mean test Dice | 平均单次时间 |
|---|---:|---:|---:|---:|---:|
| `ga` | 30 | 0.964454 | 0.963712 | 0.958018 | 0.704 h |
| `pso` | 30 | 0.963867 | 0.961313 | 0.957691 | 0.488 h |
| `qrghgs_struct_v3` | 30 | **0.967263** | **0.963958** | 0.957388 | 0.560 h |
| `qrghgs_hybrid` | 30 | 0.965078 | 0.963023 | 0.957363 | 0.803 h |
| `random` | 30 | 0.965168 | 0.962080 | 0.957300 | 0.783 h |
| `tpe` | 30 | 0.964760 | 0.961081 | 0.957194 | 0.514 h |
| `qrghgs_baseline` | 30 | 0.963337 | 0.961255 | 0.957121 | **0.405 h** |
| `hgs` | 30 | 0.964982 | 0.961883 | 0.954911 | 0.523 h |

可以这样概括主线结论：

- `qrghgs_struct_v3` 取得最高验证集 Dice 和最高测试集 Dice，适合作为 QRG-HGS 改进版的主结果。
- `ga` 的平均测试 Dice 最高，适合作为强基线对照。
- `qrghgs_baseline` 平均单次训练时间最短、稳定性排名最好，但峰值指标低于 `qrghgs_struct_v3`。
- `hgs` 平均测试表现最低且波动较大，可作为说明 QRG 改进必要性的对照。

## 项目目录

整理后的目录职责如下：

```text
B_Project_QRG_UNet/
├── configs/                  # 数据、训练、搜索配置
├── data/                     # split 文本与预览图
├── datasets/                 # 2D slice 数据集封装
├── docs/                     # 阶段性文档
├── experiments/              # 实验编排入口，按任务分类
│   ├── confirm/              # Top-K / 多 seed 确认实验
│   ├── real_compare/         # 真实训练方法对比，当前主线在这里
│   ├── runners/              # ablation、Ray Tune 等通用 runner
│   └── surrogate/            # 历史代理模型实验脚本
├── models/                   # U-Net 模型结构
├── outputs/                  # 实验输出、图表、checkpoint、trial JSON
├── results/                  # 数据检查、slice 索引、baseline 汇总
├── scripts/                  # 数据处理、训练、测试、推理可视化脚本
├── search/                   # HGS / QRG-HGS 底层搜索实现
├── tools/                    # 分析、结果合并、维护清理脚本
│   ├── analysis/             # 收敛、预算、时间效率、汇总分析
│   ├── maintenance/          # 清理、恢复、删除低质量方法
│   └── results/              # warmup 与结果合并工具
├── utils/                    # loss、metric、后处理
├── exp_utils.py              # 配置、随机种子、搜索空间采样等通用函数
├── objective_adapter.py      # 搜索算法调用训练脚本的适配层
└── qrghgs.py                 # Adaptive QRG-HGS 封装
```

根目录现在只保留项目级文件和少量核心模块。实验入口不再散在根目录，而是放进 `experiments/`；结果分析和清理脚本放进 `tools/`。

## 命名规则

本轮整理按 `QRG_HGS_UNet_代码与实验结果整合说明` 中的定位重命名：

| 定位 | 新名称 | 含义 |
|---|---|---|
| 主线结果 | `outputs/paper_main_structv3_real_benchmark` | 论文主线真实训练 benchmark |
| 主线搜索脚本 | `experiments/real_compare/run_paper_main_real_benchmark.py` | Random/TPE/PSO/GA/HGS/QRG-HGS 的完整真实对比 |
| Struct-v3 主方法脚本 | `experiments/real_compare/run_main_qrghgs_struct_v3.py` | 冻结的 Local + Top-k + Global restart 分层扰动 QRG-HGS |
| Struct-v3 探索脚本 | `experiments/real_compare/run_explore_structv3_ablation_directions.py` | elite、trigger、region、dual-branch 等探索方向 |
| Top3 确认脚本 | `experiments/confirm/confirm_paper_main_top3_multiseed.py` | 主线 Top3 多 seed confirm |
| Trial 预算分析 | `tools/analysis/analyze_completed_methods_convergence_budget.py` | 只分析跑满 30 trial 的方法 |
| 时间预算分析 | `tools/analysis/analyze_time_completed_methods.py` | 只分析跑满方法的时间效率 |
| 失败线归档脚本 | `tools/maintenance/archive_failed_lines_from_main_results.py` | 将失败/低质量方法从主结果中移出或备份 |
| 失败线清理脚本 | `tools/maintenance/clean_failed_qrghgs_lines.py` | 清理低质量 QRG-HGS 结果 |
| 历史失败代理线 | `experiments/surrogate/failed_surrogate_qrghgs_*` | 代理模型路线的历史探索归档 |

失败线在文档中对应 `struct_v2`、`struct_cons`、`struct_bal`、`v3_fitness`、`v4_memory`、`hybrid_topk_light`、`tuned` 等方向。当前主结果 CSV 已经清理为 8 个跑满方法，因此这些失败线主要保留在备份、维护脚本和历史探索脚本中，不进入主结果表。

## 代码说明

### 训练与数据

- `scripts/train_week3_unet.py`：当前主要真实训练入口。支持 `--lr`、`--dice_weight`、`--batch_size`、`--epochs`、`--seed`、`--run_name`、`--output_json` 等参数，搜索脚本都会调用它。
- `scripts/train_week2_unet.py`：早期训练版本，保留用于对照。
- `scripts/train_baseline.py`、`scripts/test_baseline.py`：baseline 训练与测试。
- `scripts/build_slice_index.py`、`scripts/check_data_nii.py`、`scripts/make_splits_nii.py`、`scripts/preview_slices.py`：数据划分、检查、slice 索引和预览。
- `datasets/dataset_2d.py`：读取 NIfTI 数据并构造 2D slice dataset。
- `models/unet_2d.py`：标准 2D U-Net 模型。
- `utils/losses.py`、`utils/metrics.py`、`utils/postprocess.py`：损失、Dice/IoU 指标和预测后处理。

### 搜索算法

- `search/hgs_core.py`：基础 HGS 搜索实现。
- `search/qrg_hgs_core.py`：QRG-HGS 核心实现。
- `search/qrghgs_dimtheta.py`：带动态角度参数的 QRG-HGS 版本。
- `search/search_hgs.py`、`search/search_hgs_dimtheta.py`：较早的命令行搜索入口。
- `qrghgs.py`：面向配置文件和 `ObjectiveAdapter` 的 Adaptive QRG-HGS 封装。
- `objective_adapter.py`：把搜索算法给出的超参数转成训练命令，并读取训练 JSON 作为 fitness。

### 当前主线实验

- `experiments/real_compare/run_paper_main_real_benchmark.py`：当前真实训练多方法对比主入口，输出到 `outputs/paper_main_structv3_real_benchmark`。
- `experiments/real_compare/run_explore_structv3_ablation_directions.py`：围绕 `qrghgs_struct_v3` 的 5 个方向扩展实验。
- `experiments/real_compare/run_main_qrghgs_struct_v3.py`：结构化扰动版 QRG-HGS 实验脚本。
- `experiments/real_compare/run_compare_random_tpe_hgs.py`、`run_compare_all_methods.py`：较早的真实训练对比入口。
- `experiments/real_compare/run_final_compare_7methods.py`：历史最终对比入口，依赖当前缺失的 `surrogate_model_checked.py`，暂时视为历史脚本。

### 确认实验

- `experiments/confirm/confirm_paper_main_top3_multiseed.py`：从当前结果中选择 Top-K 候选并做多 seed 确认。
- `experiments/confirm/run_confirm_top3.py`：手工指定 Top3 参数的确认入口。
- `experiments/confirm/multiseed_confirm.py`：基于配置文件的多 seed 确认 runner。

### 分析与图表

- `tools/analysis/analyze_completed_methods_convergence_budget.py`：读取 `outputs/paper_main_structv3_real_benchmark/compare_raw.csv`，生成当前主线收敛曲线、预算对比和排名表。
- `tools/analysis/analyze_time_completed_methods.py`：生成时间预算、time-to-target、时间-性能权衡图表。
- `tools/analysis/analyze_small_budget.py`、`plot_all_methods_convergence.py`：早期 `week4_merged` 结果分析脚本。
- `tools/analysis/summarize_results.py`：通用 CSV 汇总工具。

### 结果维护

- `tools/maintenance/clean_failed_qrghgs_lines.py`：安全清理低质量 QRG-HGS 方法，并备份原文件。
- `tools/maintenance/archive_failed_lines_from_main_results.py`：归档指定失败线方法的历史清理脚本。
- `tools/maintenance/restore_failed_line_json_from_backup.py`：从 raw backup 恢复 JSON。
- `tools/results/merge_week4_qrghgs_with_week4_compare.py`、`merge_warmup_into_history.py`、`warmup_batch_samples.py`：早期结果合并和 warmup 工具。

### 历史代理模型脚本

`experiments/surrogate/` 中的脚本用于早期代理模型对比，例如 QRG-HGS 改进前后、十个版本对比。但当前工作区里 `surrogate_model.py` 和 `surrogate_model_checked.py` 处于已删除状态，因此这些脚本暂时不能直接运行。若后续要恢复代理模型主线，需要先恢复或重写这两个模块。

## 怎么从这里入手

建议按这个顺序看：

1. 先看 `outputs/paper_main_structv3_real_benchmark/compare_summary.csv`，确认各方法最终表现。
2. 再看 `outputs/paper_main_structv3_real_benchmark/analysis_trial_budget_completed_methods/method_rank_completed.csv`，确认 best、mean、stability 的排名。
3. 看 `tools/analysis/analyze_completed_methods_convergence_budget.py` 和 `tools/analysis/analyze_time_completed_methods.py`，理解主线图表如何生成。
4. 看 `experiments/real_compare/run_paper_main_real_benchmark.py`，理解每种搜索方法如何调用真实训练。
5. 最后看 `scripts/train_week3_unet.py`，理解每个 trial 的训练、验证、测试指标来自哪里。

所有命令建议在项目根目录运行，例如：

```bash
python experiments/real_compare/run_paper_main_real_benchmark.py
python tools/analysis/analyze_completed_methods_convergence_budget.py
python tools/analysis/analyze_time_completed_methods.py
```

如果只想复现分析图表，不需要重新训练模型，直接运行两个 `tools/analysis` 脚本即可，它们会读取已有的 `compare_raw.csv`。

## 依赖

基础依赖：

```bash
pip install -r requirements.txt
pip install torch torchvision scipy optuna
```

`requirements.txt` 中未写入 `torch`、`torchvision`、`scipy`、`optuna`，但当前训练和搜索脚本会用到它们。

## 输出目录阅读方式

- `outputs/paper_main_structv3_real_benchmark`：当前主线，优先阅读。
- `outputs/week4_compare`、`outputs/week4_merged`：早期 Random/TPE/HGS/QRG-HGS 对比与合并结果。
- `outputs/qrghgs_10_versions`、`outputs/qrghgs_before_after`：历史代理模型版本分析结果。
- `outputs/batch_warmup`、`outputs/history_augmented`：早期代理模型 warmup 与历史表。
- `_backup_safe_clean_...`：清理脚本生成的备份，不作为主结果引用。

后续写论文或报告时，建议把 `outputs/paper_main_structv3_real_benchmark` 作为主线结果来源，其它 `outputs` 目录只作为历史探索或补充实验。

# B_Project_QRG_UNet

本项目用于 `2.5D U-Net` 肺部感染分割实验中的超参数搜索与对比，核心目标是比较 `Random`、`TPE`、`HGS`、`QRG-HGS`，并在此基础上加入**代理模型（surrogate model）**来减少真实训练次数、扩大搜索空间、提高候选筛选效率。

目前仓库中已经完成的重点内容包括：

- 2.5D U-Net 训练与评估主流程
- QRG-HGS / HGS / Random / TPE 等搜索方法
- 代理模型建立：`RandomForest` 集成回归器 + 不确定性估计
- 搜索空间扩大：不再只固定在 `batch_size=4`，而是支持 `batch_size in [2, 4, 8]`
- warmup 预热样本补齐与历史记录合并
- 代理空间下的多方法对比
- QRG-HGS 改进前后对比
- QRG-HGS 十个版本对比
- 小预算分析、收敛曲线绘图、Top3 多 seed 确认

## 1. 项目里这些方法分别有什么用

### 1.1 真实训练入口

核心训练脚本是：

- `scripts/train_week3_unet.py`

它的作用是：

- 按给定超参数训练 `2.5D U-Net`
- 在验证集上选择最优 checkpoint
- 在测试集上输出分割指标
- 兼容搜索脚本需要的参数格式，如 `--lr`、`--dice_weight`、`--batch_size`、`--run_name`、`--output_json`

它会生成：

- 每次运行对应的模型目录
- 训练日志与中间结果
- 一个 `json` 结果文件，常见字段包括：
  - `best_val_dice`
  - `test_dice`
  - `test_iou`
  - `test_sens`
  - `test_spec`
  - `time_sec`
  - `status`
  - `best_ckpt_path`

### 1.2 代理模型有什么用

相关文件：

- `surrogate_model.py`
- `surrogate_model_checked.py`

代理模型的作用是：

- 先读取已有历史实验结果
- 学习超参数到 `best_val_dice` 的映射关系
- 对大量候选点进行快速打分，而不是每个点都真实训练
- 同时给出预测均值和不确定性
- 使用 `acq = pred_mean + beta * pred_std` 挑选“看起来好且值得探索”的点

这里采用的是：

- `RandomForestRegressor` 集成
- 多个 bootstrap 子模型构造近似不确定性

`surrogate_model_checked.py` 比基础版多了一层检查：

- 检查历史数据在不同 `batch_size` 上是否覆盖充分
- 提醒哪些 batch 样本太少，避免代理模型偏向单一 batch

### 1.3 扩大搜索空间有什么用

仓库前期很多对比是为了公平，常常固定：

- `batch_size = 4`

后续加入代理模型后，搜索空间被扩大为：

- `lr`
- `dice_weight`
- `batch_size in [2, 4, 8]`

这样做的意义是：

- 让搜索不只在单一 batch 上做局部优化
- 让 QRG-HGS 和其他方法有机会发现更优的训练配置
- 更接近真实“自动调参”场景

为了避免代理模型只见过 `batch=4` 而导致预测失真，项目中又加入了：

- `warmup_batch_samples.py`
- `merge_warmup_into_history.py`

也就是先补少量 `batch=2` 和 `batch=8` 的真实样本，再和历史结果合并。

## 2. 推荐阅读的主线

如果你是第一次看这个仓库，建议按下面顺序理解：

1. `scripts/train_week3_unet.py`
2. `run_compare_random_tpe_hgs.py`
3. `merge_week4_qrghgs_with_week4_compare.py`
4. `warmup_batch_samples.py`
5. `merge_warmup_into_history.py`
6. `run_surrogate_search.py`
7. `run_surrogate_loop.py`
8. `run_surrogate_multi_methods.py`
9. `compare_qrghgs_before_after_surrogate.py`
10. `compare_qrghgs_10_versions_surrogate.py`

## 3. 怎么运行，以及每一步会生成什么

### 3.1 安装依赖

先安装基础依赖：

```bash
pip install -r requirements.txt
pip install scipy torch torchvision
```

如果你要运行 TPE，对应还需要：

```bash
pip install optuna
```

## 4. 基础真实搜索与方法对比

### 4.1 运行 Random / TPE / HGS 三方法对比

```bash
python run_compare_random_tpe_hgs.py
```

作用：

- 在统一协议下运行 `Random`、`TPE`、`HGS`
- 使用真实训练进行评估
- 作为 QRG-HGS 的基线对照

主要输出：

- `outputs/week4_compare/compare_raw.csv`
- `outputs/week4_compare/compare_summary.csv`
- `outputs/week4_compare/trial_json/*.json`
- `outputs/week4_compare/train_runs/...`

其中：

- `compare_raw.csv` 记录每一次 trial 的详细结果
- `compare_summary.csv` 汇总每种方法的最优值、均值、方差等

### 4.2 如果想跑更完整的多方法真实对比

```bash
python run_compare_all_methods.py
```

作用：

- 在真实训练空间里运行更多方法
- 用统一预算做更完整的方法对比

主要输出：

- `outputs/week4_compare_all/compare_raw.csv`
- `outputs/week4_compare_all/compare_summary.csv`
- `outputs/week4_compare_all/trial_json/*.json`

## 5. 合并 QRG-HGS 与基线结果

### 5.1 合并 week4 的 QRG-HGS 和 week4_compare 的基线结果

```bash
python merge_week4_qrghgs_with_week4_compare.py
```

作用：

- 从两个目录中提取已经完成的真实实验结果
- 统一字段格式
- 合并成一份总表，便于后续画图和预算分析

主要输出：

- `outputs/week4_merged/qrghgs_from_week4.csv`
- `outputs/week4_merged/baseline_from_week4_compare.csv`
- `outputs/week4_merged/all_methods_merged.csv`
- `outputs/week4_merged/all_methods_summary.csv`

## 6. 代理模型与扩大搜索空间

### 6.1 先补 warmup 样本

```bash
python warmup_batch_samples.py
```

作用：

- 真实训练少量 `batch=2` 和 `batch=8` 的样本
- 给代理模型补齐跨 batch 的历史数据
- 降低“只见过 batch=4”带来的偏置

主要输出：

- `outputs/batch_warmup/warmup_batch_results.csv`
- `outputs/batch_warmup/*.json`
- `outputs/batch_warmup/train_runs/...`

### 6.2 把 warmup 样本并入历史总表

```bash
python merge_warmup_into_history.py
```

作用：

- 把 `warmup_batch_results.csv` 合并进已有历史结果
- 形成代理模型训练使用的增强历史表

主要输出：

- `outputs/history_augmented/all_methods_merged_plus_warmup.csv`

这份文件很关键，后面的代理搜索基本都依赖它。

## 7. 代理搜索

### 7.1 单轮代理搜索

```bash
python run_surrogate_search.py
```

作用：

- 用历史结果训练代理模型
- 在代理空间中随机采样大量候选点
- 用 `pred_mean + beta * pred_std` 选出最值得真实训练的少量点
- 再把这些候选送去真实训练验证

主要输出：

- `outputs/surrogate_search/selected_candidates.csv`
- `outputs/surrogate_search/real_eval_results.csv`
- `outputs/surrogate_search/*.json`
- `outputs/surrogate_search/train_runs/...`

可以这样理解：

- `selected_candidates.csv` 是代理模型挑出来的“推荐参数”
- `real_eval_results.csv` 是这些推荐参数真实训练后的结果

### 7.2 多轮迭代代理搜索

```bash
python run_surrogate_loop.py
```

作用：

- 代理模型搜索一轮
- 真实训练 top 候选
- 把新结果加入历史
- 再重新训练代理模型进入下一轮

这相当于一个简化版的“主动学习式调参”。

主要输出：

- `outputs/surrogate_loop/history_updated.csv`
- `outputs/surrogate_loop/all_new_results.csv`
- `outputs/surrogate_loop/train_runs/...`

其中：

- `history_updated.csv` 是每轮滚动更新后的历史表
- `all_new_results.csv` 是多轮新增真实训练结果的汇总

## 8. 代理空间下的多方法对比

### 8.1 比较 Random / TPE / PSO / GA / HGS / QRG-HGS

```bash
python run_surrogate_multi_methods.py
```

作用：

- 在同一个代理模型上运行多种搜索方法
- 每种方法先在代理空间内搜索
- 再选出 top-k 候选做真实训练
- 比较不同方法在代理框架下的效果

主要输出：

- `outputs/surrogate_multi_methods/*_surrogate_search.csv`
- `outputs/surrogate_multi_methods/*_topk_candidates.csv`
- `outputs/surrogate_multi_methods/*_real_eval.csv`
- `outputs/surrogate_multi_methods/all_methods_real_eval.csv`

这些文件的含义分别是：

- `*_surrogate_search.csv`：某方法在代理空间里尝试过哪些点
- `*_topk_candidates.csv`：某方法最终挑出的候选参数
- `*_real_eval.csv`：这些候选做真实训练后的结果
- `all_methods_real_eval.csv`：所有方法的真实评估总表

## 9. QRG-HGS 改进版本对比

### 9.1 比较改进前与改进后

```bash
python compare_qrghgs_before_after_surrogate.py
```

作用：

- 在同一代理模型下比较 `before` 和 `after` 两个 QRG-HGS 版本
- 关注 QRG 提前触发、停滞触发、扰动强度增强后是否更有效

主要输出：

- `outputs/qrghgs_before_after/qrghgs_before_surrogate_search.csv`
- `outputs/qrghgs_before_after/qrghgs_after_surrogate_search.csv`
- `outputs/qrghgs_before_after/qrghgs_before_real_eval.csv`
- `outputs/qrghgs_before_after/qrghgs_after_real_eval.csv`
- `outputs/qrghgs_before_after/qrghgs_before_after_summary.csv`

### 9.2 比较 10 个 QRG-HGS 版本

```bash
python compare_qrghgs_10_versions_surrogate.py
```

作用：

- 对多个版本的 `QRG-HGS` 做系统比较
- 比较是否启用 perturb
- 比较 QRG 启动时机、停滞耐心值、elite 比例、restart 比例
- 自动支持断点续跑

主要输出：

- `outputs/qrghgs_10_versions/*_surrogate_search.csv`
- `outputs/qrghgs_10_versions/*_topk_candidates.csv`
- `outputs/qrghgs_10_versions/*_real_eval.csv`
- `outputs/qrghgs_10_versions/all_surrogate_search.csv`
- `outputs/qrghgs_10_versions/all_real_eval.csv`
- `outputs/qrghgs_10_versions/summary.csv`

这个脚本适合写论文里的“版本演化”或“设计选择分析”部分。

## 10. Top3 确认实验

### 10.1 对选出的 3 组最好参数做多 seed 验证

```bash
python run_confirm_top3.py
```

作用：

- 对当前手工指定的 3 组候选参数做 `5 seeds` 复现
- 评估最优参数是否稳定，而不是偶然跑得好

主要输出：

- `outputs/week4/confirm_top3_raw.csv`
- `outputs/week4/confirm_top3_summary.csv`
- `outputs/week4/confirm_json/*.json`
- `outputs/week4/confirm_runs/...`

其中：

- `confirm_top3_raw.csv` 是每个 seed 的原始结果
- `confirm_top3_summary.csv` 是每组参数的均值和标准差汇总

## 11. 可视化与统计分析

### 11.1 画全部方法的收敛曲线

```bash
python plot_all_methods_convergence.py
```

作用：

- 从合并总表中读取各方法 trial 结果
- 计算 `best-so-far`
- 画出方法收敛曲线

主要输出：

- `outputs/week4_merged/plots/all_methods_best_so_far.csv`
- `outputs/week4_merged/plots/all_methods_convergence.png`

### 11.2 做小预算分析

```bash
python analyze_small_budget.py
```

作用：

- 统计不同方法在小预算 `10 / 20 / 30 trials` 下的表现
- 对比谁更适合预算受限场景
- 统计达到目标 Dice 所需的最少 trial 数

主要输出：

- `outputs/week4_merged/small_budget/budget_summary.csv`
- `outputs/week4_merged/small_budget/best_at_10.png`
- `outputs/week4_merged/small_budget/best_at_20.png`
- `outputs/week4_merged/small_budget/best_at_30.png`
- `outputs/week4_merged/small_budget/convergence_budget_10.png`
- `outputs/week4_merged/small_budget/convergence_budget_20.png`
- `outputs/week4_merged/small_budget/convergence_budget_30.png`
- `outputs/week4_merged/small_budget/target_trials.csv`

## 12. 一个推荐运行顺序

如果是要从现有内容继续做实验，建议按这个顺序：

```bash
python run_compare_random_tpe_hgs.py
python merge_week4_qrghgs_with_week4_compare.py
python plot_all_methods_convergence.py
python analyze_small_budget.py
python warmup_batch_samples.py
python merge_warmup_into_history.py
python run_surrogate_search.py
python run_surrogate_loop.py
python run_surrogate_multi_methods.py
python compare_qrghgs_before_after_surrogate.py
python compare_qrghgs_10_versions_surrogate.py
python run_confirm_top3.py
```

## 13. 目前这套 README 想表达的结论

可以把当前仓库理解成两条主线：

- 第一条主线：真实训练搜索，对比 `Random / TPE / HGS / QRG-HGS`
- 第二条主线：基于历史结果训练代理模型，在更大的搜索空间中筛选优质候选，再做少量真实训练验证

其中你已经完成的“代理模型建立”和“扩大搜索空间”，在代码里主要对应：

- 代理模型：`surrogate_model.py`、`surrogate_model_checked.py`
- 扩大搜索空间：`warmup_batch_samples.py`、`merge_warmup_into_history.py`、`run_surrogate_multi_methods.py`
- QRG-HGS 改进分析：`compare_qrghgs_before_after_surrogate.py`、`compare_qrghgs_10_versions_surrogate.py`

如果后面要继续写论文，最适合直接引用和整理的结果目录通常是：

- `outputs/week4_merged`
- `outputs/history_augmented`
- `outputs/surrogate_multi_methods`
- `outputs/qrghgs_before_after`
- `outputs/qrghgs_10_versions`


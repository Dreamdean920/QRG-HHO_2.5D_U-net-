# experiments

这里放实验编排入口，不放底层模型或工具函数。

- `real_compare/`：真实训练搜索方法对比，当前主线入口在这里。
- `confirm/`：Top-K 与多 seed 确认实验。
- `runners/`：ablation、Ray Tune 等配置化 runner。
- `surrogate/`：历史代理模型实验脚本。当前缺少 `surrogate_model.py` / `surrogate_model_checked.py`，恢复前不建议作为主线运行。

建议从项目根目录运行脚本，例如：

```bash
python experiments/real_compare/run_paper_main_real_benchmark.py
```

主线脚本命名：

- `real_compare/run_paper_main_real_benchmark.py`：论文主线真实对比。
- `real_compare/run_main_qrghgs_struct_v3.py`：冻结的 Struct-v3 主方法。
- `real_compare/run_explore_structv3_ablation_directions.py`：Struct-v3 的探索和消融方向。
- `surrogate/failed_surrogate_qrghgs_*`：代理模型历史失败/归档线。

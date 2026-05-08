# experiments

这里放实验编排入口，不放底层模型或工具函数。

- `real_compare/`：真实训练搜索方法对比，当前主线入口在这里。
- `confirm/`：Top-K 与多 seed 确认实验。
- `runners/`：ablation、Ray Tune 等配置化 runner。
- `surrogate/`：历史代理模型实验脚本。当前缺少 `surrogate_model.py` / `surrogate_model_checked.py`，恢复前不建议作为主线运行。

建议从项目根目录运行脚本，例如：

```bash
python experiments/real_compare/run_real_compare_7methods.py
```

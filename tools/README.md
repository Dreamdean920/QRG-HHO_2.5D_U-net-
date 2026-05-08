# tools

这里放不直接训练模型的辅助脚本。

- `analysis/`：读取已有 CSV，生成收敛、预算、时间效率、排名等图表。
- `results/`：合并历史结果、warmup 结果等。
- `maintenance/`：清理、恢复、删除实验记录的维护脚本。

当前主线分析优先运行：

```bash
python tools/analysis/analyze_convergence_and_budget.py
python tools/analysis/analyze_time_efficiency.py
```

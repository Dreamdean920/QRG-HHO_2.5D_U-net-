# tools

这里放不直接训练模型的辅助脚本。

- `analysis/`：读取已有 CSV，生成收敛、预算、时间效率、排名等图表。
- `results/`：合并历史结果、warmup 结果等。
- `maintenance/`：清理、恢复、删除实验记录的维护脚本。

当前主线分析优先运行：

```bash
python tools/analysis/analyze_completed_methods_convergence_budget.py
python tools/analysis/analyze_time_completed_methods.py
```

维护脚本命名：

- `maintenance/archive_failed_lines_from_main_results.py`：把失败探索线从主结果中归档。
- `maintenance/clean_failed_qrghgs_lines.py`：清理低质量 QRG-HGS 结果。
- `maintenance/restore_failed_line_json_from_backup.py`：从备份恢复失败线 JSON。

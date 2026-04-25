import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 路径配置
# =========================
INPUT_CSV = Path("outputs/week4_merged/all_methods_merged.csv")
OUT_DIR = Path("outputs/week4_merged/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "all_methods_convergence.png"
OUT_CSV = OUT_DIR / "all_methods_best_so_far.csv"

# =========================
# 1. 读取数据
# =========================
df = pd.read_csv(INPUT_CSV)

# 只保留成功结果
df = df[df["status"] == "OK"].copy()

# 如果你想严格公平比较，建议打开这一行：只保留 batch_size=4
df = df[df["batch_size"] == 4].copy()

# 排序
df = df.sort_values(["method", "trial_idx"]).reset_index(drop=True)

print("方法列表：", sorted(df["method"].unique()))
print("总记录数：", len(df))

# =========================
# 2. 计算累计最优 best-so-far
# =========================
curve_rows = []

for method in sorted(df["method"].unique()):
    sub = df[df["method"] == method].sort_values("trial_idx").copy()
    sub["best_so_far"] = sub["test_dice"].cummax()

    for _, row in sub.iterrows():
        curve_rows.append({
            "method": method,
            "trial_idx": row["trial_idx"],
            "test_dice": row["test_dice"],
            "best_so_far": row["best_so_far"],
        })

curve_df = pd.DataFrame(curve_rows)
curve_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

# =========================
# 3. 画总收敛曲线
# =========================
plt.figure(figsize=(10, 6))

# 固定显示顺序，方便汇报
method_order = ["random", "tpe", "hgs", "qrghgs"]

for method in method_order:
    sub = curve_df[curve_df["method"] == method].sort_values("trial_idx")
    if len(sub) == 0:
        continue
    plt.plot(sub["trial_idx"], sub["best_so_far"], linewidth=2, label=method)

plt.xlabel("Trials", fontsize=14)
plt.ylabel("Best Dice So Far", fontsize=14)
plt.title("Convergence Curve of All Methods", fontsize=18)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.show()

print(f"\n✅ 已保存图片：{OUT_PNG}")
print(f"✅ 已保存曲线数据：{OUT_CSV}")

# =========================
# 4. 额外输出：每个方法最终最优值
# =========================
summary_rows = []
for method in method_order:
    sub = curve_df[curve_df["method"] == method]
    if len(sub) == 0:
        continue
    summary_rows.append({
        "method": method,
        "final_best_so_far": sub["best_so_far"].iloc[-1],
        "num_trials": len(sub)
    })

summary_df = pd.DataFrame(summary_rows)
print("\n===== 最终累计最优值 =====")
print(summary_df)
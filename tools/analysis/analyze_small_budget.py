import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 路径配置
# =========================
INPUT_CSV = Path("outputs/week4_merged/all_methods_merged.csv")
OUT_DIR = Path("outputs/week4_merged/small_budget")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BUDGETS = [10, 20, 30]

# =========================
# 1. 读取数据
# =========================
df = pd.read_csv(INPUT_CSV)

# 只保留成功结果
df = df[df["status"] == "OK"].copy()

# 如果你想严格公平，只保留 batch_size=4，就打开这一行
df = df[df["batch_size"] == 4].copy()

# 排序，确保按 trial 顺序分析
df = df.sort_values(["method", "trial_idx"]).reset_index(drop=True)

print("方法列表：", df["method"].unique())
print("总记录数：", len(df))


# =========================
# 2. 计算 Best@K
# =========================
summary_rows = []

for method in sorted(df["method"].unique()):
    sub = df[df["method"] == method].sort_values("trial_idx").reset_index(drop=True)

    row = {"method": method}

    for b in BUDGETS:
        sub_b = sub.head(b)
        if len(sub_b) == 0:
            row[f"best@{b}"] = None
            row[f"mean@{b}"] = None
        else:
            row[f"best@{b}"] = sub_b["test_dice"].max()
            row[f"mean@{b}"] = sub_b["test_dice"].mean()

    summary_rows.append(row)

budget_df = pd.DataFrame(summary_rows)
budget_df.to_csv(OUT_DIR / "budget_summary.csv", index=False, encoding="utf-8-sig")

print("\n===== 小预算结果表 =====")
print(budget_df)


# =========================
# 3. 画 Best@K 柱状图
# =========================
for b in BUDGETS:
    plt.figure(figsize=(7, 4))
    plot_df = budget_df.sort_values(f"best@{b}", ascending=False)

    plt.bar(plot_df["method"], plot_df[f"best@{b}"])
    plt.ylabel("Best Dice")
    plt.title(f"Best Dice under Budget={b}")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"best_at_{b}.png", dpi=300)
    plt.close()


# =========================
# 4. 画小预算收敛曲线
# =========================
for b in BUDGETS:
    plt.figure(figsize=(8, 5))

    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method].sort_values("trial_idx").head(b).copy()
        if len(sub) == 0:
            continue

        best_so_far = sub["test_dice"].cummax()
        plt.plot(sub["trial_idx"].values, best_so_far.values, label=method)

    plt.xlabel("Trials")
    plt.ylabel("Best Dice So Far")
    plt.title(f"Convergence Curve (Budget={b})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"convergence_budget_{b}.png", dpi=300)
    plt.close()


# =========================
# 5. 额外输出：达到目标 Dice 所需 trial 数
# =========================
TARGETS = [0.960, 0.962]

target_rows = []

for method in sorted(df["method"].unique()):
    sub = df[df["method"] == method].sort_values("trial_idx").reset_index(drop=True)
    best_so_far = sub["test_dice"].cummax()

    row = {"method": method}

    for t in TARGETS:
        hit = sub.loc[best_so_far >= t, "trial_idx"]
        row[f"trial_to_{t:.3f}"] = int(hit.iloc[0]) if len(hit) > 0 else None

    target_rows.append(row)

target_df = pd.DataFrame(target_rows)
target_df.to_csv(OUT_DIR / "target_trials.csv", index=False, encoding="utf-8-sig")

print("\n===== 达到目标Dice所需trial数 =====")
print(target_df)

print("\n已输出到目录：", OUT_DIR)
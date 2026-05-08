import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import cycle


# =========================================================
# 0. 路径配置
# =========================================================
INPUT_CSV = Path("outputs/paper_main_structv3_real_benchmark/compare_raw.csv")

OUT_DIR = Path("outputs/paper_main_structv3_real_benchmark/analysis_time_budget_completed_methods")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_CURVE_CSV = OUT_DIR / "time_convergence_completed.csv"
TIME_BUDGET_CSV = OUT_DIR / "time_budget_summary_completed.csv"
TIME_TARGET_CSV = OUT_DIR / "time_to_target_completed.csv"
SUMMARY_CSV = OUT_DIR / "time_method_summary_completed.csv"
RANK_CSV = OUT_DIR / "time_method_rank_completed.csv"
EXCLUDED_CSV = OUT_DIR / "excluded_unfinished_methods.csv"

# 只保留至少跑满多少次的算法
MIN_TRIALS_REQUIRED = 30

# 时间预算，单位：秒
# 可以按需要调整
TIME_BUDGETS = [
    3600,     # 1h
    7200,     # 2h
    10800,    # 3h
    21600,    # 6h
    43200,    # 12h
    86400,    # 24h
]

# 目标 Dice
TARGETS = [0.960, 0.962, 0.964]


# =========================================================
# 1. 高识别度绘图风格
# =========================================================
HIGH_CONTRAST_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
    "#000000",  # black
    "#00429d",  # dark blue
    "#93003a",  # dark magenta
    "#007d34",  # dark green
    "#ffb300",  # strong yellow
    "#803e75",  # plum
    "#ff6800",  # vivid orange
    "#a6bdd7",  # light blue
    "#c10020",  # crimson
    "#cea262",  # tan
    "#817066",  # brown gray
    "#f6768e",  # rose
    "#00538a",  # deep blue
    "#ff8e00",  # amber
    "#53377a",  # indigo
]

LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "8"]


def build_style_map(methods):
    color_cycle = cycle(HIGH_CONTRAST_COLORS)
    linestyle_cycle = cycle(LINESTYLES)
    marker_cycle = cycle(MARKERS)

    style_map = {}
    for method in methods:
        style_map[method] = {
            "color": next(color_cycle),
            "linestyle": next(linestyle_cycle),
            "marker": next(marker_cycle),
        }

    return style_map


def seconds_to_hour_label(seconds):
    h = seconds / 3600
    if abs(h - int(h)) < 1e-9:
        return f"{int(h)}h"
    return f"{h:.1f}h"


# =========================================================
# 2. 读取数据
# =========================================================
if not INPUT_CSV.exists():
    raise FileNotFoundError(f"找不到文件: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)

if "resumed" not in df.columns:
    df["resumed"] = False

required_cols = ["method", "trial_idx", "best_val_dice", "test_dice", "time_sec", "status"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"缺少必要列: {col}")

# 保留成功结果
df = df[df["status"] == "OK"].copy()

# 去重，避免 resumed 重复记录影响统计
if "run_name" in df.columns:
    df = (
        df.sort_values(["method", "trial_idx", "resumed"])
          .drop_duplicates(subset=["method", "run_name"], keep="last")
          .reset_index(drop=True)
    )
else:
    df = (
        df.sort_values(["method", "trial_idx", "resumed"])
          .drop_duplicates(subset=["method", "trial_idx"], keep="last")
          .reset_index(drop=True)
    )

# 数值处理
for col in ["trial_idx", "best_val_dice", "test_dice", "time_sec"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["trial_idx", "best_val_dice", "test_dice", "time_sec"]).copy()
df["trial_idx"] = df["trial_idx"].astype(int)

# =========================================================
# 3. 筛选：只保留跑满的算法
# =========================================================
method_counts = (
    df.groupby("method")
      .size()
      .reset_index(name="n_ok")
      .sort_values("method")
)

completed_methods = method_counts[
    method_counts["n_ok"] >= MIN_TRIALS_REQUIRED
]["method"].tolist()

excluded_methods = method_counts[
    method_counts["n_ok"] < MIN_TRIALS_REQUIRED
].copy()

excluded_methods.to_csv(EXCLUDED_CSV, index=False, encoding="utf-8-sig")

if len(completed_methods) == 0:
    raise ValueError(
        f"没有任何方法达到 MIN_TRIALS_REQUIRED={MIN_TRIALS_REQUIRED}。"
        f"请检查 compare_raw.csv 或降低 MIN_TRIALS_REQUIRED。"
    )

df = df[df["method"].isin(completed_methods)].copy()

# 为了公平，只取每个方法前 MIN_TRIALS_REQUIRED 条成功记录
df = (
    df.sort_values(["method", "trial_idx"])
      .groupby("method", group_keys=False)
      .head(MIN_TRIALS_REQUIRED)
      .reset_index(drop=True)
)

all_methods = sorted(df["method"].dropna().unique())
style_map = build_style_map(all_methods)

print("===== 已纳入时间分析的方法（已跑满） =====")
for i, m in enumerate(all_methods, start=1):
    print(f"{i:02d}. {m}")

print("\n===== 已排除的方法（未跑满） =====")
if len(excluded_methods) == 0:
    print("无")
else:
    print(excluded_methods)

print("\n纳入方法数:", len(all_methods))
print("纳入记录数:", len(df))
print("未跑完方法已保存:", EXCLUDED_CSV)


# =========================================================
# 4. 构建时间收敛曲线
# =========================================================
rows = []

for method in all_methods:
    sub = df[df["method"] == method].sort_values("trial_idx").copy()

    sub["cum_time"] = sub["time_sec"].cumsum()
    sub["cum_time_hour"] = sub["cum_time"] / 3600

    sub["best_val_so_far"] = sub["best_val_dice"].cummax()
    sub["best_test_so_far"] = sub["test_dice"].cummax()

    for _, r in sub.iterrows():
        rows.append({
            "method": method,
            "trial_idx": r["trial_idx"],
            "cum_time": r["cum_time"],
            "cum_time_hour": r["cum_time_hour"],
            "best_val_dice": r["best_val_dice"],
            "test_dice": r["test_dice"],
            "best_val_so_far": r["best_val_so_far"],
            "best_test_so_far": r["best_test_so_far"],
            "time_sec": r["time_sec"],
        })

time_curve_df = pd.DataFrame(rows)
time_curve_df.to_csv(TIME_CURVE_CSV, index=False, encoding="utf-8-sig")

print("\n已生成时间收敛数据:", TIME_CURVE_CSV)


# =========================================================
# 5. 时间收敛曲线绘图函数
# =========================================================
def plot_time_convergence(y_col, ylabel, title, output_name):
    plt.figure(figsize=(15, 8))

    for method in all_methods:
        sub = time_curve_df[time_curve_df["method"] == method].sort_values("cum_time")

        if len(sub) == 0:
            continue

        style = style_map[method]

        plt.plot(
            sub["cum_time_hour"],
            sub[y_col],
            linewidth=2.2,
            marker=style["marker"],
            markersize=4,
            linestyle=style["linestyle"],
            color=style["color"],
            label=method,
        )

    plt.xlabel("Cumulative Time (hours)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)

    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0, 0.78, 1])
    plt.savefig(OUT_DIR / output_name, dpi=300)
    plt.close()


plot_time_convergence(
    y_col="best_val_so_far",
    ylabel="Best Validation Dice So Far",
    title=f"Time-based Convergence: Validation Dice\nCompleted Methods Only (n ≥ {MIN_TRIALS_REQUIRED})",
    output_name="time_convergence_val_completed.png",
)

plot_time_convergence(
    y_col="best_test_so_far",
    ylabel="Best Test Dice So Far",
    title=f"Time-based Convergence: Test Dice\nCompleted Methods Only (n ≥ {MIN_TRIALS_REQUIRED})",
    output_name="time_convergence_test_completed.png",
)


# =========================================================
# 6. 时间预算对比
# =========================================================
budget_rows = []

for method in all_methods:
    sub = df[df["method"] == method].sort_values("trial_idx").copy()
    sub["cum_time"] = sub["time_sec"].cumsum()

    row = {
        "method": method,
        "n_total": len(sub),
        "total_time_sec": sub["time_sec"].sum(),
        "total_time_hour": sub["time_sec"].sum() / 3600,
        "mean_time_sec": sub["time_sec"].mean(),
    }

    for t in TIME_BUDGETS:
        sub_t = sub[sub["cum_time"] <= t]

        label = seconds_to_hour_label(t)

        if len(sub_t) == 0:
            row[f"best_val@{label}"] = None
            row[f"best_test@{label}"] = None
            row[f"mean_test@{label}"] = None
            row[f"std_test@{label}"] = None
            row[f"n_trials@{label}"] = 0
        else:
            row[f"best_val@{label}"] = sub_t["best_val_dice"].max()
            row[f"best_test@{label}"] = sub_t["test_dice"].max()
            row[f"mean_test@{label}"] = sub_t["test_dice"].mean()
            row[f"std_test@{label}"] = sub_t["test_dice"].std(ddof=0)
            row[f"n_trials@{label}"] = len(sub_t)

    budget_rows.append(row)

budget_df = pd.DataFrame(budget_rows)
budget_df.to_csv(TIME_BUDGET_CSV, index=False, encoding="utf-8-sig")

print("\n===== 时间预算对比 =====")
print(budget_df)
print("\n已保存:", TIME_BUDGET_CSV)


# =========================================================
# 7. 时间预算柱状图函数
# =========================================================
def plot_time_budget_bar(col, ylabel, title, output_name):
    if col not in budget_df.columns:
        return

    plot_df = budget_df.dropna(subset=[col]).copy()
    if len(plot_df) == 0:
        return

    plot_df = plot_df.sort_values(col, ascending=False)

    colors = [style_map[m]["color"] for m in plot_df["method"]]

    plt.figure(figsize=(max(12, len(plot_df) * 0.65), 6))
    plt.bar(plot_df["method"], plot_df[col], color=colors)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(OUT_DIR / output_name, dpi=300)
    plt.close()


for t in TIME_BUDGETS:
    label = seconds_to_hour_label(t)

    plot_time_budget_bar(
        col=f"best_val@{label}",
        ylabel=f"Best Val Dice @ {label}",
        title=f"Time Budget Comparison: Best Val Dice @ {label}\nCompleted Methods Only",
        output_name=f"time_budget_best_val_at_{label}_completed.png",
    )

    plot_time_budget_bar(
        col=f"best_test@{label}",
        ylabel=f"Best Test Dice @ {label}",
        title=f"Time Budget Comparison: Best Test Dice @ {label}\nCompleted Methods Only",
        output_name=f"time_budget_best_test_at_{label}_completed.png",
    )

    plot_time_budget_bar(
        col=f"mean_test@{label}",
        ylabel=f"Mean Test Dice @ {label}",
        title=f"Time Budget Comparison: Mean Test Dice @ {label}\nCompleted Methods Only",
        output_name=f"time_budget_mean_test_at_{label}_completed.png",
    )


# =========================================================
# 8. 达到目标 Dice 所需时间
# =========================================================
target_rows = []

for method in all_methods:
    sub = df[df["method"] == method].sort_values("trial_idx").copy()

    sub["cum_time"] = sub["time_sec"].cumsum()
    sub["cum_time_hour"] = sub["cum_time"] / 3600
    sub["best_val_so_far"] = sub["best_val_dice"].cummax()
    sub["best_test_so_far"] = sub["test_dice"].cummax()

    row = {"method": method}

    for target in TARGETS:
        hit_val = sub[sub["best_val_so_far"] >= target]
        hit_test = sub[sub["best_test_so_far"] >= target]

        row[f"time_sec_to_val_{target:.3f}"] = (
            float(hit_val.iloc[0]["cum_time"]) if len(hit_val) > 0 else None
        )
        row[f"time_hour_to_val_{target:.3f}"] = (
            float(hit_val.iloc[0]["cum_time_hour"]) if len(hit_val) > 0 else None
        )

        row[f"time_sec_to_test_{target:.3f}"] = (
            float(hit_test.iloc[0]["cum_time"]) if len(hit_test) > 0 else None
        )
        row[f"time_hour_to_test_{target:.3f}"] = (
            float(hit_test.iloc[0]["cum_time_hour"]) if len(hit_test) > 0 else None
        )

    target_rows.append(row)

target_df = pd.DataFrame(target_rows)
target_df.to_csv(TIME_TARGET_CSV, index=False, encoding="utf-8-sig")

print("\n===== 达到目标 Dice 所需时间 =====")
print(target_df)
print("\n已保存:", TIME_TARGET_CSV)


# =========================================================
# 9. 时间-性能散点图
# =========================================================
summary_rows = []

for method in all_methods:
    sub = df[df["method"] == method].copy()

    summary_rows.append({
        "method": method,
        "n": len(sub),
        "mean_time_sec": sub["time_sec"].mean(),
        "mean_time_hour": sub["time_sec"].mean() / 3600,
        "sum_time_sec": sub["time_sec"].sum(),
        "sum_time_hour": sub["time_sec"].sum() / 3600,
        "best_val": sub["best_val_dice"].max(),
        "best_test": sub["test_dice"].max(),
        "mean_test": sub["test_dice"].mean(),
        "std_test": sub["test_dice"].std(ddof=0),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values("mean_test", ascending=False)
summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

plt.figure(figsize=(12, 8))

for _, r in summary_df.iterrows():
    method = r["method"]
    style = style_map[method]

    plt.scatter(
        r["mean_time_sec"],
        r["mean_test"],
        s=100,
        color=style["color"],
        marker=style["marker"],
        edgecolors="black",
        linewidths=0.6,
    )

    plt.text(
        r["mean_time_sec"],
        r["mean_test"],
        method,
        fontsize=8,
    )

plt.xlabel("Mean Time per Trial (s)")
plt.ylabel("Mean Test Dice")
plt.title(f"Time vs Performance\nCompleted Methods Only (n ≥ {MIN_TRIALS_REQUIRED})")
plt.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig(OUT_DIR / "time_vs_performance_completed.png", dpi=300)
plt.close()


# =========================================================
# 10. 时间效率排名表
# =========================================================
rank_df = summary_df.copy()

rank_df["rank_mean_test"] = rank_df["mean_test"].rank(ascending=False, method="min")
rank_df["rank_best_test"] = rank_df["best_test"].rank(ascending=False, method="min")
rank_df["rank_best_val"] = rank_df["best_val"].rank(ascending=False, method="min")
rank_df["rank_mean_time"] = rank_df["mean_time_sec"].rank(ascending=True, method="min")
rank_df["rank_stability"] = rank_df["std_test"].rank(ascending=True, method="min")

rank_df = rank_df.sort_values(["rank_mean_test", "rank_best_test"])
rank_df.to_csv(RANK_CSV, index=False, encoding="utf-8-sig")

print("\n===== 时间效率排名表 =====")
print(rank_df[[
    "method",
    "n",
    "best_val",
    "best_test",
    "mean_test",
    "std_test",
    "mean_time_sec",
    "sum_time_hour",
    "rank_mean_test",
    "rank_best_test",
    "rank_mean_time",
    "rank_stability",
]])


# =========================================================
# 11. 最终提示
# =========================================================
print("\nDONE: 时间分析完成，只纳入跑满的方法，并使用高识别度颜色。")
print("输出目录:", OUT_DIR)

print("\n主要输出文件：")
print("1.", OUT_DIR / "time_convergence_val_completed.png")
print("2.", OUT_DIR / "time_convergence_test_completed.png")
print("3.", OUT_DIR / "time_budget_summary_completed.csv")
print("4.", OUT_DIR / "time_to_target_completed.csv")
print("5.", OUT_DIR / "time_vs_performance_completed.png")
print("6.", OUT_DIR / "time_method_rank_completed.csv")
print("7.", OUT_DIR / "excluded_unfinished_methods.csv")

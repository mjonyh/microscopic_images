#!/usr/bin/env python3
"""WS1 Step 1.3: Statistical significance testing on existing model comparison data."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10, "figure.dpi": 150,
})

# Load existing comparison data
df = pd.read_csv(OUTPUT_DIR / "ws1_model_comparison.csv")
print(f"Loaded {len(df)} records")

DEGRADATIONS = ["noise_50", "combined_mild"]
methods = ["DeBCR", "PI-DDPM", "N2V", "DoG", "DeBCR+DoG"]

# ── Paired t-tests ──────────────────────────────────────────
print("\nPaired t-tests (each method vs. Raw):")
print(f"  {'Method':20s} {'Mean IoU':>10s} {'Mean Delta':>10s} {'t-stat':>10s} {'p-value':>10s} {'Cohen d':>10s} {'Sig':>5s}")
print("  " + "-" * 80)

stats_results = []
for method in methods:
    for deg in DEGRADATIONS:
        sub = df[(df["method"] == method) & (df["degradation"] == deg)]
        raw_sub = df[(df["method"] == "Raw") & (df["degradation"] == deg)]

        if len(sub) > 3 and len(raw_sub) > 3:
            n = min(len(sub), len(raw_sub))
            t_stat, p_value = stats.ttest_rel(sub["iou"].values[:n], raw_sub["iou"].values[:n])
            mean_iou = sub["iou"].mean()
            mean_impr = sub["improvement"].mean()

            # Cohen's d
            diff = sub["iou"].values[:n] - raw_sub["iou"].values[:n]
            cohens_d = diff.mean() / (diff.std() + 1e-10)

            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

            print(f"  {method+'_'+deg[:5]:20s} {mean_iou:10.4f} {mean_impr:10.4f} {t_stat:10.3f} {p_value:10.4f} {cohens_d:10.3f} {sig:>5s}")

            # 95% CI for improvement
            ci = stats.t.interval(0.95, len(diff)-1, loc=diff.mean(), scale=stats.sem(diff))

            stats_results.append({
                "method": method,
                "degradation": deg,
                "n": n,
                "mean_iou": round(mean_iou, 6),
                "std_iou": round(sub["iou"].std(), 6),
                "mean_improvement": round(mean_impr, 6),
                "std_improvement": round(sub["improvement"].std(), 6),
                "t_statistic": round(t_stat, 4),
                "p_value": round(p_value, 6),
                "cohens_d": round(cohens_d, 4),
                "ci_lower": round(ci[0], 6),
                "ci_upper": round(ci[1], 6),
                "significant": p_value < 0.05,
            })

df_stats = pd.DataFrame(stats_results)
df_stats.to_csv(OUTPUT_DIR / "ws1_statistics.csv", index=False)
print(f"\nSaved: ws1_statistics.csv ({len(df_stats)} rows)")

# ── Summary table ──────────────────────────────────────────
print("\nSummary by degradation and method:")
summary = df[df["method"] != "HQ_ref"].groupby(["degradation", "method"]).agg(
    mean_iou=("iou", "mean"),
    std_iou=("iou", "std"),
    mean_impr=("improvement", "mean"),
    n=("cell_line", "count")
).round(4)
print(summary.to_string())

# Best method per degradation
print("\nBest method per degradation:")
for deg in DEGRADATIONS:
    sub = df[(df["degradation"] == deg) & (~df["method"].isin(["HQ_ref", "Raw"]))]
    best = sub.groupby("method")["iou"].mean().sort_values(ascending=False)
    print(f"\n  {deg}:")
    for method, iou in best.items():
        impr = sub[sub["method"] == method]["improvement"].mean()
        print(f"    {method:15s}: IoU={iou:.4f}  Δ={impr:+.4f}")

# ── Figure ─────────────────────────────────────────────────
print("\nGenerating comparison figure...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Physics-Informed Model Comparison — With Statistical Tests",
             fontsize=13, fontweight="bold")

# (a) IoU comparison
ax = axes[0, 0]
for deg in DEGRADATIONS:
    ious = []
    for method in methods:
        val = df[(df["method"] == method) & (df["degradation"] == deg)]["iou"].mean()
        ious.append(val if not np.isnan(val) else 0)
    ax.plot(methods, ious, marker="o", label=deg, linewidth=1.5)
ax.set_ylabel("Mean IoU")
ax.set_title("(a) IoU by Method")
ax.legend(fontsize=7)
ax.tick_params(axis="x", rotation=45)
ax.grid(True, alpha=0.3)

# (b) Improvement with significance
ax = axes[0, 1]
for deg in DEGRADATIONS:
    imprs = []
    for method in methods:
        val = df[(df["method"] == method) & (df["degradation"] == deg)]["improvement"].mean()
        imprs.append(val if not np.isnan(val) else 0)
    ax.bar(methods, imprs, alpha=0.7, label=deg)
ax.axhline(0, color="red", linestyle="--", alpha=0.5)
ax.set_ylabel("IoU Improvement")
ax.set_title("(b) Improvement over Raw")
ax.legend(fontsize=7)
ax.tick_params(axis="x", rotation=45)

# (c) Effect size (Cohen's d)
ax = axes[0, 2]
for di, deg in enumerate(DEGRADATIONS):
    effects = []
    for method in methods:
        sub_stats = df_stats[(df_stats["method"] == method) & (df_stats["degradation"] == deg)]
        if len(sub_stats) > 0:
            effects.append(sub_stats["cohens_d"].values[0])
        else:
            effects.append(0)
    x = np.arange(len(methods)) + di * 0.35
    ax.bar(x, effects, 0.35, alpha=0.7, label=deg)
ax.set_xticks(np.arange(len(methods)) + 0.175)
ax.set_xticklabels(methods, rotation=45)
ax.axhline(0.2, color="green", linestyle="--", alpha=0.3, label="Small")
ax.axhline(0.5, color="orange", linestyle="--", alpha=0.3, label="Medium")
ax.axhline(0.8, color="red", linestyle="--", alpha=0.3, label="Large")
ax.set_ylabel("Cohen's d")
ax.set_title("(c) Effect Size")
ax.legend(fontsize=6)

# (d) Per-cell-line comparison
ax = axes[1, 0]
cell_lines = sorted(df["cell_line"].unique())
x = np.arange(len(cell_lines))
width = 0.15
for idx, method in enumerate(methods[:4]):
    ious = []
    for cl in cell_lines:
        val = df[(df["method"] == method) & (df["cell_line"] == cl) & (df["degradation"] == "combined_mild")]["iou"].mean()
        ious.append(val if not np.isnan(val) else 0)
    ax.bar(x + idx*width, ious, width, label=method, alpha=0.7)
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(cell_lines, rotation=45)
ax.set_ylabel("Mean IoU")
ax.set_title("(d) Per-Cell-Line (combined)")
ax.legend(fontsize=6)

# (e) Gap closure to HQ
ax = axes[1, 1]
for deg in DEGRADATIONS:
    gap_closed = []
    for method in methods:
        sub = df[(df["method"] == method) & (df["degradation"] == deg)]
        raw_sub = df[(df["method"] == "Raw") & (df["degradation"] == deg)]
        hq_sub = df[(df["method"] == "HQ_ref") & (df["degradation"] == deg)]
        if len(sub) > 0 and len(raw_sub) > 0 and len(hq_sub) > 0:
            gap_raw = hq_sub["iou"].mean() - raw_sub["iou"].mean()
            gap_method = hq_sub["iou"].mean() - sub["iou"].mean()
            pct = (1 - gap_method / gap_raw) * 100 if gap_raw > 0 else 0
            gap_closed.append(pct)
        else:
            gap_closed.append(0)
    ax.plot(methods, gap_closed, marker="s", label=deg, linewidth=1.5)
ax.axhline(100, color="green", linestyle="--", alpha=0.3, label="Perfect")
ax.set_ylabel("Gap Closed (%)")
ax.set_title("(e) HQ Gap Closure")
ax.legend(fontsize=7)
ax.tick_params(axis="x", rotation=45)
ax.set_ylim(0, 110)
ax.grid(True, alpha=0.3)

# (f) p-value heatmap
ax = axes[1, 2]
pval_matrix = np.ones((len(methods), len(DEGRADATIONS)))
for i, method in enumerate(methods):
    for j, deg in enumerate(DEGRADATIONS):
        sub = df_stats[(df_stats["method"] == method) & (df_stats["degradation"] == deg)]
        if len(sub) > 0:
            pval_matrix[i, j] = sub["p_value"].values[0]

im = ax.imshow(np.log10(pval_matrix + 1e-10), cmap="RdYlGn_r", aspect="auto", vmin=-4, vmax=0)
ax.set_xticks(range(len(DEGRADATIONS)))
ax.set_xticklabels(DEGRADATIONS, rotation=45)
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods)
ax.set_title("(f) log10(p-value)")
plt.colorbar(im, ax=ax, fraction=0.046)

# Add significance markers
for i in range(len(methods)):
    for j in range(len(DEGRADATIONS)):
        p = pval_matrix[i, j]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(j, i, sig, ha="center", va="center", fontsize=8,
                color="white" if p < 0.05 else "black")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ws1_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ws1_comparison.png")

print("\nWS1 Step 1.3 (statistics) complete.")

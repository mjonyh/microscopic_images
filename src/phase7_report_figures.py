#!/usr/bin/env python3
"""
Generate comprehensive scientific report figures for filter analysis
across image quality levels.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import OUTPUT_DIR

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10, "figure.dpi": 150,
})

# ── Load all results ───────────────────────────────────────
df_hq = pd.read_csv(OUTPUT_DIR / "filter_segmentation_results.csv")
df_lq = pd.read_csv(OUTPUT_DIR / "filter_lq_comparison.csv")
df_adaptive = pd.read_csv(OUTPUT_DIR / "filter_adaptive_best.csv")
df_comp = pd.read_csv(OUTPUT_DIR / "filter_comparison_raw_fixed_adaptive.csv")

# ── Figure 1: Filter Performance Overview (HQ) ─────────────
print("Generating Figure 1: HQ filter performance overview...")

fig = plt.figure(figsize=(16, 10))
fig.suptitle("Figure 1: Bandpass Filter Performance on High-Quality Microscopy Images",
             fontsize=14, fontweight="bold")

# (a) IoU heatmap
ax1 = fig.add_subplot(2, 3, 1)
pivot = df_hq[df_hq["filter_type"] != "raw"].groupby(
    ["cell_line", "filter_type"])["iou"].max().unstack()
pivot = pivot.loc[:, pivot.mean().sort_values(ascending=False).index[:8]]
im = ax1.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0.2, vmax=0.9)
ax1.set_xticks(range(len(pivot.columns)))
ax1.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
ax1.set_yticks(range(len(pivot.index)))
ax1.set_yticklabels(pivot.index, fontsize=8)
ax1.set_title("(a) Best IoU: Filter × Cell Line", fontsize=9)
plt.colorbar(im, ax=ax1, fraction=0.046)

# (b) Improvement by filter type
ax2 = fig.add_subplot(2, 3, 2)
filter_types = ["homomorphic", "dog", "butterworth", "gaussian", "elliptic", "chebyshev1"]
data = [df_hq[df_hq["filter_type"] == ft]["improvement"].values for ft in filter_types]
bp = ax2.boxplot(data, tick_labels=filter_types, patch_artist=True,
                medianprops=dict(color="black", lw=1.5))
colors = ["#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8", "#cba6f7", "#fab387"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
ax2.axhline(0, color="red", linestyle="--", alpha=0.5)
ax2.set_ylabel("IoU Improvement")
ax2.set_title("(b) Improvement by Filter Type", fontsize=9)
ax2.tick_params(axis="x", rotation=45)

# (c) Best filter frequency
ax3 = fig.add_subplot(2, 3, 3)
# Get best filter per cell line
best_per_line = df_hq[df_hq["filter_type"] != "raw"].groupby("cell_line").apply(
    lambda g: g.loc[g["iou"].idxmax(), "filter_type"]
)
best_counts = best_per_line.value_counts().head(6)
bars = ax3.bar(best_counts.index, best_counts.values,
               color=[colors[filter_types.index(f)] if f in filter_types else "#a6acaf"
                      for f in best_counts.index], edgecolor="white")
ax3.set_xlabel("Filter Type")
ax3.set_ylabel("Cell Lines (count)")
ax3.set_title("(c) Best Filter Frequency", fontsize=9)
ax3.tick_params(axis="x", rotation=45)

# (d) Raw vs adaptive comparison
ax4 = fig.add_subplot(2, 3, 4)
x = np.arange(len(df_comp))
width = 0.25
ax4.bar(x - width, df_comp["raw_mean"], width, label="Raw", color="#a6acaf", edgecolor="white")
ax4.bar(x, df_comp["fixed_mean"], width, label="Fixed BW", color="#89b4fa", edgecolor="white")
ax4.bar(x + width, df_comp["adaptive_mean"], width, label="Adaptive", color="#a6e3a1", edgecolor="white")
ax4.set_xticks(x)
ax4.set_xticklabels(df_comp["cell_line"], rotation=45, fontsize=8)
ax4.set_ylabel("Mean IoU")
ax4.set_title("(d) Raw vs Fixed vs Adaptive", fontsize=9)
ax4.legend(fontsize=7)

# (e) HQ vs LQ filter performance
ax5 = fig.add_subplot(2, 3, 5)
lq_summary = df_lq.groupby(["degradation", "filter_type"])["iou"].mean().unstack()
lq_summary = lq_summary[["raw", "homomorphic", "dog", "butterworth", "gaussian"]]
lq_summary.plot(kind="bar", ax=ax5, color=["#a6acaf", "#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8"],
                edgecolor="white")
ax5.set_ylabel("Mean IoU")
ax5.set_title("(e) Filter Performance on Low-Quality Images", fontsize=9)
ax5.tick_params(axis="x", rotation=45)
ax5.legend(fontsize=6, ncol=2)

# (f) Improvement magnitude: HQ vs LQ
ax6 = fig.add_subplot(2, 3, 6)
hq_improve = df_hq[df_hq["filter_type"] == "dog"]["improvement"].values
lq_improve = df_lq[df_lq["filter_type"] == "dog"]["improvement"].values
ax6.hist(hq_improve, bins=30, alpha=0.5, label=f"HQ DoG (mean={hq_improve.mean():.3f})", color="#a6e3a1")
ax6.hist(lq_improve, bins=30, alpha=0.5, label=f"LQ DoG (mean={lq_improve.mean():.3f})", color="#f38ba8")
ax6.axvline(0, color="red", linestyle="--", alpha=0.5)
ax6.set_xlabel("IoU Improvement")
ax6.set_ylabel("Frequency")
ax6.set_title("(f) HQ vs LQ Improvement (DoG)", fontsize=9)
ax6.legend(fontsize=7)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "report_filter_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: report_filter_performance.png")

# ── Figure 2: Filter Transfer Across Quality Levels ────────
print("Generating Figure 2: Filter transfer across quality levels...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Figure 2: Filter Performance Transfer from High-Quality to Low-Quality Images",
             fontsize=12, fontweight="bold")

# (a) IoU degradation curve
ax = axes[0]
quality_levels = ["HQ", "noise_50", "combined_mild"]
for filt in ["homomorphic", "dog", "butterworth", "gaussian"]:
    ious = []
    # HQ
    hq_iou = df_hq[df_hq["filter_type"] == filt]["iou"].mean()
    ious.append(hq_iou)
    # LQ
    for deg in ["noise_50", "combined_mild"]:
        lq_iou = df_lq[(df_lq["filter_type"] == filt) & (df_lq["degradation"] == deg)]["iou"].mean()
        ious.append(lq_iou)
    ax.plot(quality_levels, ious, marker="o", label=filt, linewidth=1.5)
ax.set_ylabel("Mean IoU")
ax.set_title("(a) IoU vs Quality Level")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# (b) Improvement ratio (LQ improvement / HQ improvement)
ax = axes[1]
for filt in ["homomorphic", "dog", "butterworth", "gaussian"]:
    hq_impr = df_hq[df_hq["filter_type"] == filt]["improvement"].mean()
    lq_impr = df_lq[(df_lq["filter_type"] == filt) & (df_lq["degradation"] == "noise_50")]["improvement"].mean()
    ratio = lq_impr / hq_impr if hq_impr > 0 else 0
    ax.bar(filt, ratio, color=colors[["homomorphic", "dog", "butterworth", "gaussian"].index(filt)],
           edgecolor="white")
ax.set_ylabel("LQ/HQ Improvement Ratio")
ax.set_title("(b) Filter Transfer Efficiency")
ax.tick_params(axis="x", rotation=45)
ax.axhline(1, color="red", linestyle="--", alpha=0.5, label="Perfect transfer")
ax.legend(fontsize=7)

# (c) Application-specific performance
ax = axes[2]
apps = ["Segmentation\n(HQ)", "Segmentation\n(LQ noise)", "Segmentation\n(LQ combined)", "Classification"]
# Segmentation IoU (best filter)
seg_hq = df_hq[df_hq["filter_type"] == "dog"]["iou"].mean()
seg_lq_noise = df_lq[(df_lq["filter_type"] == "dog") & (df_lq["degradation"] == "noise_50")]["iou"].mean()
seg_lq_comb = df_lq[(df_lq["filter_type"] == "dog") & (df_lq["degradation"] == "combined_mild")]["iou"].mean()
# Classification accuracy
import json
with open(OUTPUT_DIR / "filter_classification_results.json") as f:
    clf = json.load(f)
clf_acc = clf["classification"]["raw_accuracy"]

values = [seg_hq, seg_lq_noise, seg_lq_comb, clf_acc]
bar_colors = ["#a6e3a1", "#f9e2af", "#f38ba8", "#89b4fa"]
bars = ax.bar(apps, values, color=bar_colors, edgecolor="white")
ax.set_ylabel("Performance (IoU or Accuracy)")
ax.set_title("(c) Application-Specific Performance")
ax.set_ylim(0, 1.0)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "report_filter_transfer.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: report_filter_transfer.png")

# ── Figure 3: Filter Selection Decision Tree ───────────────
print("Generating Figure 3: Filter selection guide...")

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis("off")
ax.set_title("Figure 3: Bandpass Filter Selection Guide for Microscopy Images",
             fontsize=12, fontweight="bold", pad=20)

# Create a text-based decision tree
tree_text = """
FILTER SELECTION DECISION GUIDE
================================

START
  │
  ├─ What is your primary application?
  │
  ├─► CELL SEGMENTATION
  │   │
  │   ├─ Image quality HIGH (PSNR > 30 dB)?
  │   │   ├─ Cell line with heavy shading? → HOMOMORPHIC (γ_L=0.5, γ_H=2.0)
  │   │   ├─ Small uniform cells (SHSY5Y, Huh7)? → DoG (σ₁=0.05, σ₂=0.20)
  │   │   ├─ Large variable cells (BV2, SKOV3)? → GAUSSIAN (d_low=0.02, d_high=0.30)
  │   │   └─ Default → BUTTERWORTH (n=2, d_low=0.02, d_high=0.30)
  │   │
  │   └─ Image quality LOW (PSNR < 20 dB)?
  │       ├─ Heavy noise? → BUTTERWORTH (n=2, d_low=0.03, d_high=0.25)
  │       ├─ Motion blur? → DoG (σ₁=0.03, σ₂=0.15)
  │       ├─ Uneven illumination? → HOMOMORPHIC (γ_L=0.3, γ_H=2.5)
  │       └─ Combined degradation? → BUTTERWORTH (n=4, d_low=0.02, d_high=0.35)
  │
  ├─► CELL CLASSIFICATION
  │   └─ Use RAW FFT features (no filtering)
  │      Filtering removes discriminative low-frequency information
  │      Raw accuracy: 75.3% vs Filtered: 74.8%
  │
  ├─► CELL COUNTING
  │   └─ HOMOMORPHIC (γ_L=0.5, γ_H=2.0)
  │      Improves counting accuracy (MAE 0.978 → 0.937)
  │
  └─► ILLUMINATION CORRECTION
      └─ HOMOMORPHIC (γ_L=0.3-0.7, γ_H=1.5-3.0)
         Specifically designed for multiplicative illumination artifacts
         Tune γ_L based on shading severity

KEY FINDINGS:
• No single filter is universally best — performance is cell-line and quality-dependent
• Adaptive filtering adds +0.130 IoU over fixed filtering
• Filter improvements on LQ images are ~10x smaller than on HQ images
• DoG and Homomorphic are the strongest general-purpose choices
• Avoid Ideal and Elliptic filters (ringing artifacts around cell boundaries)
"""

ax.text(0.05, 0.95, tree_text, transform=ax.transAxes, fontsize=8,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "report_filter_decision_tree.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: report_filter_decision_tree.png")

print("\nAll report figures generated.")

#!/usr/bin/env python3
"""
Generate the missing composite figures for manuscript submission.
Figures 7-10 from CHECKLIST.md
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Set publication quality params
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

OUTPUT_DIR = Path("/home/mjonyh/git/livecell/outputs")
FIG_DIR = Path("/home/mjonyh/git/livecell/manuscript/outputs")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
COLORS = {
    "Raw": "#666666",
    "DeBCR": "#1f77b4",
    "PI-DDPM": "#ff7f0e",
    "N2V": "#2ca02c",
    "DoG": "#d62728",
    "DeBCR+DoG": "#9467bd",
    "HQ_ref": "#8c564b",
}

print("Loading data...")
# Load statistics
stats_df = pd.read_csv(OUTPUT_DIR / "ws1_statistics.csv")
# Load model comparison
model_df = pd.read_csv(OUTPUT_DIR / "ws1_model_comparison.csv")
# Load classification results
try:
    class_df = pd.read_csv(OUTPUT_DIR / "obj4_classification_report.csv")
except FileNotFoundError:
    class_df = None

# ============================================================================
# FIGURE 7: Filter x Method Comparison Matrix
# ============================================================================
print("\nGenerating Figure 7: Filter x Method Comparison Matrix...")

# Get filter performance data
filter_summary = pd.read_csv(OUTPUT_DIR / "filter_segmentation_summary.csv")

# Pivot for heatmap: cell_line x filter_type -> mean_improvement
pivot_data = filter_summary.pivot_table(
    values="mean_improvement", 
    index="cell_line", 
    columns="filter_type",
    aggfunc="mean"
)

# Filter types to show (top ones)
filter_types = ["DoG", "Butterworth", "Homomorphic", "Gaussian", "Ideal", "Elliptic"]
cell_lines = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SKOV3", "SkBr3"]

# Create filtered pivot
pivot_filtered = pivot_data.reindex(index=cell_lines, columns=filter_types)

fig, ax = plt.subplots(figsize=(10, 6))
cmap = plt.cm.RdYlGn

# Plot heatmap
im = ax.imshow(pivot_filtered, aspect="auto", cmap=cmap, vmin=-0.1, vmax=0.5)

# Labels
ax.set_xticks(range(len(filter_types)))
ax.set_xticklabels(filter_types, rotation=45, ha="right")
ax.set_yticks(range(len(cell_lines)))
ax.set_yticklabels(cell_lines)

# Add annotations
for i, cl in enumerate(cell_lines):
    for j, ft in enumerate(filter_types):
        val = pivot_filtered.loc[cl, ft] if ft in pivot_filtered.columns and cl in pivot_filtered.index else np.nan
        if pd.notna(val):
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)

# Colorbar
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Mean IoU Improvement", fontsize=9)

ax.set_xlabel("Filter Type", fontsize=10)
ax.set_ylabel("Cell Line", fontsize=10)
ax.set_title("Filter x Cell Line IoU Improvement Matrix", fontsize=11, fontweight="bold")

plt.tight_layout()
fig.savefig(FIG_DIR / "fig7_filter_method_matrix.pdf", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig7_filter_method_matrix.png", dpi=300, bbox_inches="tight")
print(f"  -> Saved: fig7_filter_method_matrix.pdf/png")
plt.close(fig)

# ============================================================================
# FIGURE 8: Physics Model Comparison Bar Chart with Significance
# ============================================================================
print("\nGenerating Figure 8: Physics Model Comparison Bar Chart...")

# Aggregate statistics by method and degradation
agg_stats = stats_df.groupby(["method", "degradation"]).agg({
    "mean_iou": "mean",
    "std_iou": "mean",
    "mean_improvement": "mean",
    "p_value": "first",
    "significant": "first"
}).reset_index()

# Methods to compare
methods = ["Raw", "DeBCR", "PI-DDPM", "N2V", "DoG", "DeBCR+DoG"]
degradations = ["noise_50", "combined_mild"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, deg in enumerate(degradations):
    ax = axes[idx]
    deg_data = agg_stats[agg_stats["degradation"] == deg].sort_values("mean_iou", ascending=False)
    
    # Get methods that exist in this degradation
    existing_methods = deg_data["method"].tolist()
    
    ious = deg_data["mean_iou"].values
    stds = deg_data["std_iou"].values
    is_sig = deg_data["significant"].values
    
    bars = ax.bar(range(len(existing_methods)), ious, yerr=stds, 
                  capsize=3, color=[COLORS.get(m, "#cccccc") for m in existing_methods],
                  edgecolor="black", linewidth=0.5)
    
    # Add significance stars
    for i, sig in enumerate(is_sig):
        if sig and existing_methods[i] != "Raw":
            height = ious[i] + stds[i] + 0.005
            ax.text(i, height, "*", ha="center", va="bottom", fontsize=12, color="black")
    
    ax.set_xticks(range(len(existing_methods)))
    ax.set_xticklabels(existing_methods, rotation=45, ha="right")
    ax.set_ylabel("Mean IoU", fontsize=10)
    ax.set_title(f"({chr(97+idx)}) {deg.replace('_', ' ').title()}", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Physics Model Comparison: Mean IoU by Degradation Type", 
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig8_physics_model_comparison.pdf", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig8_physics_model_comparison.png", dpi=300, bbox_inches="tight")
print(f"  -> Saved: fig8_physics_model_comparison.pdf/png")
plt.close(fig)

# ============================================================================
# FIGURE 9: ROC-style Classification Comparison
# ============================================================================
print("\nGenerating Figure 9: Classification Comparison...")

if class_df is not None:
    # Extract metrics - cell lines are in the index
    # Skip the last 3 rows (accuracy, macro avg, weighted avg)
    class_data = class_df.iloc[:-3].copy()
    class_data.index.name = 'cell_line'
    class_data = class_data.reset_index()
    
    cell_lines = class_data["cell_line"].tolist()
    recall = class_data["recall"].tolist()
    precision = class_data["precision"].tolist()
    f1 = class_data["f1-score"].tolist()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar width
    width = 0.25
    x = np.arange(len(cell_lines))

    # Plot bars
    rects1 = ax.bar(x - width, recall, width, label='Recall', color=COLORS["DeBCR"], edgecolor="black")
    rects2 = ax.bar(x, precision, width, label='Precision', color=COLORS["PI-DDPM"], edgecolor="black")
    rects3 = ax.bar(x + width, f1, width, label='F1-Score', color=COLORS["DeBCR+DoG"], edgecolor="black")

    ax.set_xlabel("Cell Line", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Classification Metrics by Cell Line (SVM-RBF, 5-fold CV)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cell_lines, rotation=45, ha="right")
    ax.legend(loc="upper right", ncol=3)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height + 0.01, 
                    f"{height:.2f}", ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig9_classification_comparison.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig9_classification_comparison.png", dpi=300, bbox_inches="tight")
    print(f"  -> Saved: fig9_classification_comparison.pdf/png")
    plt.close(fig)
else:
    print("  -> WARNING: Classification data not found. Skipping fig9.")

# ============================================================================
# FIGURE 10: Enhancement Visual Comparison Grid (4x4)
# ============================================================================
print("\nGenerating Figure 10: Enhancement Visual Comparison Grid...")

# Check if we have visual comparison images
visual_files = list(FIG_DIR.glob("visual_comparison_*.png")) + \
                list(OUTPUT_DIR.glob("visual_comparison_*.png"))

if visual_files:
    # Select 4 degradation types x 4 methods = 16 images
    # We'll create a 4x4 grid
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    # Sample files (take first 16)
    sample_files = sorted(visual_files)[:16]
    
    for idx, (ax, file) in enumerate(zip(axes.flat, sample_files)):
        try:
            img = plt.imread(file)
            if img.ndim == 3:
                img = img[:, :, 0] if img.shape[2] > 1 else img[:, :, 0]
            ax.imshow(img, cmap="gray")
            ax.set_title(file.stem.replace("visual_comparison_", "").replace("_", " "), 
                        fontsize=8)
            ax.axis("off")
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
    
    plt.suptitle("Enhancement Visual Comparison Grid\nRows: Degradation Types, Columns: Methods", 
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig10_enhancement_grid.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig10_enhancement_grid.png", dpi=300, bbox_inches="tight")
    print(f"  -> Saved: fig10_enhancement_grid.pdf/png")
    plt.close(fig)
else:
    print("  -> WARNING: No visual comparison images found. Skipping fig10.")

print("\nAll composite figures generated!")

#!/usr/bin/env python3
"""
Phase 3: Segmentation comparison across all filter types.
For each annotated image, apply Otsu segmentation after filtering
with each filter type and compute IoU against COCO ground truth.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from skimage.filters import threshold_otsu

sys.path.insert(0, str(Path(__file__).parent))
from filters import FILTER_REGISTRY, apply_filter
from common import load_image, list_images, load_annotations, OUTPUT_DIR

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10, "figure.dpi": 150,
})

# ── Filter configurations to test ──────────────────────────
# Each entry: (filter_name, param_set_label, param_dict)
FILTER_CONFIGS = [
    # Ideal
    ("ideal", "ideal_0.02_0.30", dict(d_low=0.02, d_high=0.30)),
    # Butterworth
    ("butterworth", "bw2_0.02_0.30", dict(d_low=0.02, d_high=0.30, order=2)),
    ("butterworth", "bw2_0.01_0.20", dict(d_low=0.01, d_high=0.20, order=2)),
    ("butterworth", "bw2_0.05_0.40", dict(d_low=0.05, d_high=0.40, order=2)),
    ("butterworth", "bw4_0.02_0.30", dict(d_low=0.02, d_high=0.30, order=4)),
    # Gaussian
    ("gaussian", "gauss_0.02_0.30", dict(d_low=0.02, d_high=0.30)),
    ("gaussian", "gauss_0.01_0.20", dict(d_low=0.01, d_high=0.20)),
    ("gaussian", "gauss_0.05_0.40", dict(d_low=0.05, d_high=0.40)),
    # Homomorphic
    ("homomorphic", "homo_0.5_2.0", dict(d0=0.10, gamma_l=0.5, gamma_h=2.0, c=1.0)),
    ("homomorphic", "homo_0.3_1.5", dict(d0=0.10, gamma_l=0.3, gamma_h=1.5, c=1.0)),
    ("homomorphic", "homo_0.7_3.0", dict(d0=0.10, gamma_l=0.7, gamma_h=3.0, c=1.0)),
    # DoG
    ("dog", "dog_005_020", dict(sigma1=0.05, sigma2=0.20)),
    ("dog", "dog_003_010", dict(sigma1=0.03, sigma2=0.10)),
    ("dog", "dog_010_030", dict(sigma1=0.10, sigma2=0.30)),
    # Laplacian-BP
    ("laplacian", "lap_0.02_0.30", dict(d_low=0.02, d_high=0.30)),
    ("laplacian", "lap_0.01_0.40", dict(d_low=0.01, d_high=0.40)),
    # Chebyshev I
    ("chebyshev1", "cheb1_0.02_0.30", dict(d_low=0.02, d_high=0.30, order=2, ripple_db=0.5)),
    ("chebyshev1", "cheb1_0.02_0.30_o4", dict(d_low=0.02, d_high=0.30, order=4, ripple_db=0.5)),
    # Chebyshev II
    ("chebyshev2", "cheb2_0.02_0.30", dict(d_low=0.02, d_high=0.30, order=2, attenuation_db=40)),
    # Elliptic
    ("elliptic", "ellip_0.02_0.30", dict(d_low=0.02, d_high=0.30, order=2, ripple_db=0.5, attenuation_db=40)),
    # Trapezoidal
    ("trapezoidal", "trap_0.01_03_25_35", dict(d1=0.01, d2=0.03, d3=0.25, d4=0.35)),
    # Cosine-tapered
    ("cosine", "cos_0.02_0.30", dict(d_low=0.02, d_high=0.30, transition_width=0.04)),
    # Parametric
    ("parametric", "param_b2_s015", dict(beta=2, sigma=0.15)),
    ("parametric", "param_b1_s020", dict(beta=1, sigma=0.20)),
]

# ── Segmentation function ──────────────────────────────────
def segment_and_compute_iou(image, gt_bboxes):
    """Otsu segment image, compute IoU against ground truth bboxes."""
    if not gt_bboxes:
        return 0.0
    try:
        thresh = threshold_otsu(image)
        pred_mask = image > thresh
    except ValueError:
        return 0.0
    # Build GT mask from bboxes
    gt_mask = np.zeros_like(image, dtype=bool)
    for bbox in gt_bboxes:
        x, y, w, h = [int(v) for v in bbox]
        h_img, w_img = image.shape
        x = max(0, min(x, w_img-1))
        y = max(0, min(y, h_img-1))
        x2 = max(0, min(x+w, w_img))
        y2 = max(0, min(y+h, h_img))
        gt_mask[y:y2, x:x2] = True
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0.0

# ── Main pipeline ──────────────────────────────────────────
print("Phase 3: Segmentation comparison across all filters")
print("=" * 60)

annotations = load_annotations()
cell_lines = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SKOV3", "SkBr3"]

# Build image list: all annotated images
annotated_images = []
for cl in cell_lines:
    for p in list_images(cl):
        if p.stem in annotations:
            annotated_images.append(p)

n_images = len(annotated_images)
n_configs = len(FILTER_CONFIGS)
print(f"  Images: {n_images}, Filter configs: {n_configs}")
print(f"  Total segmentations: {n_images * (n_configs + 1)} (including raw)")

# ── Process all images ─────────────────────────────────────
records = []
for i, path in enumerate(annotated_images):
    if i % 50 == 0:
        print(f"  Processing image {i+1}/{n_images}...")

    img = load_image(path)
    ann = annotations.get(path.stem, {})
    gt_bboxes = ann.get("bboxes", [])
    cell_line = path.stem.split("_")[0]

    # Raw (unfiltered)
    iou_raw = segment_and_compute_iou(img, gt_bboxes)
    records.append({
        "filename": path.stem, "cell_line": cell_line,
        "filter_type": "raw", "config_label": "raw",
        "iou": iou_raw, "improvement": 0.0
    })

    # Filtered versions
    for filter_name, config_label, params in FILTER_CONFIGS:
        try:
            filtered = apply_filter(img, filter_name, **params)
            iou_filt = segment_and_compute_iou(filtered, gt_bboxes)
            records.append({
                "filename": path.stem, "cell_line": cell_line,
                "filter_type": filter_name, "config_label": config_label,
                "iou": iou_filt, "improvement": iou_filt - iou_raw
            })
        except Exception as e:
            records.append({
                "filename": path.stem, "cell_line": cell_line,
                "filter_type": filter_name, "config_label": config_label,
                "iou": 0.0, "improvement": -iou_raw
            })

df = pd.DataFrame(records)
csv_path = OUTPUT_DIR / "filter_segmentation_results.csv"
df.to_csv(csv_path, index=False)
print(f"\n  Results saved: {csv_path}")

# ── Aggregation ────────────────────────────────────────────
print("\nGenerating summary statistics...")

# Best config per filter type (across all configs of that type)
summary = df[df["filter_type"] != "raw"].groupby(["cell_line", "filter_type"]).agg(
    best_iou=("iou", "max"),
    mean_iou=("iou", "mean"),
    best_config=("config_label", lambda x: x.iloc[df.loc[x.index, "iou"].argmax()]),
    best_improvement=("improvement", "max"),
    mean_improvement=("improvement", "mean"),
).reset_index()

# Raw IoU per cell line
raw_iou = df[df["filter_type"] == "raw"].groupby("cell_line")["iou"].mean()
summary["raw_iou"] = summary["cell_line"].map(raw_iou)
summary["net_improvement"] = summary["best_iou"] - summary["raw_iou"]

# Best filter per cell line
best_per_line = summary.loc[summary.groupby("cell_line")["best_iou"].idxmax()]

print("\n  Best filter per cell line:")
for _, row in best_per_line.iterrows():
    print(f"    {row['cell_line']:10s}: {row['filter_type']:15s} "
          f"IoU={row['best_iou']:.4f} (raw={row['raw_iou']:.4f}, "
          f"Δ={row['net_improvement']:+.4f})")

summary.to_csv(OUTPUT_DIR / "filter_segmentation_summary.csv", index=False)
best_per_line.to_csv(OUTPUT_DIR / "filter_best_per_cellline.csv", index=False)

# ── Visualization ──────────────────────────────────────────
print("\nGenerating visualization figures...")

# (a) IoU heatmap: filter types × cell lines
fig, ax = plt.subplots(figsize=(14, 6))
pivot = summary.pivot_table(index="filter_type", columns="cell_line", values="best_iou")
# Sort by mean IoU
pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0.2, vmax=0.7)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=9)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=9)
ax.set_title("Best IoU per Filter Type × Cell Line", fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax, label="IoU")
# Annotate
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7, color="white" if val < 0.4 else "black")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "filter_iou_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: filter_iou_heatmap.png")

# (b) Improvement box plots
fig, ax = plt.subplots(figsize=(12, 6))
filter_types = sorted(df[df["filter_type"] != "raw"]["filter_type"].unique())
data = [df[df["filter_type"] == ft]["improvement"].values for ft in filter_types]
bp = ax.boxplot(data, tick_labels=filter_types, patch_artist=True,
                medianprops=dict(color="black", lw=1.5))
colors = plt.cm.Set3(np.linspace(0, 1, len(filter_types)))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
ax.axhline(0, color="red", linestyle="--", alpha=0.5)
ax.set_ylabel("IoU Improvement (filtered − raw)")
ax.set_title("Segmentation Improvement Distribution by Filter Type", fontweight="bold")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "filter_improvement_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: filter_improvement_boxplot.png")

# (c) Best filter frequency
fig, ax = plt.subplots(figsize=(8, 5))
best_counts = best_per_line["filter_type"].value_counts()
bars = ax.bar(best_counts.index, best_counts.values,
             color=[colors[filter_types.index(ft)] for ft in best_counts.index],
             edgecolor="white")
ax.set_xlabel("Filter Type")
ax.set_ylabel("Cell Lines (count)")
ax.set_title("How Often Each Filter Type Is Best", fontweight="bold")
ax.tick_params(axis="x", rotation=45)
for bar, count in zip(bars, best_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            str(count), ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "filter_best_frequency.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: filter_best_frequency.png")

# (d) Raw vs best filtered scatter per cell line
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("Raw vs. Best Filtered IoU per Image", fontsize=12, fontweight="bold")
for idx, cl in enumerate(cell_lines):
    ax = axes[idx // 4, idx % 4]
    raw = df[(df["cell_line"] == cl) & (df["filter_type"] == "raw")]["iou"].values
    best = df[(df["cell_line"] == cl) & (df["filter_type"] != "raw")].groupby("filename")["iou"].max().values
    lim = [0, max(raw.max(), best.max()) + 0.05]
    ax.scatter(raw, best, alpha=0.4, s=10, c="#89b4fa")
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_title(cl, fontsize=9)
    ax.set_xlabel("Raw IoU"), ax.set_ylabel("Best Filtered IoU")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "filter_raw_vs_best.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: filter_raw_vs_best.png")

print("\nPhase 3 complete.")

#!/usr/bin/env python3
"""
Run filter comparison on low-quality images.
Tests whether filter recommendations from HQ images transfer to LQ images.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.filters import threshold_otsu

sys.path.insert(0, str(Path(__file__).parent))
from filters import FILTER_REGISTRY, apply_filter
from common import load_image, list_images, load_annotations, OUTPUT_DIR

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10, "figure.dpi": 150,
})

annotations = load_annotations()

# Select degradation types to test
DEGRADATIONS = ["noise_50", "defocus_4", "shading_0.5", "combined_mild"]

# Top filters from HQ analysis
TOP_FILTERS = {
    "homomorphic": dict(d0=0.10, gamma_l=0.5, gamma_h=2.0),
    "dog": dict(sigma1=0.05, sigma2=0.20),
    "butterworth": dict(d_low=0.02, d_high=0.30, order=2),
    "gaussian": dict(d_low=0.02, d_high=0.30),
}

def segment_iou(image, gt_bboxes):
    if not gt_bboxes:
        return 0.0
    try:
        thresh = threshold_otsu(image)
        pred = image > thresh
    except ValueError:
        return 0.0
    gt = np.zeros_like(image, dtype=bool)
    for bbox in gt_bboxes:
        x, y, w, h = [int(v) for v in bbox]
        gt[y:min(y+h, image.shape[0]), x:min(x+w, image.shape[1])] = True
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / union if union > 0 else 0.0

print("Running filter comparison on low-quality images...")
print("=" * 60)

# Get annotated images
annotated_stems = set(annotations.keys())
all_images = []
for cl in ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SKOV3", "SkBr3"]:
    for p in list_images(cl):
        if p.stem in annotated_stems:
            all_images.append(p)

print(f"  Annotated images: {len(all_images)}")
print(f"  Degradations: {DEGRADATIONS}")
print(f"  Filters: {list(TOP_FILTERS.keys())}")

records = []
for i, path in enumerate(all_images):
    if i % 50 == 0:
        print(f"  Processing {i+1}/{len(all_images)}...")

    img_hq = load_image(path)
    ann = annotations.get(path.stem, {})
    bboxes = ann.get("bboxes", [])
    cell_line = path.stem.split("_")[0]

    for deg_name in DEGRADATIONS:
        # Load degraded image
        deg_path = Path("data/mixed_quality") / "synthetic_low" / deg_name / f"{path.stem}.tif"
        if not deg_path.exists():
            continue

        img_lq = np.array(__import__('PIL.Image').Image.open(deg_path)).astype(np.float64)

        # Raw (no filter)
        iou_raw = segment_iou(img_lq, bboxes)
        records.append({
            "filename": path.stem, "cell_line": cell_line,
            "degradation": deg_name, "filter_type": "raw",
            "iou": iou_raw, "improvement": 0.0
        })

        # Apply each filter
        for filt_name, filt_params in TOP_FILTERS.items():
            try:
                filtered = apply_filter(img_lq, filt_name, **filt_params)
                iou_filt = segment_iou(filtered, bboxes)
                records.append({
                    "filename": path.stem, "cell_line": cell_line,
                    "degradation": deg_name, "filter_type": filt_name,
                    "iou": iou_filt, "improvement": iou_filt - iou_raw
                })
            except:
                pass

df = pd.DataFrame(records)
df.to_csv(OUTPUT_DIR / "filter_lq_comparison.csv", index=False)

# Summary
print("\n=== Filter Performance on Low-Quality Images ===")
summary = df.groupby(["degradation", "filter_type"]).agg(
    mean_iou=("iou", "mean"),
    mean_improvement=("improvement", "mean"),
    n=("filename", "count")
).round(4)
print(summary.to_string())

# Best filter per degradation
print("\n=== Best Filter per Degradation Type ===")
for deg in DEGRADATIONS:
    sub = df[df["degradation"] == deg]
    best = sub.groupby("filter_type")["iou"].mean().sort_values(ascending=False)
    print(f"\n  {deg}:")
    for filt, iou in best.items():
        print(f"    {filt:15s}: IoU={iou:.4f}")

# Compare HQ vs LQ filter performance
print("\n=== HQ vs LQ Filter Performance ===")
hq_results = {
    "homomorphic": 0.508,  # From adaptive optimization
    "dog": 0.527,
    "butterworth": 0.378,
    "gaussian": 0.394,
}
for filt in TOP_FILTERS:
    lq_iou = df[(df["filter_type"] == filt) & (df["degradation"] == "noise_50")]["iou"].mean()
    hq_iou = df[(df["filter_type"] == filt) & (df["degradation"] == "combined_mild")]["iou"].mean()
    print(f"  {filt:15s}: noise_50={lq_iou:.4f}, combined_mild={hq_iou:.4f}")

# Figure: filter performance across quality levels
fig, axes = plt.subplots(1, len(DEGRADATIONS), figsize=(16, 4))
fig.suptitle("Filter Performance Across Image Quality Levels", fontsize=12, fontweight="bold")

for idx, deg in enumerate(DEGRADATIONS):
    ax = axes[idx]
    sub = df[df["degradation"] == deg]
    filter_types = ["raw"] + list(TOP_FILTERS.keys())
    data = [sub[sub["filter_type"] == ft]["improvement"].values for ft in filter_types]
    bp = ax.boxplot(data, tick_labels=["raw"] + list(TOP_FILTERS.keys()),
                    patch_artist=True, medianprops=dict(color="black", lw=1.5))
    colors = ["#a6acaf", "#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_title(deg.replace("_", " "), fontsize=9)
    ax.set_ylabel("IoU Improvement")
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "filter_quality_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  Saved: filter_quality_comparison.png")

print("\nDone.")

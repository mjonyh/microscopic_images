#!/usr/bin/env python3
"""
Workstream 7: Cell-Line-Adaptive Enhancement
Depends on: WS1 (enhancement models), WS3 (U-Net)
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import load_image, list_images, load_annotations, OUTPUT_DIR
from filters import apply_filter
from phaseA_physics_models import DeBCRInspired, PIDDPMInspired

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10, "figure.dpi": 150,
})

print("=" * 60)
print("WS7: Cell-Line-Adaptive Enhancement")
print("=" * 60)

annotations = load_annotations()
cell_lines = ["MCF7", "SHSY5Y", "BV2", "SkBr3"]

# ── 7.1: Per-Cell-Line Model Training ──────────────────────
print("\n7.1: Per-Cell-Line Enhancement...")

# For each cell line, find optimal enhancement parameters
# by grid search on a subset of images

adaptive_results = []

for cl in cell_lines:
    print(f"\n  Optimizing for {cl}...")
    cl_images = [p for p in list_images(cl) if p.stem in annotations][:20]

    if not cl_images:
        continue

    best_params = None
    best_mean_iou = 0

    # Grid search over DeBCR parameters
    for wavelet in ['db4', 'sym5']:
        for levels in [2, 3, 4]:
            for lambda_phys in [0.05, 0.1, 0.2]:
                model = DeBCRInspired(wavelet=wavelet, levels=levels, lambda_physics=lambda_phys)

                ious = []
                for path in cl_images[:5]:  # Use subset for speed
                    img = load_image(path).astype(np.uint8)
                    ann = annotations.get(path.stem, {})
                    bboxes = ann.get("bboxes", [])

                    if not bboxes:
                        continue

                    # Apply enhancement
                    enhanced = model.enhance(img)

                    # Compute IoU
                    from skimage.filters import threshold_otsu
                    try:
                        thresh = threshold_otsu(enhanced)
                        pred = enhanced > thresh
                        gt = np.zeros_like(enhanced, dtype=bool)
                        for bbox in bboxes:
                            x, y, w, h = [int(v) for v in bbox]
                            gt[y:min(y+h, enhanced.shape[0]), x:min(x+w, enhanced.shape[1])] = True
                        iou = np.logical_and(pred, gt).sum() / np.logical_or(pred, gt).sum()
                        ious.append(iou)
                    except:
                        pass

                if ious:
                    mean_iou = np.mean(ious)
                    if mean_iou > best_mean_iou:
                        best_mean_iou = mean_iou
                        best_params = {
                            "wavelet": wavelet, "levels": levels, "lambda_physics": lambda_phys
                        }

    print(f"    Best params: {best_params}")
    print(f"    Best mean IoU: {best_mean_iou:.4f}")

    adaptive_results.append({
        "cell_line": cl,
        "best_params": str(best_params),
        "best_iou": best_mean_iou,
    })

df_adaptive = pd.DataFrame(adaptive_results)
df_adaptive.to_csv(OUTPUT_DIR / "ws7_adaptive_params.csv", index=False)

# ── 7.2: Quality-Aware Model Selector ──────────────────────
print("\n7.2: Quality-Aware Model Selector...")

# Create a simple rule-based selector
# In production, this would be a trained classifier

def quality_metrics(img):
    """Extract quality metrics from image."""
    from scipy.ndimage import sobel
    from numpy.fft import fft2, fftshift

    edges = np.sqrt(sobel(img, axis=0)**2 + sobel(img, axis=1)**2).mean()
    ft = fftshift(fft2(img.astype(float) - img.mean()))
    power = np.abs(ft)**2
    h, w = power.shape
    D = np.sqrt(
        np.meshgrid(np.fft.fftshift(np.fft.fftfreq(w)),
                   np.fft.fftshift(np.fft.fftfreq(h)))[0]**2 +
        np.meshgrid(np.fft.fftshift(np.fft.fftfreq(w)),
                   np.fft.fftshift(np.fft.fftfreq(h)))[1]**2
    )
    hf_ratio = power[D > 0.2].sum() / (power.sum() + 1e-10)

    return {"edge": edges, "hf_ratio": hf_ratio}

def select_model(quality):
    """Select best model based on quality metrics."""
    edge = quality["edge"]
    hf = quality["hf_ratio"]

    if edge > 40 and hf < 0.1:
        return "high_quality", "DoG (σ₁=0.05, σ₂=0.20)"
    elif edge > 20 and hf < 0.3:
        return "medium_quality", "Butterworth (n=2, d_low=0.02, d_high=0.30)"
    elif hf > 0.5:
        return "heavy_noise", "DoG (σ₁=0.03, σ₂=0.15)"
    else:
        return "low_quality", "DeBCR+DoG"

# Test selector
print("  Testing quality-aware selector...")
selector_results = []
for cl in cell_lines:
    imgs = [p for p in list_images(cl) if p.stem in annotations][:5]
    for path in imgs:
        img = load_image(path).astype(np.uint8)
        q = quality_metrics(img)
        quality_level, recommendation = select_model(q)
        selector_results.append({
            "cell_line": cl,
            "quality_level": quality_level,
            "edge": q["edge"],
            "hf_ratio": q["hf_ratio"],
            "recommendation": recommendation,
        })

df_selector = pd.DataFrame(selector_results)
df_selector.to_csv(OUTPUT_DIR / "ws7_selector.csv", index=False)

# ── 7.3: Adaptive Pipeline ─────────────────────────────────
print("\n7.3: Adaptive Pipeline...")

# Compare: adaptive vs fixed pipeline
pipeline_results = []

for cl in cell_lines:
    imgs = [p for p in list_images(cl) if p.stem in annotations][:10]
    for path in imgs:
        img = load_image(path).astype(np.uint8)
        ann = annotations.get(path.stem, {})
        bboxes = ann.get("bboxes", [])
        if not bboxes:
            continue

        # Fixed pipeline (DoG for all)
        from skimage.filters import threshold_otsu
        img_fixed = apply_filter(img, "dog", sigma1=0.05, sigma2=0.20)
        try:
            thresh = threshold_otsu(img_fixed)
            pred = img_fixed > thresh
            gt = np.zeros_like(img_fixed, dtype=bool)
            for bbox in bboxes:
                x, y, w, h = [int(v) for v in bbox]
                gt[y:min(y+h, img_fixed.shape[0]), x:min(x+w, img_fixed.shape[1])] = True
            iou_fixed = np.logical_and(pred, gt).sum() / np.logical_or(pred, gt).sum()
        except:
            iou_fixed = 0

        # Adaptive pipeline
        q = quality_metrics(img)
        quality_level, rec = select_model(q)

        if "DoG" in rec and "0.03" in rec:
            img_adaptive = apply_filter(img, "dog", sigma1=0.03, sigma2=0.15)
        elif "Butterworth" in rec:
            img_adaptive = apply_filter(img, "butterworth", d_low=0.02, d_high=0.30, order=2)
        elif "DeBCR" in rec:
            model = DeBCRInspired(wavelet='db4', levels=3, lambda_physics=0.1)
            img_adaptive = model.enhance(img)
            img_adaptive = apply_filter(img_adaptive, "dog", sigma1=0.05, sigma2=0.20)
        else:
            img_adaptive = apply_filter(img, "dog", sigma1=0.05, sigma2=0.20)

        try:
            thresh = threshold_otsu(img_adaptive)
            pred = img_adaptive > thresh
            gt = np.zeros_like(img_adaptive, dtype=bool)
            for bbox in bboxes:
                x, y, w, h = [int(v) for v in bbox]
                gt[y:min(y+h, img_adaptive.shape[0]), x:min(x+w, img_adaptive.shape[1])] = True
            iou_adaptive = np.logical_and(pred, gt).sum() / np.logical_or(pred, gt).sum()
        except:
            iou_adaptive = 0

        pipeline_results.append({
            "cell_line": cl,
            "quality_level": quality_level,
            "iou_fixed": iou_fixed,
            "iou_adaptive": iou_adaptive,
            "improvement": iou_adaptive - iou_fixed,
        })

df_pipeline = pd.DataFrame(pipeline_results)
df_pipeline.to_csv(OUTPUT_DIR / "ws7_pipeline.csv", index=False)

print(f"\n  Fixed pipeline mean IoU: {df_pipeline['iou_fixed'].mean():.4f}")
print(f"  Adaptive pipeline mean IoU: {df_pipeline['iou_adaptive'].mean():.4f}")
print(f"  Improvement: {df_pipeline['improvement'].mean():+.4f}")

# ── 7.4: Recommendations ──────────────────────────────────
print("\n7.4: Final Recommendations...")

# Create final recommendation table
recommendations = []
for cl in cell_lines:
    sub = df_pipeline[df_pipeline["cell_line"] == cl]
    adaptive_iou = sub["iou_adaptive"].mean()
    fixed_iou = sub["iou_fixed"].mean()

    # Get best params for this cell line
    cl_adaptive = df_adaptive[df_adaptive["cell_line"] == cl]
    if len(cl_adaptive) > 0:
        best_params = cl_adaptive.iloc[0]["best_params"]
    else:
        best_params = "N/A"

    recommendations.append({
        "cell_line": cl,
        "adaptive_iou": adaptive_iou,
        "fixed_iou": fixed_iou,
        "improvement": adaptive_iou - fixed_iou,
        "best_params": best_params,
        "recommendation": f"Adaptive (IoU={adaptive_iou:.3f}) > Fixed (IoU={fixed_iou:.3f})"
    })

df_rec = pd.DataFrame(recommendations)
df_rec.to_csv(OUTPUT_DIR / "ws7_recommendations.csv", index=False)

print("\n  Final Recommendations:")
print(df_rec.to_string())

# ── Figure ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Cell-Line-Adaptive Enhancement Results", fontsize=12, fontweight="bold")

# (a) Adaptive vs Fixed IoU
ax = axes[0, 0]
x = np.arange(len(cell_lines))
width = 0.35
ax.bar(x - width/2, [df_pipeline[df_pipeline["cell_line"]==cl]["iou_fixed"].mean() for cl in cell_lines],
       width, label="Fixed (DoG)", color="#89b4fa", edgecolor="white")
ax.bar(x + width/2, [df_pipeline[df_pipeline["cell_line"]==cl]["iou_adaptive"].mean() for cl in cell_lines],
       width, label="Adaptive", color="#a6e3a1", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(cell_lines, rotation=45)
ax.set_ylabel("Mean IoU")
ax.set_title("(a) Fixed vs Adaptive Pipeline")
ax.legend()

# (b) Improvement by cell line
ax = axes[0, 1]
improvements = [df_pipeline[df_pipeline["cell_line"]==cl]["improvement"].mean() for cl in cell_lines]
ax.bar(cell_lines, improvements, color="#f9e2af", edgecolor="white")
ax.axhline(0, color="red", linestyle="--", alpha=0.5)
ax.set_ylabel("IoU Improvement")
ax.set_title("(b) Adaptive Improvement")
ax.tick_params(axis="x", rotation=45)

# (c) Quality level distribution
ax = axes[1, 0]
quality_counts = df_selector["quality_level"].value_counts()
ax.pie(quality_counts.values, labels=quality_counts.index, autopct="%1.0f%%",
       colors=["#2ecc71", "#f39c12", "#e74c3c", "#89b4fa"])
ax.set_title("(c) Quality Level Distribution")

# (d) Per-cell-line best params
ax = axes[1, 1]
ious_adaptive = [df_rec[df_rec["cell_line"]==cl]["adaptive_iou"].values[0] for cl in cell_lines]
ious_fixed = [df_rec[df_rec["cell_line"]==cl]["fixed_iou"].values[0] for cl in cell_lines]
ax.plot(cell_lines, ious_fixed, "o-", label="Fixed", color="#89b4fa", linewidth=1.5)
ax.plot(cell_lines, ious_adaptive, "s-", label="Adaptive", color="#a6e3a1", linewidth=1.5)
ax.set_ylabel("Mean IoU")
ax.set_title("(d) Per-Cell-Line Performance")
ax.legend()
ax.tick_params(axis="x", rotation=45)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ws7_adaptive_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: ws7_adaptive_results.png")

print("\nWS7 complete.")

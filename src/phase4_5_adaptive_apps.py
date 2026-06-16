#!/usr/bin/env python3
"""
Phase 4: Cell-line-adaptive filter optimization.
Phase 5: Application-specific analysis.
Phase 6: Report update with filter results.
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
cell_lines = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SKOV3", "SkBr3"]

print("=" * 60)
print("Phase 4: Adaptive Filter Optimization")
print("=" * 60)

# ── Grid search: for each cell line, find optimal parameters ──
# Focus on top 3 filter types from Phase 3: DoG, Homomorphic, Butterworth
ADAPTIVE_CONFIGS = {
    "homomorphic": [
        dict(d0=0.08, gamma_l=0.3, gamma_h=1.5),
        dict(d0=0.10, gamma_l=0.5, gamma_h=2.0),
        dict(d0=0.12, gamma_l=0.4, gamma_h=2.5),
        dict(d0=0.15, gamma_l=0.3, gamma_h=3.0),
        dict(d0=0.10, gamma_l=0.7, gamma_h=2.0),
    ],
    "dog": [
        dict(sigma1=0.03, sigma2=0.10),
        dict(sigma1=0.05, sigma2=0.15),
        dict(sigma1=0.05, sigma2=0.20),
        dict(sigma1=0.08, sigma2=0.25),
        dict(sigma1=0.03, sigma2=0.15),
    ],
    "butterworth": [
        dict(d_low=0.01, d_high=0.20, order=2),
        dict(d_low=0.02, d_high=0.30, order=2),
        dict(d_low=0.02, d_high=0.40, order=2),
        dict(d_low=0.01, d_high=0.30, order=4),
        dict(d_low=0.03, d_high=0.35, order=2),
    ],
    "gaussian": [
        dict(d_low=0.01, d_high=0.25),
        dict(d_low=0.02, d_high=0.30),
        dict(d_low=0.01, d_high=0.35),
        dict(d_low=0.03, d_high=0.40),
    ],
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

adaptive_results = []
for cl in cell_lines:
    cl_images = [p for p in list_images(cl) if p.stem in annotations]
    if not cl_images:
        continue

    print(f"\n  Optimizing for {cl} ({len(cl_images)} images)...")

    for filter_name, param_list in ADAPTIVE_CONFIGS.items():
        for params in param_list:
            ious = []
            for path in cl_images:
                img = load_image(path)
                ann = annotations.get(path.stem, {})
                filtered = apply_filter(img, filter_name, **params)
                iou = segment_iou(filtered, ann.get("bboxes", []))
                ious.append(iou)
            mean_iou = np.mean(ious)
            adaptive_results.append({
                "cell_line": cl, "filter_type": filter_name,
                "params": str(params), "mean_iou": mean_iou, "n_images": len(ious)
            })

df_adaptive = pd.DataFrame(adaptive_results)

# Find best per cell line
best_adaptive = df_adaptive.loc[df_adaptive.groupby("cell_line")["mean_iou"].idxmax()]

print("\n  Optimal adaptive filter per cell line:")
for _, row in best_adaptive.iterrows():
    print(f"    {row['cell_line']:10s}: {row['filter_type']:15s} IoU={row['mean_iou']:.4f}")
    print(f"    {'':10s}  params: {row['params']}")

df_adaptive.to_csv(OUTPUT_DIR / "filter_adaptive_results.csv", index=False)
best_adaptive.to_csv(OUTPUT_DIR / "filter_adaptive_best.csv", index=False)

# Compare: raw vs fixed Butterworth vs adaptive
print("\n  Comparing raw vs fixed vs adaptive...")
comparison_data = []
for cl in cell_lines:
    cl_images = [p for p in list_images(cl) if p.stem in annotations]
    raw_ious, fixed_ious, adaptive_ious = [], [], []

    best_row = best_adaptive[best_adaptive["cell_line"] == cl]
    if best_row.empty:
        continue
    best_params = eval(best_row.iloc[0]["params"])
    best_filter = best_row.iloc[0]["filter_type"]

    for path in cl_images:
        img = load_image(path)
        ann = annotations.get(path.stem, {})
        bboxes = ann.get("bboxes", [])

        raw_ious.append(segment_iou(img, bboxes))

        fixed = apply_filter(img, "butterworth", d_low=0.02, d_high=0.30, order=2)
        fixed_ious.append(segment_iou(fixed, bboxes))

        adaptive = apply_filter(img, best_filter, **best_params)
        adaptive_ious.append(segment_iou(adaptive, bboxes))

    comparison_data.append({
        "cell_line": cl,
        "raw_mean": np.mean(raw_ious),
        "raw_std": np.std(raw_ious),
        "fixed_mean": np.mean(fixed_ious),
        "fixed_std": np.std(fixed_ious),
        "adaptive_mean": np.mean(adaptive_ious),
        "adaptive_std": np.std(adaptive_ious),
        "best_filter": best_filter,
    })
    print(f"    {cl:10s}: raw={np.mean(raw_ious):.4f}, fixed={np.mean(fixed_ious):.4f}, "
          f"adaptive={np.mean(adaptive_ious):.4f} (Δ={np.mean(adaptive_ious)-np.mean(raw_ious):+.4f})")

df_comp = pd.DataFrame(comparison_data)
df_comp.to_csv(OUTPUT_DIR / "filter_comparison_raw_fixed_adaptive.csv", index=False)

# Figure: comparison bar chart
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(cell_lines))
width = 0.25
bars1 = ax.bar(x - width, df_comp["raw_mean"], width, yerr=df_comp["raw_std"],
               label="Raw", color="#a6acaf", edgecolor="white", capsize=2)
bars2 = ax.bar(x, df_comp["fixed_mean"], width, yerr=df_comp["fixed_std"],
               label="Fixed Butterworth", color="#89b4fa", edgecolor="white", capsize=2)
bars3 = ax.bar(x + width, df_comp["adaptive_mean"], width, yerr=df_comp["adaptive_std"],
               label="Adaptive (best filter)", color="#a6e3a1", edgecolor="white", capsize=2)
ax.set_xlabel("Cell Line")
ax.set_ylabel("Mean IoU")
ax.set_title("Segmentation: Raw vs. Fixed vs. Adaptive Filtering", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(cell_lines, rotation=45)
ax.legend(fontsize=8)
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "filter_adaptive_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: filter_adaptive_comparison.png")

# ── Phase 5: Application-specific analysis ─────────────────
print("\n" + "=" * 60)
print("Phase 5: Application-Specific Analysis")
print("=" * 60)

# Load Phase 3 results for application analysis
df_seg = pd.read_csv(OUTPUT_DIR / "filter_segmentation_results.csv")

# App 1: Counting accuracy
print("\n  App 1: Cell counting accuracy...")
from skimage.measure import label, regionprops

def count_cells(image):
    try:
        thresh = threshold_otsu(image)
        binary = image > thresh
        labeled = label(binary)
        props = regionprops(labeled)
        return sum(1 for p in props if p.area > 50)
    except:
        return 0

counting_results = []
for cl in cell_lines[:4]:  # Sample 4 lines for speed
    cl_images = [p for p in list_images(cl) if p.stem in annotations][:20]
    for path in cl_images:
        img = load_image(path)
        ann = annotations.get(path.stem, {})
        gt_count = ann.get("cell_count", 0)

        raw_count = count_cells(img)
        filtered = apply_filter(img, "homomorphic", d0=0.10, gamma_l=0.5, gamma_h=2.0)
        filt_count = count_cells(filtered)

        counting_results.append({
            "cell_line": cl, "filename": path.stem,
            "gt_count": gt_count,
            "raw_count": raw_count,
            "filt_count": filt_count,
            "raw_error": abs(raw_count - gt_count) / max(gt_count, 1),
            "filt_error": abs(filt_count - gt_count) / max(gt_count, 1),
        })

df_count = pd.DataFrame(counting_results)
print(f"    Raw count MAE: {df_count['raw_error'].mean():.3f}")
print(f"    Filt count MAE: {df_count['filt_error'].mean():.3f}")

# App 2: Classification accuracy from filtered images
print("\n  App 2: Classification from filtered images...")
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from common import compute_fft, radial_profile, azimuthal_profile, spectral_features

def extract_fft_features(img):
    power, fx, fy = compute_fft(img)
    freqs, radial = radial_profile(power, n_bins=50)
    angles, azimuthal = azimuthal_profile(power, n_bins=36)
    feats = spectral_features(power, fx, fy)
    return np.concatenate([radial, azimuthal,
        [feats["centroid"], feats["bandwidth"], feats["skewness"],
         feats["kurtosis"], feats["total_power"], feats["low_power"],
         feats["mid_power"], feats["high_power"]]])

# Sample 200 images per cell line for speed
sample_data = []
for cl in cell_lines:
    imgs = list_images(cl)[:200]
    for path in imgs:
        img = load_image(path)
        # Raw features
        feat_raw = extract_fft_features(img)
        # Homomorphic filtered features
        filtered = apply_filter(img, "homomorphic", d0=0.10, gamma_l=0.5, gamma_h=2.0)
        feat_filt = extract_fft_features(filtered)
        sample_data.append({
            "cell_line": cl,
            "feat_raw": feat_raw,
            "feat_filt": feat_filt,
        })

df_class = pd.DataFrame(sample_data)
X_raw = np.vstack(df_class["feat_raw"].values)
X_filt = np.vstack(df_class["feat_filt"].values)
y = df_class["cell_line"].values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf"))])

scores_raw = cross_val_score(clf, X_raw, y, cv=cv, scoring="accuracy", n_jobs=-1)
scores_filt = cross_val_score(clf, X_filt, y, cv=cv, scoring="accuracy", n_jobs=-1)

print(f"    Raw FFT features accuracy:    {scores_raw.mean():.4f} ± {scores_raw.std():.4f}")
print(f"    Filtered FFT features accuracy: {scores_filt.mean():.4f} ± {scores_filt.std():.4f}")

# App 3: Illumination correction
print("\n  App 3: Illumination correction (background uniformity)...")
illum_results = []
for cl in cell_lines:
    imgs = list_images(cl)[:30]
    for path in imgs:
        img = load_image(path)
        # Measure background CV (coefficient of variation in corner regions)
        corners = [
            img[:50, :50], img[:50, -50:],
            img[-50:, :50], img[-50:, -50:]
        ]
        bg_values = np.concatenate([c.flatten() for c in corners])
        raw_cv = bg_values.std() / (bg_values.mean() + 1e-10)

        filtered = apply_filter(img, "homomorphic", d0=0.10, gamma_l=0.5, gamma_h=2.0)
        corners_f = [
            filtered[:50, :50], filtered[:50, -50:],
            filtered[-50:, :50], filtered[-50:, -50:]
        ]
        bg_f = np.concatenate([c.flatten() for c in corners_f])
        filt_cv = bg_f.std() / (bg_f.mean() + 1e-10)

        illum_results.append({
            "cell_line": cl, "raw_cv": raw_cv, "filt_cv": filt_cv,
            "improvement": raw_cv - filt_cv
        })

df_illum = pd.DataFrame(illum_results)
print(f"    Raw background CV:    {df_illum['raw_cv'].mean():.4f}")
print(f"    Filtered background CV: {df_illum['filt_cv'].mean():.4f}")
print(f"    Improvement: {df_illum['improvement'].mean():.4f}")

# Save application results
app_results = {
    "counting": df_count.to_dict(),
    "classification": {
        "raw_accuracy": float(scores_raw.mean()),
        "filtered_accuracy": float(scores_filt.mean()),
    },
    "illumination": {
        "raw_cv": float(df_illum["raw_cv"].mean()),
        "filtered_cv": float(df_illum["filt_cv"].mean()),
    }
}
pd.DataFrame([app_results]).to_json(OUTPUT_DIR / "filter_application_results.json")

# Figure: application comparison
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Application-Specific Filter Performance", fontsize=12, fontweight="bold")

# Counting
ax = axes[0]
cl_means = df_count.groupby("cell_line")[["raw_error", "filt_error"]].mean()
cl_means.plot(kind="bar", ax=ax, color=["#a6acaf", "#a6e3a1"], edgecolor="white")
ax.set_title("(a) Counting Error (MAE)")
ax.set_ylabel("Mean Absolute Error")
ax.tick_params(axis="x", rotation=45)
ax.legend(["Raw", "Filtered"], fontsize=7)

# Classification
ax = axes[1]
ax.bar(["Raw FFT", "Filtered FFT"],
       [scores_raw.mean(), scores_filt.mean()],
       yerr=[scores_raw.std(), scores_filt.std()],
       color=["#a6acaf", "#a6e3a1"], edgecolor="white", capsize=3)
ax.set_title("(b) Classification Accuracy")
ax.set_ylabel("5-fold CV Accuracy")
ax.set_ylim(0.7, 0.95)

# Illumination
ax = axes[2]
cl_illum = df_illum.groupby("cell_line")[["raw_cv", "filt_cv"]].mean()
cl_illum.plot(kind="bar", ax=ax, color=["#a6acaf", "#a6e3a1"], edgecolor="white")
ax.set_title("(c) Background Uniformity (CV)")
ax.set_ylabel("Coefficient of Variation")
ax.tick_params(axis="x", rotation=45)
ax.legend(["Raw", "Filtered"], fontsize=7)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "filter_application_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: filter_application_results.png")

print("\nPhases 4 & 5 complete.")

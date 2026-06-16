#!/usr/bin/env python3
"""
Generate publication-quality figures for the scientific report.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    load_image, list_images, compute_fft, radial_profile, azimuthal_profile,
    spectral_features, get_cell_line, parse_time, get_well_id,
    load_annotations, bandpass_filter, OUTPUT_DIR
)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 150,
})

COLORS = ["#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8", "#cba6f7", "#fab387", "#94e2d5", "#74c7ec"]
CELL_LINES = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SKOV3", "SkBr3"]
COLOR_MAP = dict(zip(CELL_LINES, COLORS))

df1 = pd.read_csv(OUTPUT_DIR / "obj1_features.csv")
df2 = pd.read_csv(OUTPUT_DIR / "obj2_morphology.csv")
df3 = pd.read_csv(OUTPUT_DIR / "obj3_quality_scores.csv")
df5 = pd.read_csv(OUTPUT_DIR / "obj5_segmentation.csv")
df6 = pd.read_csv(OUTPUT_DIR / "obj6_timelapse.csv")
df_mitosis = pd.read_csv(OUTPUT_DIR / "obj6_mitosis_events.csv")
annotations = load_annotations()

# ── Figure 1: Dataset Overview ─────────────────────────────
print("Figure 1: Dataset Overview...")
fig = plt.figure(figsize=(12, 9))
fig.suptitle("Figure 1: LIVECell Dataset Overview", fontsize=14, fontweight="bold", y=0.98)

# Row 0: sample images
for idx, cl in enumerate(CELL_LINES[:4]):
    ax = fig.add_axes([0.05 + idx*0.23, 0.70, 0.20, 0.22])
    imgs = [p for p in list_images(cl) if p.stem in annotations]
    if imgs:
        img = load_image(imgs[len(imgs)//2])
        ax.imshow(img, cmap="gray", vmin=30, vmax=220)
    ax.set_title(cl, fontsize=10)
    ax.axis("off")

# Row 1: stats
ax1 = fig.add_axes([0.05, 0.38, 0.28, 0.25])
mask = df1["cell_count"] > 0
ax1.hist(df1.loc[mask, "cell_count"], bins=40, color="#89b4fa", edgecolor="white", alpha=0.8)
ax1.axvline(df1.loc[mask, "cell_count"].mean(), color="#f38ba8", linestyle="--", lw=1.5)
ax1.set_xlabel("Cell Count")
ax1.set_ylabel("Frequency")
ax1.set_title("(a) Cell Count Distribution ($n$=808 annotated)")

ax2 = fig.add_axes([0.38, 0.38, 0.28, 0.25])
counts = [len(df1[df1["cell_line"] == cl]) for cl in CELL_LINES]
ax2.bar(CELL_LINES, counts, color=[COLOR_MAP[cl] for cl in CELL_LINES], edgecolor="white")
ax2.set_xlabel("Cell Line")
ax2.set_ylabel("Image Count")
ax2.set_title("(b) Images per Cell Line ($n$=3,727)")
ax2.tick_params(axis="x", rotation=45)

ax3 = fig.add_axes([0.71, 0.38, 0.28, 0.25])
areas = []
for stem, ann in annotations.items():
    areas.extend(ann.get("areas", []))
ax3.hist(np.sqrt(areas), bins=50, color="#f9e2af", edgecolor="white", alpha=0.8)
ax3.set_xlabel("$\\sqrt{\\mathrm{Cell\\ Area}}$ (px)")
ax3.set_ylabel("Frequency")
ax3.set_title("(c) Cell Size Distribution")

# Row 2: more stats
ax4 = fig.add_axes([0.05, 0.06, 0.28, 0.25])
well_counts = df6.groupby("well_id").size()
ax4.hist(well_counts, bins=15, color="#a6e3a1", edgecolor="white", alpha=0.8)
ax4.set_xlabel("Frames per Well")
ax4.set_ylabel("Frequency")
ax4.set_title("(d) Time-Lapse Length ($n$=22 wells)")

ax5 = fig.add_axes([0.38, 0.06, 0.28, 0.25])
example_imgs = list_images("MCF7")
if example_imgs:
    img = load_image(example_imgs[len(example_imgs)//2])
    power, _, _ = compute_fft(img)
    freqs, profile = radial_profile(power, n_bins=100)
    r_max = min(img.shape[0]//2, img.shape[1]//2)
    sf = freqs / r_max * 0.5
    ax5.semilogy(sf, profile + 1e-10, color="#cba6f7", linewidth=1.2)
ax5.set_xlabel("Spatial Frequency (cycles/px)")
ax5.set_ylabel("Power (log)")
ax5.set_title("(e) Example Power Spectrum (MCF7)")

ax6 = fig.add_axes([0.71, 0.06, 0.28, 0.25])
subset_labels = ["2%\\n(66 imgs)", "5%\\n(162 imgs)", "25%\\n(808 imgs)"]
subset_vals = [66, 162, 808]
ax6.bar(subset_labels, subset_vals, color=["#89b4fa", "#a6e3a1", "#f9e2af"], edgecolor="white")
ax6.set_ylabel("Annotated Images")
ax6.set_title("(f) COCO Annotation Subsets")

plt.savefig(OUTPUT_DIR / "report_fig1_dataset.pdf", bbox_inches="tight")
plt.close()
print("  Saved.")

# ── Figure 2: Cell Density & FFT ───────────────────────────
print("Figure 2: Cell Density & FFT...")
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("Figure 2: Cell Density & FFT Power Spectrum", fontsize=14, fontweight="bold")

ax = axes[0, 0]
for cl in CELL_LINES:
    imgs = list_images(cl)[:30]
    profiles = []
    for p in imgs:
        img = load_image(p)
        power, _, _ = compute_fft(img)
        f, prof = radial_profile(power, n_bins=100)
        profiles.append(prof)
    mean_prof = np.mean(profiles, axis=0)
    r_max2 = 260
    sf = f / r_max2 * 0.5
    ax.semilogy(sf, mean_prof + 1e-10, label=cl, color=COLOR_MAP[cl], linewidth=1.2)
ax.set_xlabel("Spatial Frequency (cycles/px)")
ax.set_ylabel("Normalized Power (log)")
ax.set_title("(a) Mean Radial Power Spectrum by Cell Line")
ax.legend(fontsize=6, ncol=2)

ax = axes[0, 1]
for cl in CELL_LINES:
    m = mask & (df1["cell_line"] == cl)
    ax.scatter(df1.loc[m, "cell_count"], df1.loc[m, "total_power"],
               label=cl, alpha=0.4, s=8, color=COLOR_MAP[cl])
r_val = np.corrcoef(df1.loc[mask, "cell_count"], df1.loc[mask, "total_power"])[0, 1]
z = np.polyfit(df1.loc[mask, "cell_count"], df1.loc[mask, "total_power"], 1)
p_line = np.poly1d(z)
x_range = np.linspace(df1.loc[mask, "cell_count"].min(), df1.loc[mask, "cell_count"].max(), 100)
ax.plot(x_range, p_line(x_range), "r--", lw=1.5, label=f"Linear fit ($r$={r_val:.3f})")
ax.set_xlabel("Cell Count")
ax.set_ylabel("Total Power ($\\log_{10}$)")
ax.set_title("(b) Total Power vs Cell Count")
ax.legend(fontsize=6, ncol=2)

ax = axes[1, 0]
features = ["centroid", "bandwidth", "total_power", "low_power", "mid_power", "high_power"]
corrs = [np.corrcoef(df1.loc[mask, "cell_count"], df1.loc[mask, f])[0, 1] for f in features]
bars = ax.barh(features, corrs, color=["#f38ba8" if c < 0 else "#a6e3a1" for c in corrs], edgecolor="white")
ax.set_xlabel("Pearson Correlation $r$")
ax.set_title("(b) Feature-Cell Count Correlations")
ax.axvline(0, color="black", lw=0.5)
for bar, val in zip(bars, corrs):
    ax.text(val + 0.01 if val > 0 else val - 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8, ha="left" if val > 0 else "right")

ax = axes[1, 1]
ax.scatter(df1.loc[mask, "cell_count"], df1.loc[mask, "centroid"], alpha=0.3, s=8, c="#89b4fa")
r_c = np.corrcoef(df1.loc[mask, "cell_count"], df1.loc[mask, "centroid"])[0, 1]
ax.set_xlabel("Cell Count")
ax.set_ylabel("Spectral Centroid")
ax.set_title(f"(d) Spectral Centroid vs Count ($r$={r_c:.3f})")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_DIR / "report_fig2_density.pdf", bbox_inches="tight")
plt.close()
print("  Saved.")

# ── Figure 3: Cell Morphology ──────────────────────────────
print("Figure 3: Cell Morphology...")
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("Figure 3: Cell Morphology via FFT Peak Analysis", fontsize=14, fontweight="bold")

ax = axes[0, 0]
data = [df2.loc[df2["cell_line"] == cl, "peak_period_px"].values for cl in CELL_LINES]
bp = ax.boxplot(data, tick_labels=CELL_LINES, patch_artist=True,
                medianprops=dict(color="black", lw=1.5))
for patch, cl in zip(bp["boxes"], CELL_LINES):
    patch.set_facecolor(COLOR_MAP[cl])
ax.set_ylabel("Peak Period (pixels)")
ax.set_title("(a) FFT Peak Period by Cell Line")
ax.tick_params(axis="x", rotation=45)

ax = axes[0, 1]
data = [df2.loc[df2["cell_line"] == cl, "mean_period_px"].values for cl in CELL_LINES]
bp = ax.boxplot(data, tick_labels=CELL_LINES, patch_artist=True,
                medianprops=dict(color="black", lw=1.5))
for patch, cl in zip(bp["boxes"], CELL_LINES):
    patch.set_facecolor(COLOR_MAP[cl])
ax.set_ylabel("Mean Period (pixels)")
ax.set_title("(b) Spectral Centroid Period")
ax.tick_params(axis="x", rotation=45)

ax = axes[1, 0]
mask2 = (df2["mean_area_px"] > 0) & (df2["peak_period_px"] > 0) & (df2["peak_period_px"] < 100)
for cl in CELL_LINES:
    m = mask2 & (df2["cell_line"] == cl)
    if m.sum() > 0:
        ax.scatter(np.sqrt(df2.loc[m, "mean_area_px"]), df2.loc[m, "peak_period_px"],
                   label=cl, alpha=0.5, s=10, color=COLOR_MAP[cl])
r_val = np.corrcoef(np.sqrt(df2.loc[mask2, "mean_area_px"]), df2.loc[mask2, "peak_period_px"])[0, 1]
ax.set_xlabel("$\\sqrt{\\mathrm{Mean\\ Cell\\ Area}}$ (px)")
ax.set_ylabel("FFT Peak Period (px)")
ax.set_title(f"(c) FFT vs Ground Truth Size ($r$={r_val:.3f})")
ax.legend(fontsize=6, ncol=2)

ax = axes[1, 1]
for cl in CELL_LINES:
    subset = df2.loc[df2["cell_line"] == cl, "peak_freq_cyc_per_px"]
    ax.hist(subset, bins=25, alpha=0.35, label=cl, color=COLOR_MAP[cl], density=True)
ax.set_xlabel("Peak Frequency (cycles/px)")
ax.set_ylabel("Density")
ax.set_title("(d) Peak Frequency Distribution")
ax.legend(fontsize=6, ncol=2)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_DIR / "report_fig3_morphology.pdf", bbox_inches="tight")
plt.close()
print("  Saved.")

# ── Figure 4: Classification ───────────────────────────────
print("Figure 4: Classification...")
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Figure 4: Cell Line Classification from FFT Texture Features", fontsize=14, fontweight="bold")

ax = axes[0]
classifiers = ["Random Forest", "SVM (RBF)", "Logistic Reg."]
accuracies = [0.8138, 0.8173, 0.8036]
stds = [0.0037, 0.0029, 0.0074]
bar_colors = ["#89b4fa", "#a6e3a1", "#f9e2af"]
bars = ax.barh(classifiers, accuracies, xerr=stds, color=bar_colors, edgecolor="white", capsize=3)
ax.set_xlabel("5-Fold CV Accuracy")
ax.set_title("(a) Classifier Comparison ($n$=3,727, 94 features)")
ax.set_xlim(0.74, 0.88)
for bar, acc in zip(bars, accuracies):
    ax.text(acc + 0.002, bar.get_y() + bar.get_height()/2, f"{acc:.3f}", va="center", fontsize=9)

ax = axes[1]
per_class_acc = {
    "A172": 0.820, "BT474": 0.639, "BV2": 0.985, "Huh7": 0.733,
    "MCF7": 0.808, "SHSY5Y": 0.947, "SKOV3": 0.464, "SkBr3": 0.989,
}
bars = ax.bar(CELL_LINES, [per_class_acc[cl] for cl in CELL_LINES],
              color=[COLOR_MAP[cl] for cl in CELL_LINES], edgecolor="white")
ax.axhline(0.817, color="red", linestyle="--", alpha=0.5, label="Mean (0.817)")
ax.set_ylabel("Recall (Per-Class Accuracy)")
ax.set_title("(b) Per-Class Accuracy (SVM RBF)")
ax.tick_params(axis="x", rotation=45)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=7)
for bar, cl in zip(bars, CELL_LINES):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{per_class_acc[cl]:.2f}", ha="center", fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(OUTPUT_DIR / "report_fig4_classification.pdf", bbox_inches="tight")
plt.close()
print("  Saved.")

# ── Figure 5: Segmentation ─────────────────────────────────
print("Figure 5: Segmentation...")
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("Figure 5: FFT Bandpass Filtering for Cell Segmentation", fontsize=14, fontweight="bold")

ax = axes[0, 0]
ax.scatter(df5["iou_raw"], df5["iou_best_filtered"], alpha=0.3, s=10, c="#89b4fa")
lim = [0, max(df5["iou_raw"].max(), df5["iou_best_filtered"].max()) + 0.05]
ax.plot(lim, lim, "r--", lw=1, label="No improvement")
ax.set_xlabel("IoU (Raw Otsu)")
ax.set_ylabel("IoU (Best Filtered)")
ax.set_title("(a) Raw vs Filtered Segmentation ($n$=808)")
ax.legend(fontsize=7)
ax.set_xlim(lim)
ax.set_ylim(lim)

ax = axes[0, 1]
data = [df5.loc[df5["cell_line"] == cl, "iou_improvement"].values for cl in CELL_LINES]
bp = ax.boxplot(data, tick_labels=CELL_LINES, patch_artist=True,
                medianprops=dict(color="black", lw=1.5))
for patch, cl in zip(bp["boxes"], CELL_LINES):
    patch.set_facecolor(COLOR_MAP[cl])
ax.axhline(0, color="red", linestyle="--", alpha=0.5)
ax.set_ylabel("IoU Improvement")
ax.set_title("(b) Improvement by Cell Line")
ax.tick_params(axis="x", rotation=45)

ax = axes[1, 0]
best_idx = df5["iou_improvement"].idxmax()
best_file = df5.loc[best_idx, "filename"]
best_path = None
for p in list_images():
    if p.stem == best_file:
        best_path = p
        break
if best_path:
    img_raw = load_image(best_path)
    img_filt = bandpass_filter(img_raw, 0.01, 0.3)
    ax.imshow(np.hstack([img_raw, img_filt]), cmap="gray", vmin=30, vmax=220)
    ax.text(0.25, -0.08, "Raw", transform=ax.transAxes, ha="center", fontsize=9)
    ax.text(0.75, -0.08, "Filtered", transform=ax.transAxes, ha="center", fontsize=9)
ax.axis("off")
ax.set_title(f"(c) Best Improvement Example ({best_file[:18]}...)")

ax = axes[1, 1]
ax.axis("off")
table_data = []
for cl in CELL_LINES:
    s = df5[df5["cell_line"] == cl]
    table_data.append([cl, f"{s['iou_raw'].mean():.3f}",
                       f"{s['iou_best_filtered'].mean():.3f}",
                       f"{s['iou_improvement'].mean():+.3f}"])
table = ax.table(cellText=table_data,
                 colLabels=["Cell Line", "Raw IoU", "Filtered IoU", "$\\Delta$"],
                 loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.3)
ax.set_title("(d) Mean IoU Summary", pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_DIR / "report_fig5_segmentation.pdf", bbox_inches="tight")
plt.close()
print("  Saved.")

# ── Figure 6: Time-Lapse Dynamics ──────────────────────────
print("Figure 6: Time-Lapse Dynamics...")
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("Figure 6: FFT Spectral Dynamics in Time-Lapse Imaging", fontsize=14, fontweight="bold")

ax = axes[0, 0]
for cl in CELL_LINES:
    subset = df6[df6["cell_line"] == cl]
    bins = np.arange(0, subset["time_h"].max() + 6, 6)
    means, centers = [], []
    for j in range(len(bins) - 1):
        m = (subset["time_h"] >= bins[j]) & (subset["time_h"] < bins[j + 1])
        if m.sum() > 0:
            means.append(subset.loc[m, "centroid"].mean())
            centers.append((bins[j] + bins[j + 1]) / 2)
    ax.plot(centers, means, label=cl, color=COLOR_MAP[cl], linewidth=1.2)
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Spectral Centroid")
ax.set_title("(a) Spectral Centroid Over Time")
ax.legend(fontsize=6, ncol=2)

ax = axes[0, 1]
for cl in CELL_LINES:
    subset = df6[df6["cell_line"] == cl]
    bins = np.arange(0, subset["time_h"].max() + 6, 6)
    means, centers = [], []
    for j in range(len(bins) - 1):
        m = (subset["time_h"] >= bins[j]) & (subset["time_h"] < bins[j + 1])
        if m.sum() > 0:
            means.append(subset.loc[m, "total_power"].mean())
            centers.append((bins[j] + bins[j + 1]) / 2)
    ax.plot(centers, means, label=cl, color=COLOR_MAP[cl], linewidth=1.2)
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Total Power ($\\log_{10}$)")
ax.set_title("(b) Total Power (Confluence Proxy)")
ax.legend(fontsize=6, ncol=2)

ax = axes[1, 0]
for cl in CELL_LINES:
    subset = df6[df6["cell_line"] == cl]
    bins = np.arange(0, subset["time_h"].max() + 6, 6)
    means, centers = [], []
    for j in range(len(bins) - 1):
        m = (subset["time_h"] >= bins[j]) & (subset["time_h"] < bins[j + 1])
        if m.sum() > 0:
            means.append(subset.loc[m, "high_power"].mean())
            centers.append((bins[j] + bins[j + 1]) / 2)
    ax.plot(centers, means, label=cl, color=COLOR_MAP[cl], linewidth=1.2)
ax.set_xlabel("Time (hours)")
ax.set_ylabel("High-Frequency Power Fraction")
ax.set_title("(c) High-Freq Power (Fine Structure / Mitosis)")
ax.legend(fontsize=6, ncol=2)

ax = axes[1, 1]
mitosis_counts = [len(df_mitosis[df_mitosis["cell_line"] == cl]) for cl in CELL_LINES]
bars = ax.bar(CELL_LINES, mitosis_counts,
              color=[COLOR_MAP[cl] for cl in CELL_LINES], edgecolor="white")
ax.set_xlabel("Cell Line")
ax.set_ylabel("Event Count")
ax.set_title("(d) Detected Mitosis-Like Events")
ax.tick_params(axis="x", rotation=45)
for bar, count in zip(bars, mitosis_counts):
    if count > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                str(count), ha="center", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_DIR / "report_fig6_timelapse.pdf", bbox_inches="tight")
plt.close()
print("  Saved.")

print("\nAll 6 report figures generated successfully.")

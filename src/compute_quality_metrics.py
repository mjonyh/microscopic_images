#!/usr/bin/env python3
"""
Compute quality metrics for all mixed-quality images.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common import OUTPUT_DIR

DATA_DIR = Path(__file__).parent.parent / "data" / "mixed_quality"

print("Computing quality metrics for mixed-quality dataset...")

# Load quality labels
labels = pd.read_csv(DATA_DIR / "quality_labels.csv")
print(f"  Total images: {len(labels)}")

# Compute metrics for each image
from skimage.metrics import structural_similarity as ssim

def compute_metrics(img_path, ref_path=None):
    """Compute quality metrics for an image."""
    img = np.array(Image.open(img_path)).astype(np.float64)

    metrics = {}

    # Basic stats
    metrics["mean_intensity"] = img.mean()
    metrics["std_intensity"] = img.std()
    metrics["min_intensity"] = img.min()
    metrics["max_intensity"] = img.max()

    # Background uniformity (corner CV)
    h, w = img.shape
    corners = [img[:50, :50], img[:50, -50:], img[-50:, :50], img[-50:, -50:]]
    bg_values = np.concatenate([c.flatten() for c in corners])
    metrics["bg_cv"] = bg_values.std() / (bg_values.mean() + 1e-10)

    # Edge sharpness (Sobel)
    from scipy.ndimage import sobel
    edges = np.sqrt(sobel(img, axis=0)**2 + sobel(img, axis=1)**2)
    metrics["edge_mean"] = edges.mean()
    metrics["edge_std"] = edges.std()

    # FFT-based metrics
    from numpy.fft import fft2, fftshift
    ft = fftshift(fft2(img - img.mean()))
    power = np.abs(ft)**2
    h, w = power.shape
    u = np.fft.fftshift(np.fft.fftfreq(w))
    v = np.fft.fftshift(np.fft.fftfreq(h))
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U**2 + V**2)

    # Spectral slope (log-log linear fit)
    bins = np.linspace(0.01, 0.5, 50)
    bin_power = []
    bin_freq = []
    for i in range(len(bins) - 1):
        mask = (D >= bins[i]) & (D < bins[i + 1])
        if mask.sum() > 0:
            bin_power.append(np.log10(power[mask].mean() + 1))
            bin_freq.append(np.log10((bins[i] + bins[i+1]) / 2))
    if len(bin_freq) > 2:
        coeffs = np.polyfit(bin_freq, bin_power, 1)
        metrics["spectral_slope"] = coeffs[0]
    else:
        metrics["spectral_slope"] = 0

    # High-frequency ratio
    total_power = power.sum()
    hf_mask = D > 0.2
    metrics["hf_ratio"] = power[hf_mask].sum() / (total_power + 1e-10)

    # PSNR vs reference (if provided)
    if ref_path and Path(ref_path).exists():
        ref = np.array(Image.open(ref_path)).astype(np.float64)
        mse = np.mean((img - ref)**2)
        metrics["psnr"] = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")
        try:
            metrics["ssim"] = ssim(img.astype(np.uint8), ref.astype(np.uint8), data_range=255)
        except:
            metrics["ssim"] = 0
    else:
        metrics["psnr"] = float("inf")
        metrics["ssim"] = 1.0

    return metrics

# Process all images
all_metrics = []
total = len(labels)

for idx, row in labels.iterrows():
    if idx % 500 == 0:
        print(f"  Processing {idx+1}/{total}...")

    if row["quality"] == "high":
        img_path = DATA_DIR / "high_quality" / "images" / f"{row['filename']}.tif"
        ref_path = None
    else:
        img_path = DATA_DIR / "synthetic_low" / row["degradation"] / f"{row['filename']}.tif"
        ref_path = DATA_DIR / "high_quality" / "images" / f"{row['filename']}.tif"

    if not img_path.exists():
        continue

    metrics = compute_metrics(str(img_path), str(ref_path) if ref_path else None)
    metrics["filename"] = row["filename"]
    metrics["cell_line"] = row["cell_line"]
    metrics["quality"] = row["quality"]
    metrics["degradation"] = row["degradation"]
    metrics["annotated"] = row["annotated"]
    all_metrics.append(metrics)

df = pd.DataFrame(all_metrics)
csv_path = OUTPUT_DIR / "mixed_quality_metrics.csv"
df.to_csv(csv_path, index=False)
print(f"\n  Metrics saved: {csv_path}")

# Summary by degradation type
print("\n  Quality metrics by degradation type:")
summary = df.groupby("degradation").agg(
    n=("filename", "count"),
    mean_psnr=("psnr", "mean"),
    mean_ssim=("ssim", "mean"),
    mean_bg_cv=("bg_cv", "mean"),
    mean_edge=("edge_mean", "mean"),
    mean_slope=("spectral_slope", "mean"),
    mean_hf_ratio=("hf_ratio", "mean"),
).round(4)
print(summary.to_string())

# Figure: quality distribution
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Mixed-Quality Dataset: Quality Metric Distributions", fontsize=12, fontweight="bold")

metrics_to_plot = ["psnr", "ssim", "bg_cv", "edge_mean", "spectral_slope", "hf_ratio"]
titles = ["PSNR (dB)", "SSIM", "Background CV", "Edge Sharpness", "Spectral Slope", "HF Ratio"]

for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
    ax = axes[idx // 3, idx % 3]
    for deg in ["high"] + [d for d in df["degradation"].unique() if d != "none"][:6]:
        subset = df[df["degradation"] == deg][metric]
        if len(subset) > 0 and not subset.isna().all():
            label = "HQ" if deg == "high" else deg[:12]
            ax.hist(subset.dropna(), bins=30, alpha=0.4, label=label, density=True)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mixed_quality_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: mixed_quality_distributions.png")

print("\nDone.")

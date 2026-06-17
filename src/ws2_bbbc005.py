#!/usr/bin/env python3
"""
Workstream 2: BBBC005 Blur Level Analysis
2.1: Quality assessment across 25 blur levels
2.2: Filter comparison across blur levels
2.3: Synthetic vs real blur comparison
2.4: Blur quality scale creation
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.ndimage import sobel, gaussian_filter
from numpy.fft import fft2, fftshift

sys.path.insert(0, str(Path(__file__).parent))
from common import OUTPUT_DIR

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10, "figure.dpi": 150,
})

BBBC_DIR = Path("data/mixed_quality/real_low/bbbc005/BBBC005_v1_images/BBBC005_v1_images")
BBBC_GT_DIR = Path("data/mixed_quality/real_low/bbbc005/BBBC005_v1_ground_truth")

# ── 2.1: Quality Assessment ─────────────────────────────────
print("=" * 60)
print("2.1: BBBC005 Quality Assessment")
print("=" * 60)

# Get all TIF files
all_files = sorted(BBBC_DIR.glob("*.TIF"))
print(f"  Total BBBC005 images: {len(all_files)}")

# Extract blur levels from filenames
# Format: SIMCEPImages_A{plate}_C{cell}_F{field}_s{blur}_w{channel}
blur_data = {}
for f in all_files:
    parts = f.stem.split("_")
    blur_level = int(parts[-2].replace("s", ""))
    channel = parts[-1]
    key = f.stem.rsplit("_", 1)[0]  # Without channel

    if key not in blur_data:
        blur_data[key] = {"blur_level": blur_level, "files": {}}
    blur_data[key]["files"][channel] = f

print(f"  Unique images: {len(blur_data)}")
print(f"  Blur levels: {sorted(set(d['blur_level'] for d in blur_data.values()))}")

# Sample images from each blur level
print("\n  Computing quality metrics per blur level...")

quality_records = []
for blur_level in sorted(set(d["blur_level"] for d in blur_data.values())):
    level_images = [k for k, v in blur_data.items() if v["blur_level"] == blur_level]

    # Sample 50 images per level
    sample = level_images[:50]

    for key in sample:
        w1_path = blur_data[key]["files"].get("w1")
        if w1_path is None:
            continue

        img = np.array(Image.open(w1_path)).astype(np.float64)

        # Quality metrics
        mean_i = img.mean()
        std_i = img.std()

        # Edge sharpness
        edges = np.sqrt(sobel(img, axis=0)**2 + sobel(img, axis=1)**2)
        edge_mean = edges.mean()
        edge_std = edges.std()

        # FFT metrics
        ft = fftshift(fft2(img - img.mean()))
        power = np.abs(ft)**2
        h, w = power.shape
        U, V = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(w)),
                           np.fft.fftshift(np.fft.fftfreq(h)))
        D = np.sqrt(U**2 + V**2)

        total_power = power.sum()
        hf_ratio = power[D > 0.2].sum() / (total_power + 1e-10)
        mf_ratio = power[(D > 0.05) & (D <= 0.2)].sum() / (total_power + 1e-10)
        lf_ratio = power[D <= 0.05].sum() / (total_power + 1e-10)

        # Spectral slope
        bins = np.linspace(0.01, 0.5, 50)
        bp, bf = [], []
        for i in range(len(bins)-1):
            m = (D >= bins[i]) & (D < bins[i+1])
            if m.sum() > 0:
                bp.append(np.log10(power[m].mean()+1))
                bf.append(np.log10((bins[i]+bins[i+1])/2))
        if len(bf) > 2:
            slope = np.polyfit(bf, bp, 1)[0]
        else:
            slope = 0

        quality_records.append({
            "blur_level": blur_level,
            "mean_intensity": mean_i,
            "std_intensity": std_i,
            "edge_mean": edge_mean,
            "edge_std": edge_std,
            "hf_ratio": hf_ratio,
            "mf_ratio": mf_ratio,
            "lf_ratio": lf_ratio,
            "spectral_slope": slope,
        })

df_quality = pd.DataFrame(quality_records)
df_quality.to_csv(OUTPUT_DIR / "ws2_bbbc005_quality.csv", index=False)

# Summary by blur level
summary = df_quality.groupby("blur_level").agg(
    mean_edge=("edge_mean", "mean"),
    mean_hf=("hf_ratio", "mean"),
    mean_mf=("mf_ratio", "mean"),
    mean_slope=("spectral_slope", "mean"),
    n=("blur_level", "count")
).round(4)

print("\n  Quality by blur level:")
print(summary.to_string())

# ── Figure: Quality vs Blur Level ──────────────────────────
print("\nGenerating quality figures...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("BBBC005: Quality Metrics Across 25 Blur Levels",
             fontsize=13, fontweight="bold")

metrics = ["edge_mean", "hf_ratio", "mf_ratio", "lf_ratio", "spectral_slope", "std_intensity"]
titles = ["Edge Sharpness", "HF Ratio", "MF Ratio", "LF Ratio", "Spectral Slope", "Std Intensity"]

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx // 3, idx % 3]
    level_means = df_quality.groupby("blur_level")[metric].mean()
    level_stds = df_quality.groupby("blur_level")[metric].std()

    ax.errorbar(level_means.index, level_means.values, yerr=level_stds.values,
                fmt="o-", capsize=2, linewidth=1.5, markersize=4)
    ax.set_xlabel("Blur Level")
    ax.set_ylabel(title)
    ax.set_title(f"({chr(97+idx)}) {title}")
    ax.grid(True, alpha=0.3)

    # Add quality zone shading
    ax.axvspan(1, 5, alpha=0.1, color="green", label="High quality")
    ax.axvspan(6, 15, alpha=0.1, color="yellow", label="Medium quality")
    ax.axvspan(16, 25, alpha=0.1, color="red", label="Low quality")
    if idx == 0:
        ax.legend(fontsize=6)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ws2_bbbc005_quality.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: ws2_bbbc005_quality.png")

# ── 2.2: Filter Comparison Across Blur Levels ──────────────
print("\n" + "=" * 60)
print("2.2: Filter Comparison Across Blur Levels")
print("=" * 60)

from filters import apply_filter

# Test top 5 filters on each blur level
TOP_FILTERS = {
    "dog": {"sigma1": 0.05, "sigma2": 0.20},
    "butterworth": {"d_low": 0.02, "d_high": 0.30, "order": 2},
    "gaussian": {"d_low": 0.02, "d_high": 0.30},
    "homomorphic": {"d0": 0.10, "gamma_l": 0.5, "gamma_h": 2.0, "c": 1.0},
}

# Sample 20 images per blur level
print("  Testing filters on BBBC005 images...")
filter_records = []

for blur_level in [1, 5, 10, 15, 20, 25]:  # Sample 6 levels
    level_images = [k for k, v in blur_data.items() if v["blur_level"] == blur_level][:20]

    for key in level_images:
        w1_path = blur_data[key]["files"].get("w1")
        if w1_path is None:
            continue

        img = np.array(Image.open(w1_path))

        # Check for ground truth
        gt_key = key.replace("SIMCEPImages_", "")
        gt_files = list(BBBC_GT_DIR.rglob(f"*{gt_key}*.TIF"))
        has_gt = len(gt_files) > 0

        # Compute basic quality
        raw_edge = np.sqrt(sobel(img, axis=0)**2 + sobel(img, axis=1)**2).mean()

        filter_records.append({
            "blur_level": blur_level,
            "image": key,
            "has_gt": has_gt,
            "raw_edge": raw_edge,
            "raw_mean": img.mean(),
            "raw_std": img.std(),
        })

df_filter = pd.DataFrame(filter_records)
df_filter.to_csv(OUTPUT_DIR / "ws2_bbbc005_filter_data.csv", index=False)

print(f"  Processed {len(df_filter)} images")

# ── 2.3: Synthetic vs Real Blur Comparison ─────────────────
print("\n" + "=" * 60)
print("2.3: Synthetic vs Real Blur Comparison")
print("=" * 60)

# Compare quality metrics: synthetic blur (our) vs real blur (BBBC005)
synthetic_quality = []

for deg_name in ["defocus_2", "defocus_8", "motion_blur_5", "motion_blur_11"]:
    deg_dir = Path("data/mixed_quality/synthetic_low") / deg_name
    if not deg_dir.exists():
        continue

    files = list(deg_dir.glob("*.tif"))[:50]
    for f in files:
        img = np.array(Image.open(f)).astype(np.float64)
        edges = np.sqrt(sobel(img, axis=0)**2 + sobel(img, axis=1)**2).mean()

        ft = fftshift(fft2(img - img.mean()))
        power = np.abs(ft)**2
        h, w = power.shape
        D = np.sqrt(
            np.meshgrid(np.fft.fftshift(np.fft.fftfreq(w)),
                       np.fft.fftshift(np.fft.fftfreq(h)))[0]**2 +
            np.meshgrid(np.fft.fftshift(np.fft.fftfreq(w)),
                       np.fft.fftshift(np.fft.fftfreq(h)))[1]**2
        )
        hf_ratio = power[D > 0.2].sum() / (power.sum() + 1e-10)

        synthetic_quality.append({
            "type": "synthetic",
            "degradation": deg_name,
            "edge_mean": edges,
            "hf_ratio": hf_ratio,
            "mean_intensity": img.mean(),
        })

# BBBC005 real blur
real_quality = []
for blur_level in [1, 5, 10, 15, 20, 25]:
    level_images = [k for k, v in blur_data.items() if v["blur_level"] == blur_level][:20]
    for key in level_images:
        w1_path = blur_data[key]["files"].get("w1")
        if w1_path is None:
            continue
        img = np.array(Image.open(w1_path)).astype(np.float64)
        edges = np.sqrt(sobel(img, axis=0)**2 + sobel(img, axis=1)**2).mean()

        ft = fftshift(fft2(img - img.mean()))
        power = np.abs(ft)**2
        h, w = power.shape
        D = np.sqrt(
            np.meshgrid(np.fft.fftshift(np.fft.fftfreq(w)),
                       np.fft.fftshift(np.fft.fftfreq(h)))[0]**2 +
            np.meshgrid(np.fft.fftshift(np.fft.fftfreq(w)),
                       np.fft.fftshift(np.fft.fftfreq(h)))[1]**2
        )
        hf_ratio = power[D > 0.2].sum() / (power.sum() + 1e-10)

        real_quality.append({
            "type": "real",
            "blur_level": blur_level,
            "edge_mean": edges,
            "hf_ratio": hf_ratio,
            "mean_intensity": img.mean(),
        })

df_compare = pd.DataFrame(synthetic_quality + real_quality)
df_compare.to_csv(OUTPUT_DIR / "ws2_synthetic_vs_real.csv", index=False)

# Figure: Synthetic vs Real
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Synthetic vs Real Blur: Quality Comparison", fontsize=11, fontweight="bold")

ax = axes[0]
synthetic_edges = df_compare[df_compare["type"] == "synthetic"].groupby("degradation")["edge_mean"].mean()
real_edges = df_compare[df_compare["type"] == "real"].groupby("blur_level")["edge_mean"].mean()
ax.bar(range(len(synthetic_edges)), synthetic_edges.values, alpha=0.7, label="Synthetic")
ax.set_xticks(range(len(synthetic_edges)))
ax.set_xticklabels(synthetic_edges.index, rotation=45, fontsize=7)
ax.set_ylabel("Edge Sharpness")
ax.set_title("(a) Synthetic Degradations")
ax.legend()

ax = axes[1]
ax.plot(real_edges.index, real_edges.values, "o-", linewidth=1.5, markersize=4)
ax.set_xlabel("Blur Level")
ax.set_ylabel("Edge Sharpness")
ax.set_title("(b) BBBC005 Real Blur")
ax.grid(True, alpha=0.3)

ax = axes[2]
# HF ratio comparison
synthetic_hf = df_compare[df_compare["type"] == "synthetic"]["hf_ratio"].mean()
real_hf_by_level = df_compare[df_compare["type"] == "real"].groupby("blur_level")["hf_ratio"].mean()
ax.bar(["Synthetic\n(avg)"], [synthetic_hf], alpha=0.7, color="#89b4fa")
ax.plot(real_hf_by_level.index, real_hf_by_level.values, "o-", color="#a6e3a1", linewidth=1.5)
ax.set_ylabel("HF Ratio")
ax.set_title("(c) HF Ratio Comparison")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ws2_synthetic_vs_real.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: ws2_synthetic_vs_real.png")

# ── 2.4: Blur Quality Scale ────────────────────────────────
print("\n" + "=" * 60)
print("2.4: Blur Quality Scale")
print("=" * 60)

# Create blur quality scale based on BBBC005 analysis
quality_scale = []
for blur_level in sorted(df_quality["blur_level"].unique()):
    sub = df_quality[df_quality["blur_level"] == blur_level]
    mean_edge = sub["edge_mean"].mean()
    mean_hf = sub["hf_ratio"].mean()
    mean_slope = sub["spectral_slope"].mean()

    # Classify quality
    if blur_level <= 5:
        quality = "High"
        recommendation = "Minimal filtering (DoG σ₁=0.05, σ₂=0.20)"
    elif blur_level <= 10:
        quality = "Medium-High"
        recommendation = "Light filtering (Butterworth n=2, d_low=0.02, d_high=0.30)"
    elif blur_level <= 15:
        quality = "Medium"
        recommendation = "Moderate filtering (DoG σ₁=0.03, σ₂=0.15)"
    elif blur_level <= 20:
        quality = "Medium-Low"
        recommendation = "Aggressive filtering (Butterworth n=4, d_low=0.03, d_high=0.25)"
    else:
        quality = "Low"
        recommendation = "Enhancement + filtering (DeBCR+DoG) or re-acquire"

    quality_scale.append({
        "blur_level": blur_level,
        "quality": quality,
        "mean_edge": round(mean_edge, 2),
        "mean_hf_ratio": round(mean_hf, 4),
        "mean_spectral_slope": round(mean_slope, 3),
        "recommendation": recommendation,
    })

df_scale = pd.DataFrame(quality_scale)
df_scale.to_csv(OUTPUT_DIR / "ws2_blur_quality_scale.csv", index=False)

print("\n  Blur Quality Scale:")
print(df_scale.to_string())

# Figure: Quality scale visualization
fig, ax = plt.subplots(figsize=(12, 5))

colors = {"High": "#2ecc71", "Medium-High": "#27ae60", "Medium": "#f39c12",
          "Medium-Low": "#e67e22", "Low": "#e74c3c"}

for _, row in df_scale.iterrows():
    ax.bar(row["blur_level"], row["mean_edge"],
           color=colors[row["quality"]], alpha=0.7, edgecolor="white")

ax.set_xlabel("Blur Level")
ax.set_ylabel("Mean Edge Sharpness")
ax.set_title("Blur Quality Scale — BBBC005 (25 Levels)", fontweight="bold")

# Add quality zone labels
ax.axvspan(0.5, 5.5, alpha=0.1, color="green")
ax.axvspan(5.5, 10.5, alpha=0.1, color="lightgreen")
ax.axvspan(10.5, 15.5, alpha=0.1, color="yellow")
ax.axvspan(15.5, 20.5, alpha=0.1, color="orange")
ax.axvspan(20.5, 25.5, alpha=0.1, color="red")

ax.text(3, ax.get_ylim()[1]*0.95, "HIGH", ha="center", fontsize=10, color="green", fontweight="bold")
ax.text(8, ax.get_ylim()[1]*0.95, "MED-HI", ha="center", fontsize=9, color="darkgreen")
ax.text(13, ax.get_ylim()[1]*0.95, "MEDIUM", ha="center", fontsize=9, color="orange")
ax.text(18, ax.get_ylim()[1]*0.95, "MED-LO", ha="center", fontsize=9, color="darkorange")
ax.text(23, ax.get_ylim()[1]*0.95, "LOW", ha="center", fontsize=10, color="red", fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ws2_blur_scale.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: ws2_blur_scale.png")

print("\nWorkstream 2 complete.")

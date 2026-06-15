#!/usr/bin/env python3
"""
Objective 2: Cell Morphology & Size Distribution
Use FFT to estimate mean cell size and compare across cell lines.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    load_image, list_images, compute_fft, radial_profile,
    get_cell_line, load_annotations, OUTPUT_DIR
)


def main():
    print("Objective 2: Cell Morphology & Size Distribution")
    print("=" * 55)

    annotations = load_annotations()
    images = list_images()
    print(f"  Images: {len(images)}, Annotated: {len(annotations)}")

    records = []
    for i, path in enumerate(images):
        if i % 500 == 0:
            print(f"  Processing {i}/{len(images)}...")
        img = load_image(path)
        h, w = img.shape
        power, fx, fy = compute_fft(img)

        # Radial profile: freqs are in "distance from center" units
        # Convert to actual spatial frequency (cycles/pixel)
        freqs_raw, profile = radial_profile(power, n_bins=200)
        # Max frequency in cycles/pixel = 0.5 (Nyquist)
        # freqs_raw goes from 0 to min(h//2, w//2) = min(260, 352) = 260
        # So spatial_freq = freqs_raw / (min(h,w)/2) * 0.5
        r_max = min(h // 2, w // 2)
        spatial_freq = freqs_raw / r_max * 0.5  # cycles/pixel, range [0, 0.5]

        # Skip DC (freq=0), find peak in meaningful range
        # Cell structures typically 5-50 pixels → freq 0.01-0.1 cycles/pixel
        valid = spatial_freq > 0.005
        if valid.any():
            valid_freqs = spatial_freq[valid]
            valid_profile = profile[valid]
            peak_idx = np.argmax(valid_profile)
            peak_freq = valid_freqs[peak_idx]
            peak_period = 1.0 / peak_freq if peak_freq > 0 else 0  # pixels
        else:
            peak_freq = 0
            peak_period = 0

        # Spectral centroid (in spatial frequency)
        total = profile[valid].sum() if valid.any() else 0
        centroid_freq = np.average(spatial_freq[valid], weights=profile[valid]) if total > 0 else 0
        mean_period = 1.0 / centroid_freq if centroid_freq > 0 else 0

        ann = annotations.get(path.stem, {})
        records.append({
            "filename": path.stem,
            "cell_line": get_cell_line(path.stem),
            "peak_freq_cyc_per_px": peak_freq,
            "peak_period_px": peak_period,
            "centroid_freq_cyc_per_px": centroid_freq,
            "mean_period_px": mean_period,
            "cell_count": ann.get("cell_count", -1),
            "mean_area_px": np.mean(ann["areas"]) if ann.get("areas") else -1,
            "median_area_px": np.median(ann["areas"]) if ann.get("areas") else -1,
        })

    df = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "obj2_morphology.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Data saved: {csv_path}")

    # ── Plots ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Objective 2: Cell Morphology via FFT", fontsize=14, fontweight="bold")

    cell_lines = sorted(df["cell_line"].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(cell_lines)))
    color_map = dict(zip(cell_lines, colors))

    # (a) Peak period per cell line (box plot)
    ax = axes[0, 0]
    data = [df.loc[df["cell_line"] == cl, "peak_period_px"].values for cl in cell_lines]
    bp = ax.boxplot(data, tick_labels=cell_lines, patch_artist=True)
    for patch, cl in zip(bp["boxes"], cell_lines):
        patch.set_facecolor(color_map[cl])
    ax.set_ylabel("Peak Period (pixels)")
    ax.set_title("Dominant Cell Size (FFT Peak Period)")
    ax.tick_params(axis="x", rotation=45)

    # (b) Mean period per cell line
    ax = axes[0, 1]
    data = [df.loc[df["cell_line"] == cl, "mean_period_px"].values for cl in cell_lines]
    bp = ax.boxplot(data, tick_labels=cell_lines, patch_artist=True)
    for patch, cl in zip(bp["boxes"], cell_lines):
        patch.set_facecolor(color_map[cl])
    ax.set_ylabel("Mean Period (pixels)")
    ax.set_title("Mean Cell Size (Spectral Centroid)")
    ax.tick_params(axis="x", rotation=45)

    # (c) FFT peak period vs annotation mean area
    ax = axes[1, 0]
    mask = (df["mean_area_px"] > 0) & (df["peak_period_px"] > 0)
    if mask.sum() > 0:
        for cl in cell_lines:
            m = mask & (df["cell_line"] == cl)
            if m.sum() > 0:
                ax.scatter(np.sqrt(df.loc[m, "mean_area_px"]), df.loc[m, "peak_period_px"],
                           label=cl, alpha=0.5, s=10, color=color_map[cl])
        ax.set_xlabel("sqrt(Mean Cell Area) [px]")
        ax.set_ylabel("FFT Peak Period [px]")
        ax.set_title("FFT Size Estimate vs Ground Truth")
        ax.legend(fontsize=8)

    # (d) Distribution of peak frequencies per cell line
    ax = axes[1, 1]
    for cl in cell_lines:
        subset = df.loc[df["cell_line"] == cl, "peak_freq_cyc_per_px"]
        ax.hist(subset, bins=30, alpha=0.4, label=cl, color=color_map[cl])
    ax.set_xlabel("Peak Frequency (cycles/px)")
    ax.set_ylabel("Count")
    ax.set_title("Peak Frequency Distribution")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "obj2_morphology.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out_path}")

    # Summary statistics
    print("\n  Mean peak period (pixels) by cell line:")
    for cl in cell_lines:
        subset = df.loc[df["cell_line"] == cl]
        print(f"    {cl:10s}: {subset['peak_period_px'].mean():.1f} ± {subset['peak_period_px'].std():.1f}")

    # Correlation with ground truth
    mask = (df["mean_area_px"] > 0) & (df["peak_period_px"] > 0)
    if mask.sum() > 0:
        r = np.corrcoef(np.sqrt(df.loc[mask, "mean_area_px"]), df.loc[mask, "peak_period_px"])[0, 1]
        print(f"\n  Correlation (sqrt(area) vs peak period): r = {r:+.3f}")

    print("  Done.")


if __name__ == "__main__":
    main()

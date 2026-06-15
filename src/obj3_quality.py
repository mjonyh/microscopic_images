#!/usr/bin/env python3
"""
Objective 3: Image Quality & Artifact Detection
Quantify phase-contrast artifacts (halo, shading) via FFT isotropy.
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
    load_image, list_images, compute_fft, radial_profile, azimuthal_profile,
    get_cell_line, parse_time, OUTPUT_DIR
)


def main():
    print("Objective 3: Image Quality & Artifact Detection")
    print("=" * 55)

    images = list_images()
    print(f"  Images: {len(images)}")

    records = []
    for i, path in enumerate(images):
        if i % 500 == 0:
            print(f"  Processing {i}/{len(images)}...")
        img = load_image(path)
        power, fx, fy = compute_fft(img)

        # Azimuthal isotropy: low variance = isotropic (clean), high = directional artifacts
        angles, az_profile = azimuthal_profile(power, n_bins=36)
        mean_az = az_profile.mean()
        isotropy = 1.0 - (az_profile.std() / (mean_az + 1e-10))

        # Low-frequency dominance (shading artifact)
        freqs, radial = radial_profile(power, n_bins=100)
        low_frac = radial[:10].sum() / (radial.sum() + 1e-10)

        # High-frequency noise floor
        high_frac = radial[-20:].sum() / (radial.sum() + 1e-10)

        records.append({
            "filename": path.stem,
            "cell_line": get_cell_line(path.stem),
            "time_h": parse_time(path.stem),
            "isotropy": isotropy,
            "low_freq_frac": low_frac,
            "high_freq_frac": high_frac,
            "azimuthal_std": az_profile.std(),
        })

    df = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "obj3_quality_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Quality scores saved: {csv_path}")

    # ── Plots ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Objective 3: Image Quality & Artifact Detection", fontsize=14, fontweight="bold")

    cell_lines = sorted(df["cell_line"].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(cell_lines)))
    color_map = dict(zip(cell_lines, colors))

    # (a) Isotropy distribution per cell line
    ax = axes[0, 0]
    data = [df.loc[df["cell_line"] == cl, "isotropy"].values for cl in cell_lines]
    bp = ax.boxplot(data, tick_labels=cell_lines, patch_artist=True)
    for patch, cl in zip(bp["boxes"], cell_lines):
        patch.set_facecolor(color_map[cl])
    ax.set_ylabel("Isotropy (1 = clean)")
    ax.set_title("FFT Isotropy by Cell Line")
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Threshold")
    ax.legend()

    # (b) Low-frequency fraction (shading)
    ax = axes[0, 1]
    data = [df.loc[df["cell_line"] == cl, "low_freq_frac"].values for cl in cell_lines]
    bp = ax.boxplot(data, tick_labels=cell_lines, patch_artist=True)
    for patch, cl in zip(bp["boxes"], cell_lines):
        patch.set_facecolor(color_map[cl])
    ax.set_ylabel("Low-Frequency Fraction")
    ax.set_title("Background Shading (Low-Freq Power)")
    ax.tick_params(axis="x", rotation=45)

    # (c) Isotropy vs time
    ax = axes[1, 0]
    for cl in cell_lines:
        subset = df[df["cell_line"] == cl]
        # Bin by time and compute mean isotropy
        bins = np.arange(0, subset["time_h"].max() + 12, 12)
        means = []
        centers = []
        for j in range(len(bins) - 1):
            m = (subset["time_h"] >= bins[j]) & (subset["time_h"] < bins[j + 1])
            if m.sum() > 0:
                means.append(subset.loc[m, "isotropy"].mean())
                centers.append((bins[j] + bins[j + 1]) / 2)
        ax.plot(centers, means, label=cl, color=color_map[cl], linewidth=1.5)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Mean Isotropy")
    ax.set_title("Image Quality Over Time")
    ax.legend(fontsize=8)

    # (d) Worst/best examples
    ax = axes[1, 1]
    ax.axis("off")
    worst = df.nsmallest(3, "isotropy")
    best = df.nlargest(3, "isotropy")
    text = "Worst quality (low isotropy):\n"
    for _, row in worst.iterrows():
        text += f"  {row['cell_line']:8s} iso={row['isotropy']:.3f}  {row['filename'][:30]}\n"
    text += "\nBest quality (high isotropy):\n"
    for _, row in best.iterrows():
        text += f"  {row['cell_line']:8s} iso={row['isotropy']:.3f}  {row['filename'][:30]}\n"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "obj3_quality.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out_path}")

    # Summary
    print("\n  Quality summary by cell line:")
    for cl in cell_lines:
        subset = df[df["cell_line"] == cl]
        print(f"    {cl:10s}: isotropy={subset['isotropy'].mean():.3f} ± {subset['isotropy'].std():.3f}, "
              f"low_freq={subset['low_freq_frac'].mean():.3f}")

    print("  Done.")


if __name__ == "__main__":
    main()

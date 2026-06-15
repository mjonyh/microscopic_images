#!/usr/bin/env python3
"""
Objective 1: Cell Density & Spatial Distribution
Relate FFT spatial frequency content to cell density.
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
    spectral_features, get_cell_line, load_annotations, OUTPUT_DIR
)


def main():
    print("Objective 1: Cell Density & Spatial Distribution")
    print("=" * 55)

    annotations = load_annotations()
    images = list_images()
    print(f"  Images: {len(images)}, Annotated: {len(annotations)}")

    # Process all images
    records = []
    for i, path in enumerate(images):
        if i % 500 == 0:
            print(f"  Processing {i}/{len(images)}...")
        img = load_image(path)
        power, fx, fy = compute_fft(img)
        freqs, profile = radial_profile(power)
        feats = spectral_features(power, fx, fy)
        cell_line = get_cell_line(path.stem)
        # Get annotation data
        ann = annotations.get(path.stem, {})
        feats["filename"] = path.stem
        feats["cell_line"] = cell_line
        feats["cell_count"] = ann.get("cell_count", -1)
        feats["mean_area"] = np.mean(ann["areas"]) if ann.get("areas") else -1
        records.append(feats)

    df = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "obj1_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Features saved: {csv_path}")

    # ── Plot 1: Radial power spectra per cell line ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Objective 1: Cell Density & FFT Power Spectrum", fontsize=14, fontweight="bold")

    # (a) Mean radial power spectrum per cell line
    ax = axes[0, 0]
    cell_lines = sorted(df["cell_line"].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(cell_lines)))
    for cl, color in zip(cell_lines, colors):
        mask = df["cell_line"] == cl
        # Re-compute mean profile for this cell line
        profiles = []
        for path in [p for p in images if get_cell_line(p.stem) == cl][:50]:  # sample 50
            img = load_image(path)
            power, _, _ = compute_fft(img)
            f, p = radial_profile(power)
            profiles.append(p)
        mean_prof = np.mean(profiles, axis=0)
        ax.plot(f, mean_prof, label=cl, color=color, linewidth=1.5)
    ax.set_xlabel("Spatial Frequency (cycles/pixel)")
    ax.set_ylabel("Normalized Power")
    ax.set_title("Mean Radial Power Spectrum by Cell Line")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    # (b) Spectral centroid vs cell count
    ax = axes[0, 1]
    mask = df["cell_count"] > 0
    if mask.sum() > 0:
        scatter = ax.scatter(df.loc[mask, "cell_count"], df.loc[mask, "centroid"],
                             c=df.loc[mask, "cell_count"], cmap="viridis", alpha=0.5, s=10)
        plt.colorbar(scatter, ax=ax, label="Cell count")
        # Fit line
        z = np.polyfit(df.loc[mask, "cell_count"], df.loc[mask, "centroid"], 1)
        p_line = np.poly1d(z)
        x_range = np.linspace(df.loc[mask, "cell_count"].min(), df.loc[mask, "cell_count"].max(), 100)
        ax.plot(x_range, p_line(x_range), "r--", label=f"Linear fit (slope={z[0]:.4f})")
        ax.legend()
    ax.set_xlabel("Cell Count")
    ax.set_ylabel("Spectral Centroid")
    ax.set_title("Spectral Centroid vs Cell Density")

    # (c) Total power vs cell count
    ax = axes[1, 0]
    if mask.sum() > 0:
        for cl in cell_lines:
            m = mask & (df["cell_line"] == cl)
            if m.sum() > 0:
                ax.scatter(df.loc[m, "cell_count"], df.loc[m, "total_power"],
                           label=cl, alpha=0.5, s=10)
    ax.set_xlabel("Cell Count")
    ax.set_ylabel("Total Power (log10)")
    ax.set_title("Total FFT Power vs Cell Density")
    ax.legend(fontsize=8)

    # (d) Bandwidth vs cell count
    ax = axes[1, 1]
    if mask.sum() > 0:
        ax.scatter(df.loc[mask, "cell_count"], df.loc[mask, "bandwidth"],
                   alpha=0.3, s=10, c="#89b4fa")
        z = np.polyfit(df.loc[mask, "cell_count"], df.loc[mask, "bandwidth"], 1)
        p_line = np.poly1d(z)
        x_range = np.linspace(df.loc[mask, "cell_count"].min(), df.loc[mask, "cell_count"].max(), 100)
        ax.plot(x_range, p_line(x_range), "r--", label=f"Fit (slope={z[0]:.4f})")
        ax.legend()
    ax.set_xlabel("Cell Count")
    ax.set_ylabel("Spectral Bandwidth")
    ax.set_title("Bandwidth vs Cell Density")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "obj1_density_spectrum.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out_path}")

    # Print correlation summary
    if mask.sum() > 0:
        print("\n  Correlations with cell count:")
        for feat in ["centroid", "bandwidth", "total_power", "low_power", "mid_power", "high_power"]:
            r = np.corrcoef(df.loc[mask, "cell_count"], df.loc[mask, feat])[0, 1]
            print(f"    {feat:15s}: r = {r:+.3f}")

    print("  Done.")


if __name__ == "__main__":
    main()

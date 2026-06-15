#!/usr/bin/env python3
"""
Objective 6: Time-Lapse Dynamics
Track FFT spectral changes over time for biological insights.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    load_image, list_images, compute_fft, radial_profile,
    spectral_features, get_cell_line, parse_time, get_well_id, OUTPUT_DIR
)


def main():
    print("Objective 6: Time-Lapse Dynamics")
    print("=" * 55)

    images = list_images()
    print(f"  Images: {len(images)}")

    # Group by well
    wells = {}
    for path in images:
        wid = get_well_id(path.stem)
        if wid not in wells:
            wells[wid] = []
        wells[wid].append(path)

    print(f"  Wells: {len(wells)}")

    # Process each well's time series
    all_records = []
    for wid, paths in wells.items():
        # Sort by time
        paths.sort(key=lambda p: parse_time(p.stem))
        cell_line = get_cell_line(paths[0].stem)

        prev_centroid = None
        for path in paths:
            img = load_image(path)
            power, fx, fy = compute_fft(img)
            feats = spectral_features(power, fx, fy)
            t = parse_time(path.stem)

            # Rate of change
            centroid_rate = 0
            if prev_centroid is not None:
                centroid_rate = feats["centroid"] - prev_centroid
            prev_centroid = feats["centroid"]

            all_records.append({
                "well_id": wid,
                "cell_line": cell_line,
                "filename": path.stem,
                "time_h": t,
                "centroid": feats["centroid"],
                "bandwidth": feats["bandwidth"],
                "total_power": feats["total_power"],
                "low_power": feats["low_power"],
                "high_power": feats["high_power"],
                "centroid_rate": centroid_rate,
            })

    df = pd.DataFrame(all_records)
    csv_path = OUTPUT_DIR / "obj6_timelapse.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Time-series data saved: {csv_path}")

    # ── Plots ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Objective 6: Time-Lapse FFT Dynamics", fontsize=14, fontweight="bold")

    cell_lines = sorted(df["cell_line"].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(cell_lines)))
    color_map = dict(zip(cell_lines, colors))

    # (a) Spectral centroid over time (one line per cell line = mean of wells)
    ax = axes[0, 0]
    for cl in cell_lines:
        subset = df[df["cell_line"] == cl]
        # Bin by time
        bins = np.arange(0, subset["time_h"].max() + 6, 6)
        means = []
        stds = []
        centers = []
        for j in range(len(bins) - 1):
            m = (subset["time_h"] >= bins[j]) & (subset["time_h"] < bins[j + 1])
            if m.sum() > 0:
                means.append(subset.loc[m, "centroid"].mean())
                stds.append(subset.loc[m, "centroid"].std())
                centers.append((bins[j] + bins[j + 1]) / 2)
        means = np.array(means)
        stds = np.array(stds)
        ax.plot(centers, means, label=cl, color=color_map[cl], linewidth=1.5)
        ax.fill_between(centers, means - stds, means + stds, alpha=0.15, color=color_map[cl])
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Spectral Centroid")
    ax.set_title("Spectral Centroid Over Time")
    ax.legend(fontsize=8)

    # (b) Total power over time (confluence proxy)
    ax = axes[0, 1]
    for cl in cell_lines:
        subset = df[df["cell_line"] == cl]
        bins = np.arange(0, subset["time_h"].max() + 6, 6)
        means = []
        centers = []
        for j in range(len(bins) - 1):
            m = (subset["time_h"] >= bins[j]) & (subset["time_h"] < bins[j + 1])
            if m.sum() > 0:
                means.append(subset.loc[m, "total_power"].mean())
                centers.append((bins[j] + bins[j + 1]) / 2)
        ax.plot(centers, means, label=cl, color=color_map[cl], linewidth=1.5)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Total Power (log10)")
    ax.set_title("Total FFT Power (Confluence Proxy)")
    ax.legend(fontsize=8)

    # (c) High-frequency power (mitosis proxy)
    ax = axes[1, 0]
    for cl in cell_lines:
        subset = df[df["cell_line"] == cl]
        bins = np.arange(0, subset["time_h"].max() + 6, 6)
        means = []
        centers = []
        for j in range(len(bins) - 1):
            m = (subset["time_h"] >= bins[j]) & (subset["time_h"] < bins[j + 1])
            if m.sum() > 0:
                means.append(subset.loc[m, "high_power"].mean())
                centers.append((bins[j] + bins[j + 1]) / 2)
        ax.plot(centers, means, label=cl, color=color_map[cl], linewidth=1.5)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("High-Frequency Power Fraction")
    ax.set_title("High-Freq Power (Fine Structure / Mitosis)")
    ax.legend(fontsize=8)

    # (d) Bandwidth over time (heterogeneity)
    ax = axes[1, 1]
    for cl in cell_lines:
        subset = df[df["cell_line"] == cl]
        bins = np.arange(0, subset["time_h"].max() + 6, 6)
        means = []
        centers = []
        for j in range(len(bins) - 1):
            m = (subset["time_h"] >= bins[j]) & (subset["time_h"] < bins[j + 1])
            if m.sum() > 0:
                means.append(subset.loc[m, "bandwidth"].mean())
                centers.append((bins[j] + bins[j + 1]) / 2)
        ax.plot(centers, means, label=cl, color=color_map[cl], linewidth=1.5)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Spectral Bandwidth")
    ax.set_title("Spectral Bandwidth (Size Heterogeneity)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "obj6_timelapse.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out_path}")

    # Detect mitosis-like events (spikes in high-freq power)
    print("\n  Mitosis-like event detection (high-freq power spikes):")
    mitosis_events = []
    for wid in df["well_id"].unique():
        subset = df[df["well_id"] == wid].sort_values("time_h")
        if len(subset) < 10:
            continue
        high_vals = subset["high_power"].values
        peaks, props = find_peaks(high_vals, height=np.mean(high_vals) + 2 * np.std(high_vals),
                                   distance=5)
        for p in peaks:
            mitosis_events.append({
                "well_id": wid,
                "cell_line": subset.iloc[p]["cell_line"],
                "time_h": subset.iloc[p]["time_h"],
                "high_power": high_vals[p],
            })

    mitosis_df = pd.DataFrame(mitosis_events)
    if len(mitosis_df) > 0:
        mitosis_csv = OUTPUT_DIR / "obj6_mitosis_events.csv"
        mitosis_df.to_csv(mitosis_csv, index=False)
        print(f"  Detected {len(mitosis_df)} mitosis-like events")
        print(f"  Events saved: {mitosis_csv}")
        for cl in cell_lines:
            n = (mitosis_df["cell_line"] == cl).sum()
            print(f"    {cl:10s}: {n} events")
    else:
        print("  No significant mitosis-like events detected")

    print("  Done.")


if __name__ == "__main__":
    main()

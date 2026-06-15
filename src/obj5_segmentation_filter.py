#!/usr/bin/env python3
"""
Objective 5: FFT-Based Segmentation Preprocessing
Use frequency-domain filtering to improve cell segmentation.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    load_image, list_images, compute_fft, bandpass_filter,
    get_cell_line, load_annotations, OUTPUT_DIR
)


def simple_segment(image: np.ndarray) -> np.ndarray:
    """Simple Otsu threshold segmentation."""
    thresh = threshold_otsu(image)
    binary = image > thresh
    # Clean up small objects
    labeled = label(binary)
    props = regionprops(labeled)
    for p in props:
        if p.area < 50:
            labeled[labeled == p.label] = 0
    return (labeled > 0).astype(np.uint8)


def compute_iou(pred: np.ndarray, gt_masks: list) -> float:
    """Compute IoU between predicted binary mask and ground truth."""
    if not gt_masks:
        return 0.0
    # Build GT binary mask from COCO segmentations
    # For simplicity, use bounding boxes as proxy
    gt_binary = np.zeros_like(pred, dtype=bool)
    for bbox in gt_masks:
        x, y, w, h = [int(v) for v in bbox]
        gt_binary[y:y + h, x:x + w] = True
    intersection = np.logical_and(pred, gt_binary).sum()
    union = np.logical_or(pred, gt_binary).sum()
    return intersection / union if union > 0 else 0.0


def main():
    print("Objective 5: FFT-Based Segmentation Preprocessing")
    print("=" * 55)

    annotations = load_annotations()
    # Only process annotated images
    annotated_stems = set(annotations.keys())
    images = [p for p in list_images() if p.stem in annotated_stems]
    print(f"  Annotated images: {len(images)}")

    # Test different filter parameters
    filter_params = [
        (0.005, 0.15),
        (0.01, 0.2),
        (0.01, 0.3),
        (0.02, 0.25),
        (0.005, 0.4),
    ]

    records = []
    for i, path in enumerate(images):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(images)}...")
        img = load_image(path)
        ann = annotations.get(path.stem, {})
        bboxes = ann.get("bboxes", [])

        # Segment raw image
        seg_raw = simple_segment(img)
        iou_raw = compute_iou(seg_raw, bboxes)

        # Segment with each filter setting
        best_iou = iou_raw
        best_params = None
        for low, high in filter_params:
            filtered = bandpass_filter(img, low_cut=low, high_cut=high)
            seg_filt = simple_segment(filtered)
            iou_filt = compute_iou(seg_filt, bboxes)
            if iou_filt > best_iou:
                best_iou = iou_filt
                best_params = (low, high)

        records.append({
            "filename": path.stem,
            "cell_line": get_cell_line(path.stem),
            "cell_count": ann.get("cell_count", 0),
            "iou_raw": iou_raw,
            "iou_best_filtered": best_iou,
            "iou_improvement": best_iou - iou_raw,
            "best_low_cut": best_params[0] if best_params else -1,
            "best_high_cut": best_params[1] if best_params else -1,
        })

    df = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "obj5_segmentation.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Results saved: {csv_path}")

    # ── Plots ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Objective 5: FFT Filtering for Segmentation", fontsize=14, fontweight="bold")

    cell_lines = sorted(df["cell_line"].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(cell_lines)))
    color_map = dict(zip(cell_lines, colors))

    # (a) IoU raw vs filtered
    ax = axes[0, 0]
    ax.scatter(df["iou_raw"], df["iou_best_filtered"], alpha=0.4, s=15, c="#89b4fa")
    lim = [0, max(df["iou_raw"].max(), df["iou_best_filtered"].max()) + 0.05]
    ax.plot(lim, lim, "r--", label="No improvement")
    ax.set_xlabel("IoU (Raw)")
    ax.set_ylabel("IoU (Best Filtered)")
    ax.set_title("Segmentation Quality: Raw vs FFT-Filtered")
    ax.legend()
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    # (b) Improvement distribution per cell line
    ax = axes[0, 1]
    data = [df.loc[df["cell_line"] == cl, "iou_improvement"].values for cl in cell_lines]
    bp = ax.boxplot(data, labels=cell_lines, patch_artist=True)
    for patch, cl in zip(bp["boxes"], cell_lines):
        patch.set_facecolor(color_map[cl])
    ax.set_ylabel("IoU Improvement")
    ax.set_title("IoU Improvement by Cell Line")
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)

    # (c) Best filter parameters distribution
    ax = axes[1, 0]
    mask = df["best_low_cut"] > 0
    if mask.sum() > 0:
        ax.scatter(df.loc[mask, "best_low_cut"], df.loc[mask, "best_high_cut"],
                   c=df.loc[mask, "iou_improvement"], cmap="RdYlGn", s=20)
        ax.set_xlabel("Low Cutoff")
        ax.set_ylabel("High Cutoff")
        ax.set_title("Best Filter Parameters (color = improvement)")

    # (d) Summary table
    ax = axes[1, 1]
    ax.axis("off")
    summary = df.groupby("cell_line").agg(
        mean_iou_raw=("iou_raw", "mean"),
        mean_iou_filt=("iou_best_filtered", "mean"),
        mean_improve=("iou_improvement", "mean"),
    ).round(4)
    table_text = "Mean IoU by Cell Line:\n\n"
    table_text += f"{'Cell Line':10s}  {'Raw':>8s}  {'Filtered':>8s}  {'Improve':>8s}\n"
    table_text += "-" * 40 + "\n"
    for cl, row in summary.iterrows():
        table_text += f"{cl:10s}  {row['mean_iou_raw']:8.4f}  {row['mean_iou_filt']:8.4f}  {row['mean_improve']:+.4f}\n"
    ax.text(0.05, 0.95, table_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "obj5_segmentation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out_path}")

    # Summary
    print(f"\n  Overall mean IoU: raw={df['iou_raw'].mean():.4f}, filtered={df['iou_best_filtered'].mean():.4f}")
    print(f"  Mean improvement: {df['iou_improvement'].mean():+.4f}")
    improved = (df["iou_improvement"] > 0).sum()
    print(f"  Images improved: {improved}/{len(df)} ({improved / len(df) * 100:.1f}%)")

    print("  Done.")


if __name__ == "__main__":
    main()

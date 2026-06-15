#!/usr/bin/env python3
"""
LIVECell Dataset Summary
========================
Explores and summarizes the LIVECell phase-contrast microscopy dataset.
"""

import json
import os
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def find_annotations(data_dir: Path) -> list[Path]:
    """Find all JSON annotation files."""
    return list(data_dir.rglob("*.json"))


def find_images(data_dir: Path) -> dict[str, list[Path]]:
    """Find all image files grouped by extension."""
    exts = ["*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg"]
    images = {}
    for ext in exts:
        files = list(data_dir.rglob(ext))
        if files:
            images[ext] = files
    return images


def load_coco_annotations(json_path: Path) -> dict:
    """Load COCO-format annotation file."""
    with open(json_path) as f:
        return json.load(f)


def summarize_dataset(data_dir: Path):
    """Main summary function."""
    print("=" * 60)
    print("LIVECell Dataset Summary")
    print("=" * 60)

    # 1. Directory structure
    print("\n--- Directory Structure ---")
    for item in sorted(data_dir.rglob("*")):
        depth = len(item.relative_to(data_dir).parts)
        if depth <= 2:
            if item.is_dir():
                n_files = sum(1 for _ in item.rglob("*") if _.is_file())
                print(f"  {'  ' * depth}{item.name}/ ({n_files} files)")

    # 2. Find images
    print("\n--- Image Files ---")
    images = find_images(data_dir)
    total_images = 0
    for ext, files in images.items():
        print(f"  {ext}: {len(files)} files")
        total_images += len(files)
    print(f"  Total: {total_images} images")

    # 3. Find annotations
    print("\n--- Annotations ---")
    ann_files = find_annotations(data_dir)
    print(f"  Found {len(ann_files)} JSON annotation files:")
    for af in ann_files:
        print(f"    {af.relative_to(data_dir)}")

    # 4. Parse all subsets and report
    print("\n--- Subset Comparison ---")
    subsets = []
    for ann_file in ann_files:
        coco = load_coco_annotations(ann_file)
        n_imgs = len(coco.get("images", []))
        n_anns = len(coco.get("annotations", []))
        subsets.append((ann_file.name, n_imgs, n_anns))
        print(f"  {ann_file.name}: {n_imgs} images, {n_anns} cell annotations")

    # Use largest subset for detailed summary
    best_file = max(ann_files, key=lambda f: len(load_coco_annotations(f).get("annotations", [])))
    coco = load_coco_annotations(best_file)
    print(f"\n--- Detailed Summary ({best_file.name}) ---")

    # Basic info
    if "info" in coco:
        info = coco["info"]
        print(f"  Description: {info.get('description', 'N/A')}")
        print(f"  Version: {info.get('version', 'N/A')}")

    # Categories
    categories = coco.get("categories", [])
    print(f"\n  Categories ({len(categories)}):")
    for cat in categories:
        print(f"    id={cat['id']}: {cat['name']}")

    # Images
    imgs = coco.get("images", [])
    print(f"\n  Images: {len(imgs)}")

    # Annotations
    anns = coco.get("annotations", [])
    print(f"  Annotations (cells): {len(anns)}")

    if imgs:
        widths = [img["width"] for img in imgs]
        heights = [img["height"] for img in imgs]
        print(f"\n  Image dimensions:")
        print(f"    Width:  {min(widths)}-{max(widths)} (mean {np.mean(widths):.0f})")
        print(f"    Height: {min(heights)}-{max(heights)} (mean {np.mean(heights):.0f})")
        size_counter = Counter((w, h) for w, h in zip(widths, heights))
        print(f"    Unique sizes: {len(size_counter)}")
        for (w, h), count in size_counter.most_common(5):
            print(f"      {w}x{h}: {count} images")

    if anns:
        img_id_to_name = {img["id"]: img.get("file_name", str(img["id"])) for img in imgs}
        cells_per_image = Counter(ann["image_id"] for ann in anns)
        counts = list(cells_per_image.values())

        print(f"\n  Cells per image:")
        print(f"    Min:    {min(counts)}")
        print(f"    Max:    {max(counts)}")
        print(f"    Mean:   {np.mean(counts):.1f}")
        print(f"    Median: {np.median(counts):.1f}")

        areas = []
        for ann in anns:
            if "area" in ann:
                areas.append(ann["area"])
            elif "segmentation" in ann:
                bbox = ann.get("bbox", [])
                if len(bbox) == 4:
                    areas.append(bbox[2] * bbox[3])

        if areas:
            areas = np.array(areas)
            print(f"\n  Cell area (pixels):")
            print(f"    Min:    {np.min(areas):.0f}")
            print(f"    Max:    {np.max(areas):.0f}")
            print(f"    Mean:   {np.mean(areas):.0f}")
            print(f"    Median: {np.median(areas):.0f}")

        if categories:
            cat_counts = Counter(ann.get("category_id", 0) for ann in anns)
            cat_name_map = {c["id"]: c["name"] for c in categories}
            print(f"\n  Cell type distribution:")
            for cat_id, count in sorted(cat_counts.items()):
                name = cat_name_map.get(cat_id, f"id_{cat_id}")
                pct = count / len(anns) * 100
                print(f"    {name}: {count} ({pct:.1f}%)")

        # --- Generate plots ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("LIVECell Dataset Overview", fontsize=14, fontweight="bold")

        axes[0, 0].hist(counts, bins=50, color="#89b4fa", edgecolor="white")
        axes[0, 0].set_title("Cells per Image")
        axes[0, 0].set_xlabel("Number of cells")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].axvline(np.mean(counts), color="#f38ba8", linestyle="--",
                           label=f"Mean: {np.mean(counts):.1f}")
        axes[0, 0].legend()

        if areas.size > 0:
            axes[0, 1].hist(np.log10(areas + 1), bins=50, color="#a6e3a1", edgecolor="white")
            axes[0, 1].set_title("Cell Area Distribution (log10)")
            axes[0, 1].set_xlabel("log10(area + 1)")
            axes[0, 1].set_ylabel("Frequency")

        if len(size_counter) > 1:
            sizes = [f"{w}x{h}" for (w, h), _ in size_counter.most_common(10)]
            counts_sz = [count for _, count in size_counter.most_common(10)]
            axes[1, 0].barh(sizes, counts_sz, color="#fab387")
            axes[1, 0].set_title("Image Size Distribution")
            axes[1, 0].set_xlabel("Count")
        else:
            axes[1, 0].text(0.5, 0.5, f"All images:\n{widths[0]}x{heights[0]}",
                            ha="center", va="center", fontsize=16,
                            transform=axes[1, 0].transAxes)
            axes[1, 0].set_title("Image Size")

        if categories:
            cat_names = [cat_name_map.get(cid, f"id_{cid}") for cid in sorted(cat_counts.keys())]
            cat_values = [cat_counts[cid] for cid in sorted(cat_counts.keys())]
            colors = ["#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8", "#cba6f7", "#fab387"]
            axes[1, 1].pie(cat_values, labels=cat_names, autopct="%1.1f%%",
                           colors=colors[:len(cat_names)])
            axes[1, 1].set_title("Cell Type Distribution")

        plt.tight_layout()
        out_path = OUTPUT_DIR / "livecell_summary.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Summary plot saved: {out_path}")

        # Save CSV
        df_data = [{"image": img_id_to_name.get(iid, str(iid)), "n_cells": n}
                   for iid, n in cells_per_image.items()]
        df = pd.DataFrame(df_data)
        csv_path = OUTPUT_DIR / "livecell_image_cell_counts.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Cell counts CSV saved: {csv_path}")

    # 5. Sample images
    print("\n--- Sample Images ---")
    for ext, files in images.items():
        sample = files[0]
        img = Image.open(sample)
        print(f"  {sample.name}: {img.size}, mode={img.mode}")
        if len(files) > 1:
            img2 = Image.open(files[-1])
            print(f"  {files[-1].name}: {img2.size}, mode={img2.mode}")

    # 6. Cell line names from filenames
    print("\n--- Cell Lines (from filenames) ---")
    tif_files = images.get("*.tif", [])
    cell_lines = Counter()
    for f in tif_files:
        # Format: CellLine_Phase_...
        parts = f.stem.split("_")
        if parts:
            cell_lines[parts[0]] += 1
    for line, count in cell_lines.most_common():
        print(f"  {line}: {count} images")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    summarize_dataset(DATA_DIR)

#!/usr/bin/env python3
"""
Synthetic quality degradation pipeline.
Applies controlled degradations to LIVECell images to create
paired high/low quality versions with known ground truth.
"""
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import json
import csv

sys.path.insert(0, str(Path(__file__).parent))
from common import load_image, list_images, load_annotations, OUTPUT_DIR

DATA_DIR = Path(__file__).parent.parent / "data"
MQ_DIR = DATA_DIR / "mixed_quality"
ANNOTATIONS = load_annotations()

# ── Degradation functions ──────────────────────────────────

def add_gaussian_noise(image, sigma):
    """Add Gaussian noise with given sigma."""
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_motion_blur(image, kernel_size):
    """Add uniform motion blur."""
    from scipy.ndimage import convolve
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
    return convolve(image.astype(np.float64), kernel).astype(np.uint8)

def add_defocus_blur(image, sigma):
    """Add Gaussian defocus blur (out-of-focus)."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(image.astype(np.float64), sigma=sigma).astype(np.uint8)

def add_shading(image, alpha):
    """Add quadratic illumination shading (vignetting)."""
    h, w = image.shape
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    X, Y = np.meshgrid(x, y)
    shading = 1.0 + alpha * (X**2 + Y**2)
    shaded = image.astype(np.float64) * shading
    return np.clip(shaded, 0, 255).astype(np.uint8)

def add_jpeg_compression(image, quality):
    """Add JPEG compression artifacts."""
    from io import BytesIO
    img_uint8 = image.astype(np.uint8) if image.dtype != np.uint8 else image
    img_pil = Image.fromarray(img_uint8)
    buffer = BytesIO()
    img_pil.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer))

def add_combined(image, noise_sigma, blur_sigma, shading_alpha):
    """Combined degradation: noise + defocus + shading."""
    from scipy.ndimage import gaussian_filter
    noisy = image.astype(np.float64) + np.random.normal(0, noise_sigma, image.shape)
    blurred = gaussian_filter(noisy, sigma=blur_sigma)
    h, w = image.shape
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    X, Y = np.meshgrid(x, y)
    shading = 1.0 + shading_alpha * (X**2 + Y**2)
    result = blurred * shading
    return np.clip(result, 0, 255).astype(np.uint8)

# ── Configuration ──────────────────────────────────────────

DEGRADATIONS = {
    # Gaussian noise
    "noise_25":  {"fn": add_gaussian_noise, "params": {"sigma": 25}},
    "noise_50":  {"fn": add_gaussian_noise, "params": {"sigma": 50}},
    "noise_100": {"fn": add_gaussian_noise, "params": {"sigma": 100}},
    # Motion blur
    "motion_blur_5":  {"fn": add_motion_blur, "params": {"kernel_size": 5}},
    "motion_blur_11": {"fn": add_motion_blur, "params": {"kernel_size": 11}},
    # Defocus blur
    "defocus_2": {"fn": add_defocus_blur, "params": {"sigma": 2}},
    "defocus_8": {"fn": add_defocus_blur, "params": {"sigma": 8}},
    # Shading
    "shading_0.3": {"fn": add_shading, "params": {"alpha": 0.3}},
    "shading_0.7": {"fn": add_shading, "params": {"alpha": 0.7}},
    # JPEG compression
    "jpeg_70": {"fn": add_jpeg_compression, "params": {"quality": 70}},
    "jpeg_30": {"fn": add_jpeg_compression, "params": {"quality": 30}},
    # Combined
    "combined_mild":   {"fn": add_combined, "params": {"noise_sigma": 15, "blur_sigma": 1, "shading_alpha": 0.2}},
    "combined_severe": {"fn": add_combined, "params": {"noise_sigma": 50, "blur_sigma": 4, "shading_alpha": 0.5}},
}

# ── Main pipeline ──────────────────────────────────────────

print("=" * 60)
print("Synthetic Quality Degradation Pipeline")
print("=" * 60)

# Select images: all annotated + some unannotated for each cell line
cell_lines = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SKOV3", "SkBr3"]
selected_images = []

for cl in cell_lines:
    cl_all = list_images(cl)
    cl_annotated = [p for p in cl_all if p.stem in ANNOTATIONS]
    # Take all annotated + up to 50 unannotated per line
    cl_unannotated = [p for p in cl_all if p.stem not in ANNOTATIONS][:50]
    selected_images.extend(cl_annotated)
    selected_images.extend(cl_unannotated)

print(f"  Selected images: {len(selected_images)}")
print(f"  Degradation types: {len(DEGRADATIONS)}")
print(f"  Total output images: {len(selected_images) * (1 + len(DEGRADATIONS))}")

# Create directories
MQ_DIR.mkdir(parents=True, exist_ok=True)
(MQ_DIR / "high_quality" / "images").mkdir(parents=True, exist_ok=True)
for deg_name in DEGRADATIONS:
    (MQ_DIR / "synthetic_low" / deg_name).mkdir(parents=True, exist_ok=True)

# Process images
quality_labels = []
total = len(selected_images)

for i, path in enumerate(selected_images):
    if i % 100 == 0:
        print(f"  Processing {i+1}/{total}...")

    img = load_image(path)
    img_uint8 = img.astype(np.uint8)
    stem = path.stem
    cell_line = stem.split("_")[0]
    is_annotated = stem in ANNOTATIONS

    # Save high quality
    hq_path = MQ_DIR / "high_quality" / "images" / f"{stem}.tif"
    Image.fromarray(img_uint8).save(hq_path)

    quality_labels.append({
        "filename": stem, "cell_line": cell_line,
        "quality": "high", "degradation": "none",
        "annotated": is_annotated,
        "psnr": float("inf"), "original_path": str(path)
    })

    # Apply each degradation
    for deg_name, deg_config in DEGRADATIONS.items():
        degraded = deg_config["fn"](img_uint8, **deg_config["params"])
        out_path = MQ_DIR / "synthetic_low" / deg_name / f"{stem}.tif"
        Image.fromarray(degraded).save(out_path)

        # Compute PSNR
        mse = np.mean((img.astype(float) - degraded.astype(float))**2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")

        quality_labels.append({
            "filename": stem, "cell_line": cell_line,
            "quality": "low", "degradation": deg_name,
            "annotated": is_annotated,
            "psnr": round(psnr, 2),
            "original_path": str(path)
        })

# Save quality labels
labels_path = MQ_DIR / "quality_labels.csv"
with open(labels_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=quality_labels[0].keys())
    writer.writeheader()
    writer.writerows(quality_labels)

print(f"\n  Quality labels saved: {labels_path}")
print(f"  Total images generated: {len(quality_labels)}")

# Summary
hq_count = sum(1 for q in quality_labels if q["quality"] == "high")
lq_count = sum(1 for q in quality_labels if q["quality"] == "low")
ann_count = sum(1 for q in quality_labels if q["annotated"])

print(f"\n  Summary:")
print(f"    High quality:  {hq_count}")
print(f"    Low quality:   {lq_count}")
print(f"    Annotated:     {ann_count}")
print(f"    Degradation types: {len(DEGRADATIONS)}")

# Per-degradation PSNR summary
print(f"\n  PSNR by degradation type:")
for deg_name in DEGRADATIONS:
    psnrs = [q["psnr"] for q in quality_labels if q["degradation"] == deg_name]
    if psnrs:
        print(f"    {deg_name:20s}: mean PSNR = {np.mean(psnrs):.1f} dB")

print("\nDone.")

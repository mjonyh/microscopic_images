#!/usr/bin/env python3
"""
Generate visual comparison figure: original images + enhancement results
for each method, on both HQ and LQ images.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from filters import apply_filter
from common import load_image, list_images, load_annotations, OUTPUT_DIR

# Import our physics-informed models
sys.path.insert(0, str(Path(__file__).parent))
from phaseA_physics_models import DeBCRInspired, PIDDPMInspired, PSFLearningPhysics

# ── Select representative images ───────────────────────────
annotations = load_annotations()

# Select one good image per degradation type from different cell lines
test_cases = {
    "MCF7_noise": list_images("MCF7")[len(list_images("MCF7"))//2],
    "SHSY5Y_combined": list_images("SHSY5Y")[len(list_images("SHSY5Y"))//2],
    "BV2_noise": list_images("BV2")[len(list_images("BV2"))//2],
    "SkBr3_combined": list_images("SkBr3")[len(list_images("SkBr3"))//2],
}

# Initialize models
model_debcr = DeBCRInspired(wavelet='db4', levels=3, lambda_physics=0.1)
model_piddpm = PIDDPMInspired(n_steps=30, lr=0.02, lambda_physics=0.3)
model_psf = PSFLearningPhysics(zernike_order=4)

# ── Process each test case ─────────────────────────────────
print("Generating visual comparison...")

for case_name, path in test_cases.items():
    if not path.exists():
        print(f"  Skipping {case_name}: file not found")
        continue

    img_hq = load_image(path).astype(np.uint8)
    cell_line = case_name.split("_")[0]
    deg_type = case_name.split("_")[1]

    # Load degraded version
    deg_path = Path("data/mixed_quality") / "synthetic_low" / deg_type / f"{path.stem}.tif"
    if not deg_path.exists():
        print(f"  Skipping {case_name}: degraded file not found")
        continue

    img_lq = np.array(Image.open(deg_path))

    print(f"  Processing {case_name}...")

    # Apply all methods
    results = {
        "HQ (Original)": img_hq,
        "LQ (Degraded)": img_lq,
    }

    # DeBCR
    try:
        results["DeBCR"] = model_debcr.enhance(img_lq)
    except:
        results["DeBCR"] = img_lq

    # PI-DDPM
    try:
        results["PI-DDPM"] = model_piddpm.enhance(img_lq)
    except:
        results["PI-DDPM"] = img_lq

    # PSF-Learning
    try:
        results["PSF-Learning"] = model_psf.enhance(img_lq)
    except:
        results["PSF-Learning"] = img_lq

    # DoG Filter
    try:
        results["DoG Filter"] = apply_filter(img_lq, "dog", sigma1=0.05, sigma2=0.20)
    except:
        results["DoG Filter"] = img_lq

    # DeBCR + DoG
    try:
        results["DeBCR+DoG"] = apply_filter(results["DeBCR"], "dog", sigma1=0.05, sigma2=0.20)
    except:
        results["DeBCR+DoG"] = img_lq

    # ── Create figure ───────────────────────────────────────
    n_methods = len(results)
    fig, axes = plt.subplots(2, n_methods, figsize=(3.5 * n_methods, 7))
    fig.suptitle(f"Visual Comparison: {cell_line} — {deg_type} degradation",
                 fontsize=13, fontweight="bold", y=0.98)

    vmin, vmax = 30, 220  # Consistent intensity range

    for idx, (method_name, img) in enumerate(results.items()):
        # Top row: full image
        ax_top = axes[0, idx]
        ax_top.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax_top.set_title(method_name, fontsize=9, fontweight="bold")
        ax_top.axis("off")

        # Bottom row: zoomed region (center 200×200)
        h, w = img.shape
        cy, cx = h // 2, w // 2
        zoom = img[cy-100:cy+100, cx-100:cx+100]
        ax_bot = axes[1, idx]
        ax_bot.imshow(zoom, cmap="gray", vmin=vmin, vmax=vmax)
        ax_bot.set_title("Zoom (center)", fontsize=7)
        ax_bot.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUTPUT_DIR / f"visual_comparison_{case_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")

# ── Create summary comparison figure ───────────────────────
print("\nGenerating summary comparison figure...")

# Use one representative image
rep_path = list_images("MCF7")[len(list_images("MCF7"))//2]
img_hq = load_image(rep_path).astype(np.uint8)
img_lq = np.array(Image.open(Path("data/mixed_quality") / "synthetic_low" / "combined_mild" / f"{rep_path.stem}.tif"))

# Apply all methods
results = {
    "HQ\n(Original)": img_hq,
    "LQ\n(Degraded)": img_lq,
    "DeBCR": model_debcr.enhance(img_lq),
    "PI-DDPM": model_piddpm.enhance(img_lq),
    "PSF-Learning": model_psf.enhance(img_lq),
    "DoG\nFilter": apply_filter(img_lq, "dog", sigma1=0.05, sigma2=0.20),
    "DeBCR\n+DoG": apply_filter(model_debcr.enhance(img_lq), "dog", sigma1=0.05, sigma2=0.20),
}

fig, axes = plt.subplots(2, len(results), figsize=(3.2 * len(results), 7))
fig.suptitle("Visual Comparison of Enhancement Methods — MCF7 Phase-Contrast (Combined Degradation)",
             fontsize=13, fontweight="bold", y=0.98)

vmin, vmax = 30, 220

for idx, (method_name, img) in enumerate(results.items()):
    # Full image
    ax_top = axes[0, idx]
    ax_top.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax_top.set_title(method_name, fontsize=9, fontweight="bold",
                     color="#f38ba8" if "LQ" in method_name else
                     "#a6e3a1" if "HQ" in method_name else "#89b4fa")
    ax_top.axis("off")

    # Zoomed region
    h, w = img.shape
    cy, cx = h // 2, w // 2
    zoom = img[cy-80:cy+80, cx-80:cx+80]
    ax_bot = axes[1, idx]
    ax_bot.imshow(zoom, cmap="gray", vmin=vmin, vmax=vmax)
    ax_bot.axis("off")

    # Add IoU text if we have annotations
    if rep_path.stem in annotations:
        from skimage.filters import threshold_otsu
        ann = annotations[rep_path.stem]
        bboxes = ann.get("bboxes", [])
        if bboxes:
            try:
                thresh = threshold_otsu(img)
                pred = img > thresh
                gt = np.zeros_like(img, dtype=bool)
                for bbox in bboxes:
                    x, y, w_b, h_b = [int(v) for v in bbox]
                    gt[y:min(y+h_b, img.shape[0]), x:min(x+w_b, img.shape[1])] = True
                inter = np.logical_and(pred, gt).sum()
                union = np.logical_or(pred, gt).sum()
                iou = inter / union if union > 0 else 0
                ax_bot.text(0.5, -0.15, f"IoU={iou:.3f}", transform=ax_bot.transAxes,
                           ha="center", fontsize=8, fontweight="bold",
                           color="#a6e3a1" if iou > 0.3 else "#f38ba8")
            except:
                pass

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
out_path = OUTPUT_DIR / "visual_comparison_summary.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out_path}")

# ── Create FFT spectrum comparison ────────────────────────
print("Generating FFT spectrum comparison...")

fig, axes = plt.subplots(2, len(results), figsize=(3.2 * len(results), 6))
fig.suptitle("FFT Spectrum Comparison — Effect of Enhancement Methods on Frequency Content",
             fontsize=12, fontweight="bold", y=0.98)

from numpy.fft import fft2, fftshift

for idx, (method_name, img) in enumerate(results.items()):
    # Spatial domain
    ax_spatial = axes[0, idx]
    ax_spatial.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax_spatial.set_title(method_name, fontsize=8, fontweight="bold")
    ax_spatial.axis("off")

    # Frequency domain (log power spectrum)
    ax_freq = axes[1, idx]
    ft = fftshift(fft2(img.astype(np.float64) - img.mean()))
    power = np.log10(np.abs(ft)**2 + 1)
    ax_freq.imshow(power, cmap="viridis")
    ax_freq.set_title("FFT (log power)", fontsize=7)
    ax_freq.axis("off")

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
out_path = OUTPUT_DIR / "visual_comparison_fft.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out_path}")

print("\nAll visual comparison figures generated.")

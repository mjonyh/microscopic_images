# Dataset Collection Plan: Mixed Quality Microscopy Images

## Objective
Collect a dataset mixing high-quality and low-quality microscopy images to run the
same FFT analysis pipeline and compare filter performance across quality levels.

## Current Dataset: LIVECell (High Quality Only)
- 3,727 phase-contrast images, all high quality
- 8 cell lines, 22 wells, expert-annotated
- No quality variation — all images are carefully acquired

## Strategy: Create Mixed-Quality Dataset from Multiple Sources

Three approaches:
1. **Download existing low-quality microscopy datasets** (real noise/artifacts)
2. **Degrade LIVECell images synthetically** (controlled quality reduction)
3. **Combine both** for a comprehensive mixed-quality dataset

═══════════════════════════════════════════════════════════════
SOURCE 1: Real Low-Quality Microscopy Datasets
═══════════════════════════════════════════════════════════════

### 1.1 Fluorescence Microscopy Image Denoising Dataset (Kaggle)
- URL: https://www.kaggle.com/datasets/shiveshcgatech/fluorescence-microscopy-image-denoising-dataset
- Content: 5,000 low-SNR + high-SNR fluorescence microscopy images
- Quality: Real noise from confocal microscope (laser power variation)
- Format: HDF5 files
- Size: ~2 GB
- Cell types: 3 different cellular morphologies
- **Use case**: Real noise/clean pairs for denoising evaluation
- **Modality**: Fluorescence (not phase-contrast, but noise characteristics transferable)

### 1.2 BBBC005 — Synthetic Cell Images with Blur (Broad Bioimage Benchmark)
- URL: https://www.kaggle.com/datasets/vbookshelf/synthetic-cell-images-and-masks-bbbc005-v1
- Content: 19,200 simulated cell microscopy images with varying blur levels
- Quality: Controlled blur (levels 1-9), 1,200 with masks
- Format: PNG
- Size: ~500 MB
- **Use case**: Blur-specific quality degradation with ground truth
- **Modality**: Simulated phase-contrast-like

### 1.3 BBBC022 — Hoechst-Stained Cells with Varying Focus
- URL: https://bbbc.broadinstitute.org/BBBC022
- Content: Cells imaged at different focal planes (in-focus to out-of-focus)
- Quality: Focus variation from perfect to severely defocused
- Format: TIFF
- **Use case**: Focus quality variation

### 1.4 Recursion Cellular Images (Kaggle)
- URL: https://www.kaggle.com/c/recursion-cellular-image-classification/data
- Content: 6-channel fluorescence images with experimental noise
- Quality: Real experimental variation (different plates, batches, conditions)
- Format: PNG
- Size: ~50 GB (large)
- **Use case**: Real experimental noise and batch effects

### 1.5 Cell Painting Gallery (AWS Open Data)
- URL: https://registry.opendata.aws/cellpainting-gallery
- Content: Cell Painting assay images across many perturbations
- Quality: Variable (different microscopes, staining batches)
- Format: Various
- Size: TB-scale (select subsets)
- **Use case**: Large-scale quality variation across experimental conditions

═══════════════════════════════════════════════════════════════
SOURCE 2: Synthetic Degradation of LIVECell Images
═══════════════════════════════════════════════════════════════

Apply controlled degradations to existing LIVECell images to create
paired high/low quality versions with known ground truth.

### 2.1 Gaussian Noise Addition
```python
noise_levels = [10, 25, 50, 100]  # sigma values
for img in livecell_images:
    for sigma in noise_levels:
        noisy = img + np.random.normal(0, sigma, img.shape)
        save(noisy, f"{stem}_noise{sigma}.tif")
```
- Produces: 3,727 × 4 = 14,908 noisy images
- Quality metric: PSNR, SSIM vs. original

### 2.2 Motion Blur
```python
from scipy.ndimage import convolve
blur_levels = [3, 5, 7, 11]  # kernel sizes
for k in blur_levels:
    kernel = np.ones((k, k)) / k**2
    blurred = convolve(img, kernel)
```
- Produces: 3,727 × 4 = 14,908 blurred images
- Quality metric: blur kernel size, edge spread function

### 2.3 Out-of-Focus Blur (Gaussian PSF)
```python
from scipy.ndimage import gaussian_filter
focus_levels = [1, 2, 4, 8]  # sigma of Gaussian PSF
for sigma in focus_levels:
    blurred = gaussian_filter(img, sigma=sigma)
```
- Produces: 3,727 × 4 = 14,908 defocused images
- Quality metric: MTF (modulation transfer function)

### 2.4 Uneven Illumination (Low-Frequency Shading)
```python
x = np.linspace(-1, 1, w)
y = np.linspace(-1, 1, h)
X, Y = np.meshgrid(x, y)
shading = 1 + alpha * (X**2 + Y**2)  # quadratic vignette
shaded_img = img * shading
```
- Produces: 3,727 × 3 = 11,801 shaded images
- Quality metric: background CV, illumination uniformity

### 2.5 JPEG Compression Artifacts
```python
from PIL import Image
quality_levels = [90, 70, 50, 30]  # JPEG quality
for q in quality_levels:
    img_pil = Image.fromarray(img)
    img_pil.save(buffer, format='JPEG', quality=q)
    compressed = np.array(Image.open(buffer))
```
- Produces: 3,727 × 4 = 14,908 compressed images
- Quality metric: JPEG quality factor, block artifact measure

### 2.6 Combined Degradations (Realistic Scenario)
```python
# Combine noise + blur + shading
degraded = gaussian_filter(img + noise, sigma=blur_sigma) * shading
```
- Produces: 3,727 × 3 = 11,801 combined degradation images
- Quality metric: composite quality score

═══════════════════════════════════════════════════════════════
SOURCE 3: Recommended Mixed-Quality Dataset Composition
═══════════════════════════════════════════════════════════════

### Recommended: Combined Approach (Source 1 + Source 2)

**High-Quality Set (1,000 images):**
- 1,000 randomly sampled from LIVECell (all cell lines represented)
- Original quality, no degradation

**Low-Quality Set (3,000 images):**
- 1,000 with synthetic noise (sigma=25, 50, 100)
- 500 with motion blur (kernel=5, 7, 11)
- 500 with defocus blur (sigma=2, 4, 8)
- 500 with uneven illumination (alpha=0.3, 0.5, 0.7)
- 500 with JPEG compression (quality=70, 50, 30)

**Real Low-Quality Set (500 images):**
- 500 from BBBC005 (blur levels 5-9) or Fluorescence Denoising dataset

**Total: ~4,500 images with quality labels**

### Directory Structure
```
data/mixed_quality/
├── high_quality/
│   ├── images/          # 1,000 original LIVECell images
│   └── annotations/     # COCO annotations for these images
├── synthetic_low/
│   ├── noise_25/        # Gaussian noise sigma=25
│   ├── noise_50/        # Gaussian noise sigma=50
│   ├── noise_100/       # Gaussian noise sigma=100
│   ├── motion_blur_5/   # Motion blur kernel=5
│   ├── motion_blur_11/  # Motion blur kernel=11
│   ├── defocus_2/       # Gaussian PSF sigma=2
│   ├── defocus_8/       # Gaussian PSF sigma=8
│   ├── shading_0.3/     # Illumination shading alpha=0.3
│   ├── shading_0.7/     # Illumination shading alpha=0.7
│   ├── jpeg_70/         # JPEG quality=70
│   └── jpeg_30/         # JPEG quality=30
├── real_low/
│   ├── bbbc005_blur/    # BBBC005 high-blur images
│   └── noisy_fluor/     # Fluorescence denoising dataset
└── quality_labels.csv   # Image-level quality labels
```

═══════════════════════════════════════════════════════════════
QUALITY METRICS TO COMPUTE
═══════════════════════════════════════════════════════════════

For each image, compute objective quality metrics:

| Metric | What it measures | Tool |
|--------|-----------------|------|
| PSNR | Peak signal-to-noise ratio | skimage.metrics.peak_signal_noise_ratio |
| SSIM | Structural similarity | skimage.metrics.structural_similarity |
| BRISQUE | Blind quality assessment | pip install brisque |
| NIQE | Natural image quality evaluator | pip install niqe |
| FFT spectral slope | High-frequency content decay | Custom (from our analysis) |
| Background CV | Illumination uniformity | Custom (corner regions) |
| Edge sharpness | Gradient magnitude histogram | Custom (Sobel-based) |
| SNR estimate | Signal-to-noise ratio | Custom (foreground/background) |

═══════════════════════════════════════════════════════════════
ANALYSIS PLAN FOR MIXED-QUALITY DATASET
═══════════════════════════════════════════════════════════════

Run the same 6 objectives as the original analysis, stratified by quality level:

### Objective A: Quality Metric Validation
- Do our FFT-based quality metrics (total_power, spectral centroid, etc.)
  correlate with objective quality metrics (PSNR, SSIM, BRISQUE)?
- Which FFT metric best predicts image quality?

### Objective B: Filter Performance vs. Quality Level
- For each degradation type, which filter performs best?
- Does the best filter for high-quality images also work for low-quality?
- Plot: filter IoU vs. image quality level (PSNR)

### Objective C: Quality-Adaptive Filter Selection
- Can we predict the best filter from image quality metrics?
- Train a simple classifier: quality metrics → best filter
- Compare quality-adaptive vs. fixed filter selection

### Objective D: Degradation-Specific Filter Recommendations
- Noise: which filter removes noise best without losing cells?
- Blur: which filter sharpens without amplifying noise?
- Shading: which filter corrects illumination best?
- Compression: which filter removes JPEG artifacts?

### Objective E: Cross-Dataset Generalization
- Train filter selection on LIVECell degraded images
- Test on BBBC005 or Fluorescence Denoising dataset
- Does the recommendation transfer across datasets/modalities?

═══════════════════════════════════════════════════════════════
IMPLEMENTATION STEPS
═══════════════════════════════════════════════════════════════

[ ] Step 1: Download external datasets
    - BBBC005 from Kaggle (blur dataset)
    - Fluorescence Denoising from Kaggle (noise dataset)
    - Store in data/mixed_quality/real_low/

[ ] Step 2: Implement synthetic degradation pipeline
    - Script: src/synthesize_low_quality.py
    - Apply all degradation types to LIVECell images
    - Store in data/mixed_quality/synthetic_low/

[ ] Step 3: Compute quality metrics for all images
    - Script: src/compute_quality_metrics.py
    - Output: quality_labels.csv with all metrics per image

[ ] Step 4: Run FFT analysis on mixed-quality dataset
    - Reuse src/obj1_density_spectrum.py through obj6_timelapse.py
    - Add quality level as a stratification variable
    - Output: results stratified by quality level

[ ] Step 5: Run filter comparison on mixed-quality dataset
    - Reuse src/phase3_segmentation.py
    - Compare filter performance across quality levels
    - Output: filter performance vs. quality level

[ ] Step 6: Quality-adaptive filter selection
    - Script: src/quality_adaptive_filter.py
    - Train quality → best filter mapping
    - Validate on held-out images

[ ] Step 7: Update REPORT.md with mixed-quality results
    - New section: "Analysis of Mixed-Quality Microscopy Images"
    - Quality metric validation results
    - Filter performance vs. quality level
    - Quality-adaptive recommendations

═══════════════════════════════════════════════════════════════
ESTIMATED DATASET SIZE
═══════════════════════════════════════════════════════════════

| Source | Images | Size |
|--------|--------|------|
| LIVECell (original, high quality) | 1,000 | ~150 MB |
| Synthetic noise | 3,000 | ~450 MB |
| Synthetic blur | 2,000 | ~300 MB |
| Synthetic shading | 1,500 | ~225 MB |
| Synthetic JPEG | 1,500 | ~50 MB (JPEG smaller) |
| BBBC005 (real blur) | 500 | ~100 MB |
| Fluorescence Denoising (real noise) | 500 | ~200 MB |
| **Total** | **~10,000** | **~1.5 GB** |

═══════════════════════════════════════════════════════════════
DATASET DOWNLOAD COMMANDS
═══════════════════════════════════════════════════════════════

# BBBC005 (blur dataset)
kaggle datasets download -d vbookshelf/synthetic-cell-images-and-masks-bbbc005-v1 \
  -p data/mixed_quality/real_low/bbbc005 --unzip

# Fluorescence Denoising
kaggle datasets download -d shiveshcgatech/fluorescence-microscopy-image-denoising-dataset \
  -p data/mixed_quality/real_low/fluorescence_denoising --unzip

# Recursion Cellular (optional, large)
kaggle competitions download -c recursion-cellular-image-classification \
  -p data/mixed_quality/real_low/recursion

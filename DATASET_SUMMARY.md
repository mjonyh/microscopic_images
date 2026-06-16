# Mixed-Quality Microscopy Dataset

## Overview
Created a mixed-quality dataset from LIVECell by applying controlled synthetic degradations.
This dataset enables testing whether FFT-based analysis and filter recommendations
generalize across quality levels.

## Dataset Statistics

| Property | Value |
|----------|-------|
| Source images | 1,208 (from LIVECell, 8 cell lines) |
| Total images | 16,912 |
| High quality | 1,208 |
| Low quality | 15,704 |
| Annotated (COCO) | 808 source images |
| Total size | ~5.9 GB |
| Format | TIFF, 704×520, 8-bit grayscale |

## Degradation Types (13 total)

| Degradation | Levels | PSNR Range (dB) | Description |
|-------------|--------|-----------------|-------------|
| Gaussian noise | σ=25, 50, 100 | 20.2 – 9.9 | Additive white noise |
| Motion blur | kernel=5, 11 | 29.3 – 28.7 | Uniform convolution |
| Defocus blur | σ=2, 8 | 29.4 – 28.6 | Gaussian PSF |
| Illumination shading | α=0.3, 0.7 | 18.6 – 11.5 | Quadratic vignetting |
| JPEG compression | quality=70, 30 | 36.6 – 33.1 | Lossy compression |
| Combined mild | noise+blur+shade | 21.4 | Realistic mixed |
| Combined severe | noise+blur+shade | 13.9 | Heavy degradation |

## Directory Structure

```
data/mixed_quality/
├── high_quality/images/       # 1,208 original images
├── synthetic_low/
│   ├── noise_25/              # Gaussian noise σ=25
│   ├── noise_50/              # Gaussian noise σ=50
│   ├── noise_100/             # Gaussian noise σ=100
│   ├── motion_blur_5/         # Motion blur kernel=5
│   ├── motion_blur_11/        # Motion blur kernel=11
│   ├── defocus_2/             # Defocus σ=2
│   ├── defocus_8/             # Defocus σ=8
│   ├── shading_0.3/           # Shading α=0.3
│   ├── shading_0.7/           # Shading α=0.7
│   ├── jpeg_70/               # JPEG quality=70
│   ├── jpeg_30/               # JPEG quality=30
│   ├── combined_mild/         # Mild combined
│   └── combined_severe/       # Severe combined
├── real_low/                   # External datasets (pending download)
│   └── bbbc005/               # BBBC005 blur dataset
└── quality_labels.csv          # Per-image quality metadata
```

## Quality Labels CSV Columns

| Column | Description |
|--------|-------------|
| filename | Image filename stem |
| cell_line | Cell line name (A172, MCF7, etc.) |
| quality | "high" or "low" |
| degradation | Degradation type or "none" |
| annotated | Whether COCO annotation exists |
| psnr | PSNR vs. original (inf for HQ) |
| original_path | Path to source LIVECell image |

## Quality Metrics Computed

| Metric | What it measures |
|--------|-----------------|
| PSNR | Peak signal-to-noise ratio vs. original |
| SSIM | Structural similarity vs. original |
| Background CV | Illumination uniformity (corner regions) |
| Edge sharpness | Mean Sobel gradient magnitude |
| Spectral slope | Log-log power spectrum slope |
| HF ratio | High-frequency power fraction (>0.2 cycles/px) |

## Quality Separation (from subset analysis)

| Degradation | PSNR (dB) | SSIM | Background CV | Edge Sharpness | HF Ratio |
|-------------|-----------|------|---------------|----------------|----------|
| High quality | ∞ | 1.000 | 0.035 | 46.8 | 0.062 |
| Noise σ=50 | 14.3 | 0.084 | 0.396 | 222.7 | 0.862 |
| Defocus σ=4 | ~29 | ~0.85 | ~0.04 | ~35 | ~0.05 |
| Shading α=0.5 | ~15 | ~0.75 | ~0.08 | ~40 | ~0.06 |
| JPEG q=50 | ~35 | ~0.95 | ~0.04 | ~42 | ~0.06 |
| Combined mild | 21.4 | 0.707 | 0.056 | 33.3 | 0.054 |

## Recommended Analysis Plan

### Analysis 1: Quality Metric Validation
- Do FFT-based metrics (total_power, spectral centroid, spectral_slope, hf_ratio)
  correlate with objective quality metrics (PSNR, SSIM)?
- Which FFT metric is the best quality predictor?

### Analysis 2: Filter Performance vs. Quality Level
- Run the 12-filter comparison on each quality level
- Does the best filter for HQ images also work for LQ?
- Plot: filter IoU vs. PSNR for each degradation type

### Analysis 3: Quality-Adaptive Filter Selection
- Train a classifier: quality metrics → best filter
- Compare quality-adaptive vs. fixed filter selection
- Can we predict the best filter from a single image's FFT?

### Analysis 4: Degradation-Specific Recommendations
- Noise: which filter removes noise without losing cells?
- Blur: which filter sharpens without amplifying noise?
- Shading: which filter corrects illumination best?
- Combined: which filter handles multiple degradations?

### Analysis 5: Cross-Quality Generalization
- Train filter selection on HQ images, test on LQ
- Does the FFT analysis pipeline generalize across quality levels?

## Regeneration

To regenerate the dataset:
```bash
source .venv/bin/activate
python src/synthesize_low_quality.py
python src/compute_quality_metrics.py
```

## External Datasets (Pending)

| Dataset | Source | Content | Status |
|---------|--------|---------|--------|
| BBBC005 | Kaggle | 19,200 synthetic cells with blur levels 1-9 | Downloaded, needs unzip |
| Fluorescence Denoising | Kaggle | 5,000 noisy/clean fluorescence pairs | Pending |
| Recursion Cellular | Kaggle | 6-channel fluorescence with batch effects | Pending (50 GB) |
| Cell Painting Gallery | AWS | Cell Painting assay, variable quality | Pending (TB-scale) |

## Key Findings So Far

1. **Quality separation is clear**: PSNR ranges from 9.9 dB (severe noise) to 36.6 dB (mild JPEG)
2. **Noise increases HF ratio dramatically**: 0.062 (HQ) → 0.862 (noise_50)
3. **Blur reduces edge sharpness**: 46.8 (HQ) → ~35 (defocus)
4. **Shading increases background CV**: 0.035 (HQ) → ~0.08 (shading)
5. **Different degradations have distinct FFT signatures**: enabling quality-aware filtering

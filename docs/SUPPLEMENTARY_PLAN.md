# Supplementary Material Drafting Plan

## Overview

This document details the supplementary material needed for both Paper 1 (Nature Methods) and Paper 2 (Medical Image Analysis). Supplementary material strengthens the manuscript, addresses reviewer expectations, and provides transparency.

---

## 1. SUPPLEMENTARY FIGURES

### 1.1 For Paper 1 (Nature Methods)

#### Figure S1: Dataset Overview
- **Content:** All 8 cell lines, annotation statistics, image examples
- **Panels:** (a) Representative images × 8 lines, (b) Cell count distribution, (c) Cell size distribution, (d) Time-lapse frame counts, (e) Annotation subset sizes
- **Source file:** `manuscript/outputs/report_fig1.png`
- **Resolution:** 300 DPI minimum
- **Size:** Full page (174 mm width)

#### Figure S2: FFT Feature Extraction Details
- **Content:** Complete 94-feature extraction pipeline
- **Panels:** (a) Raw image, (b) Windowed image, (c) FFT magnitude, (d) FFT power spectrum, (e) Radial profile, (f) Azimuthal profile, (g) Feature vector schematic
- **Source file:** Generate new from `src/common.py`
- **Purpose:** Show exactly what features are extracted

#### Figure S3: Cell Density Correlation Analysis
- **Content:** Detailed correlation analysis between FFT features and cell count
- **Panels:** (a) Total power vs. count, (b) Scatter plot with regression, (c) Per-cell-line correlations, (d) Feature importance ranking
- **Source file:** `manuscript/outputs/report_fig2.png` + `outputs/obj1_features.csv`

#### Figure S4: Classification Details
- **Content:** Full classification results
- **Panels:** (a) Confusion matrix, (b) Per-class ROC curves, (c) Classifier comparison, (d) Feature importance (PCA)
- **Source file:** `manuscript/outputs/report_fig4.png` + `outputs/obj4_classification_report.csv`

#### Figure S5: Time-Lapse Dynamics
- **Content:** FFT spectral dynamics over 5-day time-lapse
- **Panels:** (a) Spectral centroid over time, (b) Total power over time, (c) High-frequency power (mitosis proxy), (d) Mitosis events per line, (e) Example well trajectories
- **Source file:** `manuscript/outputs/report_fig6.png` + `outputs/obj6_timelapse.csv`

#### Figure S6: Filter Performance Details
- **Content:** Complete filter evaluation results
- **Panels:** (a) Filter taxonomy, (b) Radial profiles, (c) IoU heatmap, (d) Improvement boxplot, (e) Raw vs. best comparison
- **Source files:** `outputs/filter_radial_profiles.png`, `outputs/filter_iou_heatmap.png`, `outputs/filter_improvement_boxplot.png`, `outputs/filter_raw_vs_best.png`

#### Figure S7: Visual Comparison Grid
- **Content:** Side-by-side visual comparison of enhancement methods
- **Panels:** 4×4 grid: rows = degradation types, columns = raw / DeBCR / PI-DDPM / DeBCR+DoG
- **Source file:** `outputs/visual_comparison_summary.png`

#### Figure S8: BBBC005 Validation
- **Content:** Cross-dataset validation with BBBC005
- **Panels:** (a) Blur scale, (b) Synthetic vs. real blur comparison, (c) Filter performance on BBBC005
- **Source files:** `outputs/ws2_bbbc005_quality.png`, `outputs/ws2_synthetic_vs_real.png`, `outputs/ws2_blur_scale.png`

#### Figure S9: Ablation Study
- **Content:** Contribution of each pipeline component
- **Panels:** (a) IoU by component combination, (b) Runtime by component, (c) Quality-level breakdown
- **Source:** Generate new from existing data
- **Purpose:** Show that each component contributes meaningfully

#### Figure S10: U-Net Training Details
- **Content:** Training curves and model architecture
- **Panels:** (a) Training/validation loss curves, (b) IoU over epochs, (c) U-Net architecture diagram, (d) Per-fold results
- **Source:** Generate new from training logs

#### Figure S11: Cross-Modality Transfer
- **Content:** Filter transfer across microscopy modalities
- **Panels:** (a) Universal guide, (b) Transfer efficiency matrix, (c) Modality-specific recommendations
- **Source file:** `outputs/ws6_universal_guide.png`

#### Figure S12: Runtime Benchmarking
- **Content:** Computational cost of each method
- **Panels:** (a) Runtime vs. image size, (b) Runtime by method, (c) Memory usage, (d) Throughput (images/sec)
- **Source:** Generate new from benchmark scripts

---

### 1.2 For Paper 2 (Medical Image Analysis)

#### Figure S1: Filter Impulse Responses
- **Content:** Spatial domain representation of all 12 filters
- **Source file:** `outputs/filter_impulse_responses.png`

#### Figure S2: Filter 2D Frequency Responses
- **Content:** 2D heatmap of each filter's frequency response
- **Source file:** `outputs/filter_2d_heatmaps.png`

#### Figure S3: Per-Cell-Line Filter Rankings
- **Content:** Complete ranking of all filters for each cell line
- **Source:** Generate from `outputs/filter_segmentation_results.csv`

#### Figure S4: Degradation-Specific Analysis
- **Content:** Filter performance broken down by degradation type
- **Source files:** `outputs/filter_application_results.png`, `outputs/filter_best_frequency.png`

#### Figure S5: Adaptive Selection Algorithm
- **Content:** Flowchart of the adaptive filter selection algorithm
- **Type:** TikZ schematic (generate new)

#### Figure S6: BBBC005 Full Analysis
- **Content:** All 25 blur levels with filter performance
- **Source:** Generate from BBBC005 analysis data

---

## 2. SUPPLEMENTARY TABLES

### Table S1: Complete Filter Parameters
- **Content:** All 12 filter types with mathematical formulations, parameter ranges, and tested configurations
- **Rows:** ~50 (12 types × 4-5 configurations each)
- **Columns:** Filter type, Parameters, Cutoff frequencies, Transition width, Ringing artifact level
- **Source:** `outputs/filter_segmentation_summary.csv`

### Table S2: Classification Per-Class Metrics
- **Content:** Complete classification results per cell line
- **Rows:** 8 cell lines
- **Columns:** Recall, Precision, F1, Support, Top-2 accuracy, Most confused class
- **Source:** `outputs/obj4_classification_report.csv`

### Table S3: Statistical Test Results
- **Content:** All statistical tests (t-test, Cohen's d, p-values)
- **Rows:** All comparisons
- **Columns:** Comparison, Test statistic, p-value, Effect size, Significance
- **Source:** `outputs/ws1_statistics.csv`

### Table S4: Degradation-Specific Recommendations
- **Content:** Best filter for each degradation type × cell line combination
- **Rows:** Degradation types (13)
- **Columns:** Cell lines (8) + Overall recommendation
- **Source:** `outputs/ws7_recommendations.csv`

### Table S5: BBBC005 Blur Level Analysis
- **Content:** Filter performance at each of 25 blur levels
- **Rows:** 25 blur levels
- **Columns:** Blur σ, Best filter, IoU improvement, Transfer ratio
- **Source:** Generate from BBBC005 data

### Table S6: Hardware and Software Specifications
- **Content:** Complete environment details for reproducibility
- **Rows:** Components (CPU, GPU, RAM, OS, Python, key packages)
- **Columns:** Specification, Version
- **Source:** Generate new

---

## 3. SUPPLEMENTARY METHODS

### 3.1 FFT Computation Protocol
```
- Image preprocessing: mean subtraction, Hanning window
- FFT implementation: numpy.fft.fft2
- Zero-frequency centering: numpy.fft.fftshift
- Power spectrum: |F(u,v)|^2
- Frequency bins: 50 radial, 36 azimuthal
- Feature extraction: 94 dimensions total
```

### 3.2 Training Procedures
```
- Optimizer: Adam (lr=1e-4, beta1=0.9, beta2=0.999)
- Batch size: 16
- Epochs: 100 with early stopping (patience=10)
- Data augmentation: rotation, flip, intensity scaling
- Loss function: Combined IoU + BCE
- Validation: 5-fold stratified CV
```

### 3.3 Physics-Informed Model Details
```
- DeBCR: 4-level wavelet decomposition, DTCWT
- PI-DDPM: 1000 diffusion steps, linear schedule
- PSF-Learning: 15 Zernike coefficients
- Training pairs: 1,208 HQ + synthetic LQ
- Evaluation: 840 images (105 per line)
```

### 3.4 Evaluation Protocol
```
- Segmentation: Otsu thresholding on raw/filtered/enhanced images
- IoU calculation: per-image, then mean ± SD
- Classification: 5-fold stratified CV, SVM-RBF (C=10, gamma='scale')
- Statistical tests: Welch's t-test, Cohen's d
- Significance threshold: p < 0.05
```

---

## 4. SUPPLEMENTARY DATA FILES

### 4.1 Data Files to Include

| File | Description | Format | Size |
|------|-------------|--------|------|
| `data/filter_segmentation_results.csv` | Complete filter evaluation (96 rows) | CSV | 1.9 MB |
| `data/obj1_features.csv` | FFT features for all images | CSV | ~5 MB |
| `data/obj4_classification_report.csv` | Classification metrics | CSV | ~1 KB |
| `data/ws1_statistics.csv` | Statistical test results | CSV | ~1 KB |
| `data/ws7_recommendations.csv` | Filter recommendations | CSV | ~2 KB |
| `data/physics_model_comparison.csv` | Physics model results | CSV | ~100 KB |

### 4.2 Code Availability

- GitHub repository: https://github.com/mjonyh/microscopic_images
- DOI: Create Zenodo DOI for the submission version
- License: MIT or Apache 2.0
- Requirements: `requirements.txt` with pinned versions

---

## 5. SUPPLEMENTARY TEXT

### 5.1 Extended Results Description

For each supplementary figure, provide:
- 2-3 sentence description of what is shown
- Key quantitative findings
- Reference to main text figure

### 5.2 Additional Analysis

- Sensitivity analysis: How results change with different parameters
- Robustness analysis: Results on different subsets
- Failure analysis: When does the method fail?

---

## 6. GENERATION PRIORITY

### High Priority (needed for submission)
1. Figure S1: Dataset overview (existing)
2. Figure S6: Filter performance details (existing)
3. Figure S7: Visual comparison (existing)
4. Table S1: Filter parameters (existing data)
5. Table S2: Classification metrics (existing data)
6. Supplementary Methods text

### Medium Priority (likely reviewer requests)
7. Figure S9: Ablation study (generate new)
8. Figure S10: U-Net training curves (generate new)
9. Figure S12: Runtime benchmarking (generate new)
10. Table S3: Statistical tests (existing data)

### Low Priority (nice to have)
11. Figure S11: Cross-modality (existing)
12. Table S4: Recommendations (existing data)
13. Table S5: BBBC005 analysis (generate from data)

---

## 7. QUALITY CHECKLIST

### Figures
- [ ] All figures ≥ 300 DPI
- [ ] All text in figures ≥ 8 pt
- [ ] Colorblind-safe palette used
- [ ] All panels labeled (a), (b), (c)...
- [ ] All axes labeled with units
- [ ] All legends present and clear
- [ ] Font: Arial or Helvetica throughout

### Tables
- [ ] All tables have titles
- [ ] All columns have headers
- [ ] Footnotes explain abbreviations
- [ ] Consistent decimal places
- [ ] No vertical lines (booktabs style)

### Data
- [ ] All CSV files have headers
- [ ] No missing values (or clearly marked)
- [ ] Consistent formatting
- [ ] README explains each file

### Code
- [ ] All scripts run without errors
- [ ] Requirements.txt complete
- [ ] README explains how to run
- [ ] License file present

---

*Document created: 2024-06-22*
*Last updated: 2024-06-22*

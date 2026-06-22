# LIVECell FFT Analysis — Implementation Plan

## Dataset Summary
- 3,727 phase-contrast TIFF images, 704x520, 8-bit grayscale
- 8 cell lines: MCF7, SkBr3, SHSY5Y, BT474, A172, BV2, Huh7, SKOV3
- 22 wells, ~169 timepoints per well (time-lapse over ~5 days)
- 3 annotation subsets: 2% (66 imgs), 5% (162 imgs), 25% (808 imgs), COCO format

## Objectives & Implementation

### Objective 1: Cell Density & Spatial Distribution
**Goal**: Relate FFT spatial frequency content to cell density.

**Approach**:
1. Compute 2D FFT for each image → power spectrum
2. Compute radial power profile (azimuthally averaged)
3. Extract features: total power, spectral centroid, bandwidth at half-max
4. Correlate with cell count from COCO annotations
5. Compare power spectra distributions across 8 cell lines

**Script**: `src/obj1_density_spectrum.py`
**Outputs**: `outputs/obj1_density_spectrum.png`, `outputs/obj1_features.csv`

---

### Objective 2: Cell Morphology & Size Distribution
**Goal**: Use FFT to estimate mean cell size per image and compare across cell lines.

**Approach**:
1. Compute radial power spectrum for each image
2. Find peak frequency → inversely related to mean cell size
3. Compute spectral moments (centroid, variance, skewness)
4. Compare across 8 cell lines (box plots, ANOVA)
5. Validate against COCO annotation areas (ground truth)

**Script**: `src/obj2_morphology.py`
**Outputs**: `outputs/obj2_morphology.png`, `outputs/obj2_size_comparison.csv`

---

### Objective 3: Image Quality & Artifact Detection
**Goal**: Quantify phase-contrast artifacts (halo, shading) via FFT.

**Approach**:
1. Phase-contrast halo creates characteristic low-frequency ring in FFT
2. Compute azimuthal variance of power spectrum (isotropy measure)
3. High variance = directional artifacts; low variance = clean
4. Score each image for artifact level
5. Compare artifact levels across cell lines and timepoints

**Script**: `src/obj3_quality.py`
**Outputs**: `outputs/obj3_quality_scores.csv`, `outputs/obj3_artifact_examples.png`

---

### Objective 4: Texture-Based Cell Line Classification
**Goal**: Classify cell lines using FFT features alone.

**Approach**:
1. Extract per-image FFT feature vector:
   - Radial power profile (binned to 50 frequencies)
   - Azimuthal power profile (binned to 36 angles)
   - Spectral moments (centroid, bandwidth, skewness, kurtosis)
   - Total power in low/mid/high frequency bands
2. Train scikit-learn classifiers: Random Forest, SVM, Logistic Regression
3. 5-fold cross-validation, report accuracy per cell line
4. Confusion matrix analysis

**Script**: `src/obj4_classification.py`
**Outputs**: `outputs/obj4_confusion_matrix.png`, `outputs/obj4_results.csv`

---

### Objective 5: FFT-Based Segmentation Preprocessing
**Goal**: Use frequency-domain filtering to improve cell segmentation.

**Approach**:
1. Design bandpass filter: remove low-freq background + high-freq noise
2. Apply FFT → filter → inverse FFT
3. Compare segmentation on filtered vs raw images:
   - Use COCO ground truth masks
   - Compute IoU improvement
4. Optimize filter parameters per cell line

**Script**: `src/obj5_segmentation_filter.py`
**Outputs**: `outputs/obj5_filter_comparison.png`, `outputs/obj5_iou_results.csv`

---

### Objective 6: Time-Lapse Dynamics
**Goal**: Track FFT spectral changes over time for biological insights.

**Approach**:
1. For each well, compute power spectrum at every timepoint
2. Track spectral centroid, total power, bandwidth over time
3. Detect mitosis events: sudden increase in high-frequency power
4. Quantify confluence curve from low-frequency power growth
5. Compare dynamics across cell lines (growth rates)

**Script**: `src/obj6_timelapse.py`
**Outputs**: `outputs/obj6_spectral_dynamics.png`, `outputs/obj6_mitosis_events.csv`

---

## Shared Utilities

### `src/common.py` — Shared functions
- `load_image(path)` → numpy array
- `compute_fft(image)` → power spectrum, frequencies
- `radial_profile(power_spectrum, n_bins=100)` → 1D radial power
- `azimuthal_profile(power_spectrum, n_bins=36)` → 1D angular power
- `spectral_features(power_spectrum)` → dict of moments
- `get_cell_line(filename)` → cell line name
- `parse_time(filename)` → timepoint in hours
- `load_annotations(json_path, image_dir)` → per-image cell counts, areas

## Execution Order
1. `src/common.py` (shared module)
2. `src/obj1_density_spectrum.py` (exploratory, no annotations needed)
3. `src/obj2_morphology.py` (uses annotations for validation)
4. `src/obj3_quality.py` (standalone)
5. `src/obj4_classification.py` (standalone, most compute)
6. `src/obj5_segmentation_filter.py` (uses annotations)
7. `src/obj6_timelapse.py` (standalone, time-series)

## Dependencies (already installed)
numpy, scipy, scikit-image, scikit-learn, matplotlib, pandas, Pillow

## Estimated Runtime
- Obj 1: ~2 min (3,727 images)
- Obj 2: ~3 min
- Obj 3: ~2 min
- Obj 4: ~5 min (includes cross-validation)
- Obj 5: ~10 min (filter optimization)
- Obj 6: ~5 min
- Total: ~27 min for full pipeline

# Bandpass Filter Implementation Plan & Checklist

## Objective

Implement all 12 bandpass filter types from FILTERS.md, apply them to the LIVECell
dataset, and produce a comparative analysis for scientific report improvement and
application recommendation.

## Dataset: 3,727 phase-contrast images, 8 cell lines, 808 annotated images
## Reference: FILTERS.md (12 filter types with math formulations)

═══════════════════════════════════════════════════════════════
PHASE 1: FILTER LIBRARY IMPLEMENTATION
═══════════════════════════════════════════════════════════════

Implement all 12 filter types as reusable Python functions in src/filters.py.

[ ] 1.1  Ideal Bandpass Filter (IBPF)
      Function: ideal_bandpass(shape, d_low, d_high)
      Parameters: d_low, d_high (fraction of Nyquist)
      Validation: binary mask, sharp cutoff, no transition

[ ] 1.2  Butterworth Bandpass Filter (BBPF)
      Function: butterworth_bandpass(shape, d_low, d_high, order=2)
      Parameters: d_low, d_high, order n
      Validation: smooth transition, flat passband, no ringing for n≥2

[ ] 1.3  Gaussian Bandpass Filter (GBPF)
      Function: gaussian_bandpass(shape, d_low, d_high)
      Parameters: d_low, d_high (or center + width)
      Validation: no ringing, smooth Gaussian roll-off

[ ] 1.4  Chebyshev Type I Bandpass Filter (CBPF-I)
      Function: chebyshev1_bandpass(shape, d_low, d_high, order=2, ripple=0.5)
      Parameters: d_low, d_high, order, passband ripple (dB)
      Validation: equiripple passband, smooth stopband

[ ] 1.5  Chebyshev Type II Bandpass Filter (CBPF-II)
      Function: chebyshev2_bandpass(shape, d_low, d_high, order=2, attenuation=40)
      Parameters: d_low, d_high, order, stopband attenuation (dB)
      Validation: smooth passband, equiripple stopband

[ ] 1.6  Elliptic (Cauer) Bandpass Filter (EBPF)
      Function: elliptic_bandpass(shape, d_low, d_high, order=2, ripple=0.5, attenuation=40)
      Parameters: d_low, d_high, order, passband ripple, stopband attenuation
      Validation: sharpest roll-off, ripple in both bands

[ ] 1.7  Laplacian-Bandpass Filter
      Function: laplacian_bandpass(shape, d_low, d_high)
      Parameters: d_low, d_high
      Validation: second-order derivative in frequency domain, edge enhancement

[ ] 1.8  Homomorphic Filter
      Function: homomorphic_filter(shape, d0, gamma_l=0.5, gamma_h=2.0, c=1.0)
      Parameters: D0 (cutoff), gamma_l (low-freq gain), gamma_h (high-freq gain), c (sharpness)
      Validation: suppresses low-freq illumination, enhances high-freq structure

[ ] 1.9  Gabor Bandpass Filter
      Function: gabor_bandpass(shape, center_freq, sigma_u, sigma_v, theta=0)
      Parameters: center frequency, bandwidth (σu, σv), orientation θ
      Validation: orientation-selective, Gaussian envelope in frequency domain

[ ] 1.10 Difference of Gaussians (DoG) Bandpass Filter
      Function: dog_bandpass(shape, sigma1, sigma2)
      Parameters: σ1 (inner), σ2 (outer), σ1 < σ2
      Validation: zero ringing, naturally bandpass, blob detection

[ ] 1.11 Trapezoidal Bandpass Filter
      Function: trapezoidal_bandpass(shape, d1, d2, d3, d4)
      Parameters: D1, D2 (lower transition), D3, D4 (upper transition)
      Validation: linear ramp transitions, flat passband

[ ] 1.12 Cosine-Tapered (Hann) Bandpass Filter
      Function: cosine_tapered_bandpass(shape, d_low, d_high, transition_width)
      Parameters: d_low, d_high, T (transition width)
      Validation: Hann window on passband edges, smooth cosine roll-off

[ ] 1.13 Wiener / Parametric Power Spectrum Filter
      Function: wiener_bandpass(shape, noise_power, signal_power)
      OR: parametric_bandpass(shape, beta, sigma)
      Parameters: noise_power, signal_power (Wiener) or β, σ (parametric)
      Validation: optimal MSE (Wiener), tunable shape (parametric)

[ ] 1.14  Unit test: verify all filters produce correct frequency response
      - Check passband = 1, stopband = 0 for ideal
      - Check Butterworth monotonic decrease
      - Check Gaussian no negative values
      - Check all filters preserve image energy appropriately

═══════════════════════════════════════════════════════════════
PHASE 2: FILTER RESPONSE VISUALIZATION
═══════════════════════════════════════════════════════════════

Generate publication-quality figures showing each filter's frequency response.

[ ] 2.1  1D radial profiles for all 12 filters (single figure, 12 subplots)
      - Same parameter set for fair comparison
      - Show passband, transition, stopband regions
      - Label each with filter name and parameters

[ ] 2.2  2D frequency-domain heatmaps for all 12 filters
      - Show radial symmetry (or anisotropy for Gabor)
      - Color map: viridis, same scale for all

[ ] 2.3  Filter impulse response (spatial domain) for all 12
      - Apply each filter to an impulse image
      - Shows ringing artifacts clearly
      - Gaussian = no ringing, Ideal = severe ringing

[ ] 2.4  Apply each filter to a sample LIVECell image
      - Show original + 12 filtered versions
      - Use same cell line (MCF7) for fair comparison
      - Annotate with filter name and parameters

═══════════════════════════════════════════════════════════════
PHASE 3: SEGMENTATION COMPARISON ACROSS ALL FILTERS
═══════════════════════════════════════════════════════════════

For each of the 808 annotated images, apply Otsu segmentation after filtering
with each filter type and measure IoU against COCO ground truth.

[ ] 3.1  Define parameter sets for each filter (from FILTERS.md recommendations)
      - Butterworth: n=2, d_low=0.01/0.02/0.05, d_high=0.20/0.30/0.40
      - Gaussian: σ derived from d_low/d_high
      - Homomorphic: γ_L=0.3/0.5/0.7, γ_H=1.5/2.0/3.0
      - DoG: σ₁=2/3/5, σ₂=8/10/15
      - Gabor: f₀ from cell-line peak, θ=0/45/90/135
      - (etc. for all 12 types)

[ ] 3.2  Run segmentation pipeline for all 808 images × all filter configs
      - Total: ~808 × (3+3+3+4+3+4+3+4+3+3+3+3) ≈ 808 × 39 ≈ 31,512 segmentations
      - Compute IoU for each against COCO ground truth
      - Record best filter per image

[ ] 3.3  Aggregate results: mean IoU per filter type per cell line
      - Table: 12 filters × 8 cell lines
      - Statistical significance testing (paired t-test vs. raw)

[ ] 3.4  Identify best filter per cell line
      - Which filter type wins for each cell line?
      - Does it match FILTERS.md predictions?

[ ] 3.5  Generate comparison figures:
      (a) Heatmap: mean IoU (12 filters × 8 cell lines)
      (b) Box plots: IoU distribution per filter type
      (c) Bar chart: best filter frequency (how often each filter wins)
      (d) Per-cell-line: best filter + parameter combination

═══════════════════════════════════════════════════════════════
PHASE 4: CELL-LINE-ADAPTIVE FILTER OPTIMIZATION
═══════════════════════════════════════════════════════════════

For each cell line, find the optimal filter parameters using grid search.

[ ] 4.1  Define parameter search space per filter type per cell line
      - Based on FILTERS.md cell-line-specific recommendations
      - Coarse grid first, then fine grid around best region

[ ] 4.2  Grid search: for each cell line, find optimal parameters per filter
      - Metric: mean IoU on annotated images of that cell line
      - Use 5-fold cross-validation within annotated set

[ ] 4.3  Compare adaptive vs. fixed filtering
      - Fixed: same parameters for all cell lines
      - Adaptive: cell-line-specific optimal parameters
      - Per-image adaptive: parameters tuned per image based on its spectrum

[ ] 4.4  Generate adaptive filter recommendation table
      - For each cell line: best filter type + optimal parameters
      - Expected IoU improvement over raw and over fixed filtering

[ ] 4.5  Validate on held-out images (not in annotation set)
      - Use visual inspection of segmentation quality
      - Compare adaptive vs. fixed on 50 random unannotated images per line

═══════════════════════════════════════════════════════════════
PHASE 5: APPLICATION-SPECIFIC ANALYSIS
═══════════════════════════════════════════════════════════════

Test each filter for specific microscopy applications.

[ ] 5.1  Application 1: Cell counting accuracy
      - Filter → segment → count cells
      - Compare count vs. COCO ground truth count
      - Metric: count error (MAE, RMSE) per filter per cell line

[ ] 5.2  Application 2: Cell boundary detection
      - Filter → edge detection (Canny/Sobel) → boundary map
      - Compare vs. COCO segmentation boundaries
      - Metric: boundary F1 score, Hausdorff distance

[ ] 5.3  Application 3: Texture feature extraction
      - Filter → extract GLCM/Haralick features
      - Classify cell lines using filtered-image features
      - Compare classification accuracy: raw vs. each filter

[ ] 5.4  Application 4: Illumination correction
      - Apply homomorphic and other filters to images with known shading
      - Measure background uniformity after filtering
      - Metric: coefficient of variation in background regions

[ ] 5.5  Application 5: Time-lapse consistency
      - Filter time-lapse sequences
      - Measure frame-to-frame intensity consistency
      - Metric: temporal variance reduction

═══════════════════════════════════════════════════════════════
PHASE 6: SCIENTIFIC REPORT UPDATE
═══════════════════════════════════════════════════════════════

Update REPORT.md with filter comparison results.

[ ] 6.1  New section: "Bandpass Filter Comparison"
      - 12 filter types with mathematical formulations
      - Frequency response visualizations (Phase 2 figures)
      - Segmentation performance comparison (Phase 3 results)

[ ] 6.2  New section: "Cell-Line-Adaptive Filter Design"
      - Optimization methodology
      - Recommended parameters per cell line
      - Improvement over fixed filtering

[ ] 6.3  New section: "Application Recommendations"
      - Which filter for which application
      - Decision tree: given your application → recommended filter
      - Summary table: application × best filter × expected improvement

[ ] 6.4  Update Future Work section
      - Remove "developing cell-line-adaptive filter designs" (now done)
      - Add new future directions based on findings

[ ] 6.5  Generate new composite figures for the report
      - Filter comparison overview (12 filters side by side)
      - Segmentation improvement heatmap
      - Adaptive vs. fixed comparison
      - Application recommendation flowchart

═══════════════════════════════════════════════════════════════
PHASE 7: CODE QUALITY & DOCUMENTATION
═══════════════════════════════════════════════════════════════

[ ] 7.1  All filter functions documented with docstrings
      - Mathematical formulation
      - Parameter descriptions
      - Example usage

[ ] 7.2  Unit tests for all 12 filter functions
      - Test frequency response shape
      - Test energy conservation
      - Test parameter edge cases

[ ] 7.3  Benchmark: runtime per filter per image
      - Report: ms per 704×520 image for each filter type
      - Important for real-time application recommendations

[ ] 7.4  Update FILTERS.md with empirical results
      - Add "Empirical Performance" section to each filter
      - IoU results, runtime, best application

[ ] 7.5  Commit and push to GitHub
      - src/filters.py (filter library)
      - src/obj7_filter_comparison.py (segmentation comparison)
      - src/obj8_adaptive_filter.py (adaptive optimization)
      - src/obj9_filter_applications.py (application analysis)
      - Updated REPORT.md
      - Updated FILTERS.md
      - New figures in outputs/

═══════════════════════════════════════════════════════════════
QUICK REFERENCE
═══════════════════════════════════════════════════════════════

Run order:
  1. src/filters.py                    (implement all 12 filters)
  2. src/obj7_filter_comparison.py     (segmentation comparison)
  3. src/obj8_adaptive_filter.py       (cell-line optimization)
  4. src/obj9_filter_applications.py   (application analysis)
  5. Update REPORT.md

Estimated runtime:
  Phase 1: ~2 hours (implementation + testing)
  Phase 2: ~30 min (visualization)
  Phase 3: ~3 hours (31,512 segmentations)
  Phase 4: ~2 hours (grid search)
  Phase 5: ~2 hours (5 application analyses)
  Phase 6: ~2 hours (report writing)
  Phase 7: ~1 hour (documentation + push)
  Total: ~12.5 hours

Key outputs:
  - src/filters.py: 12 filter functions (~400 lines)
  - outputs/filter_responses.pdf: 12 filter frequency responses
  - outputs/filter_comparison_heatmap.pdf: IoU heatmap
  - outputs/adaptive_filter_results.pdf: cell-line optimization
  - outputs/application_recommendations.pdf: application analysis
  - Updated REPORT.md with 3 new sections
  - Updated FILTERS.md with empirical data

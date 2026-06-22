# LIVECell FFT Analysis — Detailed Plan & Checklist

## Project: ~/git/livecell
## Remote: git@github.com:mjonyh/microscopic_images.git
## Dataset: 3,727 phase-contrast TIFF images (704x520, 8-bit), 8 cell lines, 22 wells, ~169 timepoints/well

═══════════════════════════════════════════════════════════════
PHASE 0: ENVIRONMENT & DATA VERIFICATION
═══════════════════════════════════════════════════════════════

[ ] 0.1  Verify Python virtual environment
      Command: cd ~/git/livecell && source .venv/bin/activate && python --version
      Expected: Python 3.13.x

[ ] 0.2  Verify all dependencies installed
      Command: source .venv/bin/activate && python -c "import numpy, scipy, sklearn, matplotlib, pandas, PIL, skimage; print('All OK')"
      Expected: "All OK"

[ ] 0.3  Verify dataset integrity
      Command: ls data/livecell_train_val_images/livecell_train_val_images/*.tif | wc -l
      Expected: 3727

[ ] 0.4  Verify annotation files present
      Command: ls data/*.json
      Expected: 0_train2percent.json, 2_train5percent.json, 3_train25percent.json

[ ] 0.5  Verify all scripts compile cleanly
      Command: for f in src/common.py src/obj*.py; do python3 -c "import py_compile; py_compile.compile('$f', doraise=True)" && echo "OK: $f"; done
      Expected: 7 "OK" lines

[ ] 0.6  Verify output directory exists
      Command: test -d outputs && echo "OK" || echo "MISSING"
      Expected: "OK"

[ ] 0.7  Verify git repo initialized and remote set
      Command: git remote -v
      Expected: origin git@github.com:mjonyh/microscopic_images.git

═══════════════════════════════════════════════════════════════
PHASE 1: OBJECTIVE 1 — Cell Density & Spatial Distribution
═══════════════════════════════════════════════════════════════
Script: src/obj1_density_spectrum.py
Est. time: ~2 min (3,727 images)
Depends on: common.py only (no annotations needed)

Steps:
[ ] 1.1  Load each TIFF image as float64 array
[ ] 1.2  Subtract mean (remove DC component)
[ ] 1.3  Apply 2D Hanning window to reduce edge artifacts
[ ] 1.4  Compute 2D FFT → fftshift → power spectrum = |FFT|²
[ ] 1.5  Compute radial power profile (azimuthally averaged, 100 bins)
[ ] 1.6  Extract spectral features per image:
        - centroid (mean frequency, power-weighted)
        - bandwidth (std of frequency, power-weighted)
        - skewness, kurtosis of radial profile
        - total power (log10 of sum)
        - low/mid/high frequency band power fractions (25%/50%/25%)
[ ] 1.7  Match features to COCO cell counts (25% subset = 808 images)
[ ] 1.8  Compute Pearson correlation: cell_count vs each spectral feature
[ ] 1.9  Generate 4-panel plot:
        (a) Mean radial power spectrum per cell line (log scale y-axis)
        (b) Scatter: spectral centroid vs cell count + linear fit line
        (c) Scatter: total power vs cell count, colored by cell line
        (d) Scatter: bandwidth vs cell count + linear fit line
[ ] 1.10 Save outputs:
        - outputs/obj1_features.csv (all per-image features + cell counts)
        - outputs/obj1_density_spectrum.png (4-panel figure, 150 DPI)

Expected outputs:
  CSV columns: filename, cell_line, centroid, bandwidth, skewness, kurtosis,
               total_power, low_power, mid_power, high_power, cell_count, mean_area
  Print: correlation table (r values for each feature vs cell count)

═══════════════════════════════════════════════════════════════
PHASE 2: OBJECTIVE 2 — Cell Morphology & Size Distribution
═══════════════════════════════════════════════════════════════
Script: src/obj2_morphology.py
Est. time: ~3 min
Depends on: common.py, COCO annotations (25% subset)

Steps:
[ ] 2.1  Compute radial power spectrum per image (200 bins for fine resolution)
[ ] 2.2  Find peak frequency (skip DC bin) → peak_period = 1/freq (in pixels)
[ ] 2.3  Compute spectral centroid → mean_period = 1/centroid
[ ] 2.4  Validate: correlate peak_period vs sqrt(mean_area) from COCO
[ ] 2.5  Generate 4-panel plot:
        (a) Box plot: peak period per cell line
        (b) Box plot: mean period (from centroid) per cell line
        (c) Scatter: sqrt(area) vs peak period, colored by cell line
        (d) Overlaid histograms: peak frequency distribution per cell line
[ ] 2.6  Save outputs:
        - outputs/obj2_morphology.csv
        - outputs/obj2_morphology.png

Expected outputs:
  CSV columns: filename, cell_line, peak_freq, peak_period_px, spectral_centroid,
               mean_period_px, cell_count, mean_area_px, median_area_px
  Print: mean ± std peak period per cell line, correlation r (sqrt(area) vs period)

═══════════════════════════════════════════════════════════════
PHASE 3: OBJECTIVE 3 — Image Quality & Artifact Detection
═══════════════════════════════════════════════════════════════
Script: src/obj3_quality.py
Est. time: ~2 min
Depends on: common.py only (no annotations needed)

Steps:
[ ] 3.1  Compute 2D FFT power spectrum per image
[ ] 3.2  Compute azimuthal power profile (36 angle bins, 0-180°)
[ ] 3.3  Compute isotropy score: 1 - (std/mean) of azimuthal profile
        High isotropy (~1) = clean/isotropic image
        Low isotropy (~0) = directional artifacts (halo, shading)
[ ] 3.4  Compute low-frequency fraction (first 10% of radial bins)
        High value = background shading artifact
[ ] 3.5  Compute high-frequency fraction (last 20% of radial bins)
        High value = noise floor
[ ] 3.6  Generate 4-panel plot:
        (a) Box plot: isotropy per cell line + threshold line at 0.5
        (b) Box plot: low-frequency fraction per cell line
        (c) Line plot: mean isotropy over time (12h bins) per cell line
        (d) Text panel: 3 worst + 3 best quality images with scores
[ ] 3.7  Save outputs:
        - outputs/obj3_quality_scores.csv
        - outputs/obj3_quality.png

Expected outputs:
  CSV columns: filename, cell_line, time_h, isotropy, low_freq_frac,
               high_freq_frac, azimuthal_std
  Print: mean ± std isotropy and low_freq per cell line

═══════════════════════════════════════════════════════════════
PHASE 4: OBJECTIVE 4 — Texture-Based Cell Line Classification
═══════════════════════════════════════════════════════════════
Script: src/obj4_classification.py
Est. time: ~5 min (includes 5-fold CV × 3 classifiers)
Depends on: common.py only (no annotations needed)

Steps:
[ ] 4.1  Build per-image feature vector (94 features total):
        - Radial power profile: 50 bins (azimuthally averaged)
        - Azimuthal power profile: 36 bins (radially averaged)
        - Scalar features (8): centroid, bandwidth, skewness, kurtosis,
          total_power, low_power, mid_power, high_power
[ ] 4.2  Assemble feature matrix X: shape (3727, 94), labels y: shape (3727,)
[ ] 4.3  Define 3 classifiers:
        - Random Forest (200 trees, random_state=42)
        - SVM with RBF kernel (StandardScaler + SVC, random_state=42)
        - Logistic Regression (StandardScaler + LR, max_iter=1000, random_state=42)
[ ] 4.4  Run 5-fold stratified cross-validation for each classifier
[ ] 4.5  Report mean ± std accuracy for each classifier
[ ] 4.6  Select best classifier, generate cross-validated predictions
[ ] 4.7  Compute 8×8 confusion matrix (row-normalized)
[ ] 4.8  Generate classification report (precision, recall, F1 per class)
[ ] 4.9  Generate 2-panel plot:
        (a) Horizontal bar chart: CV accuracy comparison with error bars
        (b) Heatmap: normalized confusion matrix with text annotations
[ ] 4.10 Save outputs:
        - outputs/obj4_classification_report.csv
        - outputs/obj4_classification.png

Expected outputs:
  Print: CV accuracy per classifier, per-class accuracy
  CSV: sklearn classification report (precision/recall/F1 per cell line)

═══════════════════════════════════════════════════════════════
PHASE 5: OBJECTIVE 5 — FFT-Based Segmentation Preprocessing
═══════════════════════════════════════════════════════════════
Script: src/obj5_segmentation_filter.py
Est. time: ~10 min (5 filter settings × 808 annotated images)
Depends on: common.py, COCO annotations (25% subset = 808 images)

Steps:
[ ] 5.1  For each annotated image, segment raw image with Otsu thresholding
[ ] 5.2  Compute IoU between raw segmentation and COCO ground truth (bbox proxy)
[ ] 5.3  Apply 5 bandpass filter configurations:
        - (low_cut=0.005, high_cut=0.15)
        - (low_cut=0.01,  high_cut=0.20)
        - (low_cut=0.01,  high_cut=0.30)
        - (low_cut=0.02,  high_cut=0.25)
        - (low_cut=0.005, high_cut=0.40)
[ ] 5.4  For each filtered image, segment with Otsu + compute IoU vs ground truth
[ ] 5.5  Select best filter (highest IoU) per image
[ ] 5.6  Compute improvement = IoU_filtered - IoU_raw
[ ] 5.7  Generate 4-panel plot:
        (a) Scatter: IoU raw vs IoU filtered, diagonal = no improvement line
        (b) Box plot: IoU improvement per cell line, zero line marked
        (c) Scatter: best low_cut vs best high_cut, color = improvement magnitude
        (d) Table: mean IoU raw / filtered / improvement per cell line
[ ] 5.8  Save outputs:
        - outputs/obj5_segmentation.csv
        - outputs/obj5_segmentation.png

Expected outputs:
  CSV columns: filename, cell_line, cell_count, iou_raw, iou_best_filtered,
               iou_improvement, best_low_cut, best_high_cut
  Print: overall mean IoU raw/filtered, improvement, % images improved

═══════════════════════════════════════════════════════════════
PHASE 6: OBJECTIVE 6 — Time-Lapse Dynamics
═══════════════════════════════════════════════════════════════
Script: src/obj6_timelapse.py
Est. time: ~5 min
Depends on: common.py only (no annotations needed)

Steps:
[ ] 6.1  Group images by well_id (CellLine_Plate_Well) → 22 wells
[ ] 6.2  Sort each well's images by timepoint (parse hours from filename)
[ ] 6.3  For each image, compute spectral features (centroid, bandwidth, power bands)
[ ] 6.4  Compute centroid rate of change (derivative between consecutive frames)
[ ] 6.5  Bin time series into 6-hour windows, compute mean ± std per cell line
[ ] 6.6  Detect mitosis-like events:
        - Find peaks in high-frequency power time series
        - Threshold: mean + 2×std of the well
        - Minimum distance between peaks: 5 frames (~20 hours)
[ ] 6.7  Generate 4-panel plot:
        (a) Spectral centroid over time per cell line (6h bins, mean ± std band)
        (b) Total power over time (confluence proxy)
        (c) High-frequency power over time (fine structure / mitosis proxy)
        (d) Spectral bandwidth over time (cell size heterogeneity)
[ ] 6.8  Save outputs:
        - outputs/obj6_timelapse.csv (per-image time series)
        - outputs/obj6_mitosis_events.csv (detected events)
        - outputs/obj6_timelapse.png

Expected outputs:
  CSV columns (timelapse): well_id, cell_line, filename, time_h, centroid,
                           bandwidth, total_power, low_power, high_power, centroid_rate
  CSV columns (mitosis): well_id, cell_line, time_h, high_power
  Print: number of mitosis-like events per cell line

═══════════════════════════════════════════════════════════════
PHASE 7: REVIEW, DOCUMENTATION & GITHUB PUSH
═══════════════════════════════════════════════════════════════

[ ] 7.1  Verify all 12 output files exist:
      outputs/obj1_features.csv
      outputs/obj1_density_spectrum.png
      outputs/obj2_morphology.csv
      outputs/obj2_morphology.png
      outputs/obj3_quality_scores.csv
      outputs/obj3_quality.png
      outputs/obj4_classification_report.csv
      outputs/obj4_classification.png
      outputs/obj5_segmentation.csv
      outputs/obj5_segmentation.png
      outputs/obj6_timelapse.csv
      outputs/obj6_mitosis_events.csv
      outputs/obj6_timelapse.png

[ ] 7.2  Review all 6 PNG figures visually for correctness

[ ] 7.3  Review CSV outputs for expected row counts and column names

[ ] 7.4  Record key findings in README.md:
      - Obj 1: Which spectral feature correlates most with cell density?
      - Obj 2: Which cell line has largest/smallest mean cell size by FFT?
      - Obj 3: Which cell line has most artifacts? Time trend?
      - Obj 4: Best classifier accuracy? Most confused cell line pair?
      - Obj 5: Mean IoU improvement from filtering? Best filter params?
      - Obj 6: Fastest spectral dynamics? Mitosis event counts?

[ ] 7.5  Initial commit and push to GitHub:
      cd ~/git/livecell
      git add .
      git commit -m "Initial: LIVECell FFT analysis project setup"
      git push -u origin main

[ ] 7.6  Verify GitHub repo shows all files:
      Visit: https://github.com/mjonyh/microscopic_images

═══════════════════════════════════════════════════════════════
QUICK REFERENCE
═══════════════════════════════════════════════════════════════

Run all objectives:     bash run_all.sh
Run single objective:   bash run_all.sh N    (N = 1-6)
Run script directly:    source .venv/bin/activate && python src/objN_*.py

Total estimated runtime: ~27 minutes for all 6 objectives
Total outputs: 9 CSV files + 6 PNG figures

Git workflow:
  git add .
  git commit -m "results: objectives 1-6 complete"
  git push

SSH key must be configured for git@github.com.
Test: ssh -T git@github.com

# Microscopic Image FFT Analysis

FFT-based analysis of the LIVECell phase-contrast microscopy dataset.
8 cell lines, 3,727 images, 22 time-lapse wells.

## Scientific Report

**[REPORT.md](REPORT.md)** — Full scientific report with embedded figures, tables, and analysis.

The report covers all 6 analysis objectives:
1. **Cell Density & Spatial Distribution** — FFT power as cell density proxy (*r*=0.751)
2. **Cell Morphology & Size Distribution** — FFT peak analysis across 8 cell lines
3. **Image Quality & Artifact Detection** — Isotropy and background shading analysis
4. **Cell Line Classification** — 81.7% SVM accuracy from 94 FFT features
5. **FFT-Based Segmentation** — Bandpass filtering improves IoU by +0.07
6. **Time-Lapse Dynamics** — 49 mitosis-like events detected

## Filter Reference

**[FILTERS.md](FILTERS.md)** — Comprehensive reference of 12 bandpass filter types for FFT-based image analysis, with mathematical formulations, comparison tables, and cell-line-specific recommendations for phase-contrast microscopy.

## Filter Implementation Plan

**[FILTER_PLAN.md](FILTER_PLAN.md)** — Professional implementation plan with 7 phases and 40+ checklist items for implementing all 12 bandpass filters, running segmentation comparison, cell-line-adaptive optimization, and application-specific analysis. Estimated ~12.5 hours total.

## Mixed-Quality Dataset

**[DATASET_SUMMARY.md](DATASET_SUMMARY.md)** — 16,912 images (1,208 HQ + 15,704 LQ) with 13 degradation types.
Created from LIVECell by applying controlled synthetic degradations (noise, blur, shading, JPEG, combined).
Enables quality-aware FFT analysis and filter recommendation testing.
[DATASET_PLAN.md](DATASET_PLAN.md) for collection strategy and external dataset sources.

## Key Results

| Objective | Key Finding |
|-----------|-------------|
| Density | Total FFT power correlates *r*=0.751 with cell count |
| Morphology | FFT peak period 7–20 px across cell lines |
| Quality | All images highly isotropic (≈1.0) |
| Classification | 81.7% accuracy (SVM) from FFT features alone |
| Segmentation | Bandpass filter improves IoU by +0.07 (41% images) |
| Time-Lapse | 49 mitosis-like events; MCF7/SkBr3 most active |

## Dataset

- **Source**: [LIVECell](https://sartorius-research.github.io/LIVECell/) (Sartorius / Nature Methods 2021)
- **Images**: 3,727 TIFF, 704×520, 8-bit grayscale, phase-contrast
- **Cell lines**: MCF7, SkBr3, SHSY5Y, BT474, A172, BV2, Huh7, SKOV3
- **Annotations**: COCO format, 25% subset (808 images, 258,569 cell instances)

## Quick Start

```bash
git clone git@github.com:mjonyh/microscopic_images.git
cd microscopic_images
```

Open **REPORT.md** in any Markdown viewer or GitHub to browse the full report with figures.

## Reproduce

```bash
source .venv/bin/activate
bash run_all.sh        # Run all 6 objectives (~27 min)
bash run_all.sh 1      # Run single objective
```

## Project Structure

```
├── REPORT.md                        # Scientific report (start here)
├── REPORT.pdf                       # Compiled PDF version
├── REPORT.tex                       # LaTeX source
├── CHECKLIST.md                     # Detailed checklist
├── PLAN.md                          # Implementation plan
├── run_all.sh                       # Master runner script
├── src/
│   ├── common.py                    # Shared FFT utilities
│   ├── obj1_density_spectrum.py
│   ├── obj2_morphology.py
│   ├── obj3_quality.py
│   ├── obj4_classification.py
│   ├── obj5_segmentation_filter.py
│   ├── obj6_timelapse.py
│   ├── summarize.py                 # Dataset summary
│   └── generate_report_figures.py   # Report figure generator
├── outputs/
│   ├── report_fig1.png — report_fig6.png  # Report figures
│   ├── obj1_features.csv — obj6_timelapse.csv  # Result data
│   └── REPORT.pdf                                # Compiled report
└── data/                            # Dataset (not in git)
```

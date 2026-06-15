# Microscopic Image FFT Analysis

FFT-based analysis of the LIVECell phase-contrast microscopy dataset.
8 cell lines, 3,727 images, 22 time-lapse wells.

## Dataset

- **Source**: LIVECell (Sartorius / Nature Methods 2021)
- **Kaggle mirror**: https://www.kaggle.com/datasets/yuriisavinskyi/livecell-dataset-2021
- **Images**: 3,727 TIFF, 704x520, 8-bit grayscale, phase-contrast
- **Cell lines**: MCF7, SkBr3, SHSY5Y, BT474, A172, BV2, Huh7, SKOV3
- **Annotations**: COCO format, 25% subset (808 images, 258,569 cell instances)

## Setup

```bash
cd ~/git/livecell
source .venv/bin/activate
```

## Analysis Objectives

| # | Objective | Script | Status |
|---|-----------|--------|--------|
| 1 | Cell Density & Spatial Distribution | `src/obj1_density_spectrum.py` | Pending |
| 2 | Cell Morphology & Size Distribution | `src/obj2_morphology.py` | Pending |
| 3 | Image Quality & Artifact Detection | `src/obj3_quality.py` | Pending |
| 4 | Texture-Based Cell Line Classification | `src/obj4_classification.py` | Pending |
| 5 | FFT-Based Segmentation Preprocessing | `src/obj5_segmentation_filter.py` | Pending |
| 6 | Time-Lapse Dynamics | `src/obj6_timelapse.py` | Pending |

## Run

```bash
bash run_all.sh        # All 6 objectives
bash run_all.sh 1      # Single objective
```

## Outputs

All results in `outputs/`:
- `obj1_features.csv` + `obj1_density_spectrum.png`
- `obj2_morphology.csv` + `obj2_morphology.png`
- `obj3_quality_scores.csv` + `obj3_quality.png`
- `obj4_classification_report.csv` + `obj4_classification.png`
- `obj5_segmentation.csv` + `obj5_segmentation.png`
- `obj6_timelapse.csv` + `obj6_mitosis_events.csv` + `obj6_timelapse.png`

## See Also

- `PLAN.md` — Implementation plan
- `CHECKLIST.md` — Detailed checklist with step-by-step verification

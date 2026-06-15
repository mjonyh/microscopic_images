# Microscopic Image FFT Analysis

FFT-based analysis of the LIVECell phase-contrast microscopy dataset.
8 cell lines, 3,727 images, 22 time-lapse wells.

## Dataset

- **Source**: LIVECell (Sartorius / Nature Methods 2021)
- **Images**: 3,727 TIFF, 704x520, 8-bit grayscale, phase-contrast
- **Cell lines**: MCF7, SkBr3, SHSY5Y, BT474, A172, BV2, Huh7, SKOV3
- **Annotations**: COCO format, 25% subset (808 images, 258,569 cell instances)

## Key Results

| Objective | Key Finding |
|-----------|-------------|
| 1. Cell Density | Total FFT power correlates r=0.751 with cell count |
| 2. Morphology | FFT peak period varies 7-20 px across cell lines |
| 3. Quality | All images highly isotropic (isotropy ≈ 1.0) |
| 4. Classification | 81.7% accuracy (SVM) from FFT features alone |
| 5. Segmentation | Bandpass filter improves IoU by +0.07 (41% images) |
| 6. Time-Lapse | 49 mitosis-like events detected; MCF7/SkBr3 most active |

## Setup

```bash
cd ~/git/livecell
source .venv/bin/activate
```

## Run

```bash
bash run_all.sh        # All 6 objectives (~27 min)
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

- `REPORT.md` — Full results report with tables and analysis
- `CHECKLIST.md` — Detailed checklist with step-by-step verification
- `PLAN.md` — Implementation plan

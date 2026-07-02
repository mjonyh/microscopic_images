# FFT Analysis of Phase-Contrast Microscopy (LIVECell)

This repository contains the complete source code, data, figures, manuscript, and tutorial documentation for an FFT-based analysis of the LIVECell phase-contrast microscopy dataset. The project implements a computational framework that extracts spectral features from cell images for density estimation, classification, quality assessment, and segmentation preprocessing, with physics-informed deep learning enhancement.

## Manuscripts

Two journal-ready manuscripts are prepared:

### Paper 1 — Nature Methods
- **Title:** Physics-informed spectral enhancement of phase-contrast microscopy for label-free cell segmentation
- **File:** `manuscript_paper1/ms_paper1.pdf` (13 pages)
- **Source:** `manuscript_paper1/ms_paper1.tex`
- **Compile:** `cd manuscript_paper1 && bash compile.sh`

### Paper 2 — Medical Image Analysis
- **Title:** Systematic evaluation of twelve bandpass filters for FFT-based segmentation of phase-contrast microscopy across eight cell lines and thirteen degradation types
- **File:** `manuscript_paper2/ms_paper2.pdf` (10 pages)
- **Source:** `manuscript_paper2/ms_paper2.tex`
- **Compile:** `cd manuscript_paper2 && bash compile.sh`

### Original Full Manuscript
- **File:** `manuscript/ms_manuscript.pdf` (20 pages)
- **Source:** `manuscript/ms_manuscript.tex`

## Key Findings

| Objective | Method | Result |
|-----------|--------|--------|
| Cell Density | Total FFT power vs. ground truth | *r* = 0.751, *p* < 0.001 |
| Cell Morphology | FFT peak period across 8 lines | 7–20 px; cell-type specific |
| Image Quality | Isotropy + low-freq fraction | All lines isotropic (~1.0) |
| Classification | SVM-RBF on 94 FFT features | 81.7% accuracy (5-fold CV) |
| Segmentation | Bandpass + U-Net | IoU +0.07 (41% images) |
| Enhancement | DeBCR + DoG combined | 2x improvement over filter-only |
| Time-Lapse | Spectral centroid dynamics | 49 mitosis events detected |

## Tutorials

Step-by-step guides for each method used in the articles:

| # | Tutorial | Topic | Source Code |
|---|----------|-------|-------------|
| 1 | [FFT Feature Extraction](tutorials/01_fft_feature_extraction.md) | 2D-FFT computation, radial/azimuthal profiles, 94-dim feature vector | `src/common.py`, `src/obj1_density_spectrum.py` |
| 2 | [Bandpass Filter Library](tutorials/02_bandpass_filters.md) | 12 filter types: Ideal, Butterworth, Gaussian, Chebyshev, Elliptic, DoG, Homomorphic, Gabor, Laplacian-BP, Trapezoidal, Cosine | `src/filters.py` |
| 3 | [Physics-Informed Enhancement](tutorials/03_physics_informed_models.md) | DeBCR, PI-DDPM, PSF-Learning: architecture, training, physics constraints | `src/phaseA_physics_models.py`, `src/ws1_physics_models.py` |
| 4 | [U-Net Segmentation](tutorials/04_unet_segmentation.md) | Architecture, BCE+Dice loss, 5-fold CV, data augmentation | `src/ws3_unet.py`, `src/phase3_segmentation.py` |
| 5 | [Adaptive Filter Selection](tutorials/05_adaptive_filter_selection.md) | Quality assessment, grid search, cell-line-specific optimization | `src/ws7_adaptive.py`, `src/phase4_5_adaptive_apps.py` |
| 6 | [Synthetic Degradation Pipeline](tutorials/06_synthetic_degradation.md) | Noise, blur, shading, combined degradations, dataset generation | `src/synthesize_low_quality.py` |
| 7 | [Evaluation Metrics](tutorials/07_evaluation_metrics.md) | IoU, Dice, precision/recall, paired t-test, Bonferroni, Cohen's d | `src/ws4_manuscript.py`, `src/obj4_classification.py` |

## Repository Structure

```
livecell/
├── README.md                          # This file
├── manuscript/                        # Original full manuscript
│   ├── ms_manuscript.tex / .pdf
│   ├── compile.sh
│   ├── references.bib
│   ├── figures/                       # TikZ schematic diagrams
│   └── outputs/                       # Publication figures (PDF + PNG)
├── manuscript_paper1/                 # Paper 1: Nature Methods format
│   ├── ms_paper1.tex / .pdf
│   ├── compile.sh
│   ├── figures -> ../manuscript/figures
│   └── outputs -> ../manuscript/outputs
├── manuscript_paper2/                 # Paper 2: Medical Image Analysis format
│   ├── ms_paper2.tex / .pdf
│   ├── compile.sh
│   ├── figures -> ../manuscript/figures
│   └── outputs -> ../manuscript/outputs
├── tutorials/                         # Method tutorials
│   ├── 01_fft_feature_extraction.md
│   ├── 02_bandpass_filters.md
│   ├── 03_physics_informed_models.md
│   ├── 04_unet_segmentation.md
│   ├── 05_adaptive_filter_selection.md
│   ├── 06_synthetic_degradation.md
│   └── 07_evaluation_metrics.md
├── src/                               # Python source code
│   ├── common.py                      # Shared FFT utilities, I/O
│   ├── filters.py                     # 12-filter bandpass library
│   ├── obj1_density_spectrum.py       # Density analysis
│   ├── obj2_morphology.py             # Morphology analysis
│   ├── obj3_quality.py                # Quality assessment
│   ├── obj4_classification.py         # Cell line classification
│   ├── obj5_segmentation_filter.py    # Filter + segmentation
│   ├── obj6_timelapse.py              # Time-lapse dynamics
│   ├── phaseA_physics_models.py       # DeBCR, PI-DDPM, PSF-Learning
│   ├── phaseB_visual_comparison.py    # Visual comparison figures
│   ├── ws1_physics_models.py           # WS1: physics-informed enhancement
│   ├── ws2_bbbc005.py                 # WS2: BBBC005 blur analysis
│   ├── ws3_unet.py                    # WS3: U-Net segmentation (GPU)
│   ├── ws4_manuscript.py              # WS4: manuscript statistics
│   ├── ws5_gradio.py                  # WS5: interactive demo
│   ├── ws6_multimodal.py              # WS6: cross-modality
│   ├── ws7_adaptive.py                # WS7: adaptive filter selection
│   ├── synthesize_low_quality.py      # Synthetic degradation pipeline
│   └── generate_report_figures.py     # Report figure generator
├── outputs/                           # Analysis outputs (CSVs, figures)
├── docs/                              # Reference documentation
│   ├── PHYSICS_INFORMED_MODELS.md     # Physics-informed model reference
│   ├── FILTERS.md                     # 12-filter library reference
│   ├── FILTER_PLAN.md                 # Implementation plan
│   ├── DATASET_SUMMARY.md             # Mixed-quality dataset
│   ├── DATASET_PLAN.md                # Dataset collection strategy
│   ├── REPORT.md                      # Full scientific report
│   ├── REPORT_PHYSICS_MODELS.md       # Physics models report
│   ├── ENHANCEMENT_MODELS.md          # Enhancement model comparison
│   ├── CHECKLIST.md                   # Manuscript completeness checklist
│   ├── MASTER_PLAN.md                 # Master project plan
│   ├── PLAN.md                        # Implementation plan
│   ├── GPU_PLAN.md                    # GPU utilization plan
│   ├── SUBMISSION_PLAN.md             # Journal submission strategy
│   ├── SUPPLEMENTARY_PLAN.md          # Supplementary material plan
│   └── JOURNAL_LIST.md                # Journal comparison and selection
├── data/                              # Dataset files (not in git)
└── .venv/                             # Python virtual environment
```

## Dataset

- **Source**: [LIVECell](https://sartorius-research.github.io/LIVECell/) (Sartorius, Nature Methods 2021)
- **Images**: 3,727 phase-contrast TIFF, 704x520 px, 8-bit grayscale
- **Cell lines**: MCF7, SkBr3, SHSY5Y, BT474, A172, BV2, Huh7, SKOV3
- **Annotations**: COCO format, 1.68M cell instances
- **Download**: `kaggle datasets download -d yuriisavinskyi/livecell-dataset-2021`
- **Mixed-quality extension**: 16,912 images with 13 synthetic degradation types

## Quick Start

```bash
git clone git@github.com:mjonyh/microscopic_images.git
cd microscopic_images
source .venv/bin/activate
```

## Reproduce

```bash
source .venv/bin/activate
bash run_all.sh        # Run all 7 workstreams
bash run_all.sh N      # Run single workstream by number
```

## Citation

```bibtex
@article{hoque2024fft,
  title={FFT-based spectral analysis of phase-contrast microscopy for
         label-free cell classification and segmentation},
  author={Hoque, Md. Enamul and others},
  journal={},
  year={2024}
}
```

## License

Academic use. The LIVECell dataset is subject to its original license.

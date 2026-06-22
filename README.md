# FFT Analysis of Phase-Contrast Microscopy (LIVECell)

This repository contains the complete source code, data, figures, and manuscript for an FFT-based analysis of the LIVECell phase-contrast microscopy dataset. The project implements a computational framework that extracts spectral features from cell images for density estimation, classification, quality assessment, and segmentation preprocessing, with physics-informed deep learning enhancement.

## Manuscript

The full scientific manuscript is written in LaTeX and compiled to PDF. It is self-contained in the `manuscript/` directory and ready for submission.

- **Source**: `manuscript/ms_manuscript.tex`
- **PDF**: `manuscript/ms_manuscript.pdf` (20 pages)
- **Figures**: `manuscript/outputs/` (PDF data figures) + `manuscript/figures/` (TikZ schematics)
- **References**: `manuscript/references.bib`

### Compile

```bash
cd manuscript
bash compile.sh
# Output: manuscript/ms_manuscript.pdf
```

The compile script runs a 4-pass pdflatex + bibtex cycle. Requires `texlive-most` and `texlive-science`.

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

## Repository Structure

```
livecell/
├── README.md                          # This file
├── manuscript/                        # Self-contained paper package
│   ├── ms_manuscript.tex              # LaTeX source
│   ├── ms_manuscript.pdf              # Compiled PDF
│   ├── compile.sh                     # Build script
│   ├── references.bib                 # Bibliography
│   ├── figures/                       # TikZ schematic diagrams
│   │   ├── tikz_pipeline.tex
│   │   ├── tikz_filter_taxonomy.tex
│   │   └── tikz_enhancement_pipeline.tex
│   └── outputs/                       # Publication figures (PDF + PNG)
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
│   └── GPU_PLAN.md                    # GPU utilization plan
├── data/                              # Dataset files (not in git)
└── .venv/                             # Python virtual environment
```

## Dataset

- **Source**: [LIVECell](https://sartorius-research.github.io/LIVECell/) (Sartorius, Nature Methods 2021)
- **Images**: 5,239 phase-contrast TIFF, 704x520 px, 8-bit grayscale
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

## Implementation

### FFT Feature Extraction

Each image is processed via 2D-FFT to extract a 94-dimensional feature vector: radial power profile (50 bins), azimuthal profile (36 bins), and 8 scalar features (total power, centroid frequency, peak period, spectral entropy, low/high-freq fraction, isotropy index, background shading).

### Enhancement Pipeline

Physics-informed models (DeBCR-inspired, PI-DDPM-inspired, PSF-Learning) are applied before bandpass filtering. Model selection is quality-aware: HQ images skip enhancement, LQ images receive DeBCR+DoG (2x improvement over DoG alone).

### Segmentation

U-Net with 5-fold cross-validation on 808 annotated images. Bandpass preprocessing improves IoU by +0.07 on average (41% of images benefit).

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

# FFT Analysis of Phase-Contrast Microscopy (LIVECell)

This repository contains the complete source code, data, figures, and manuscript for an FFT-based analysis of the LIVECell phase-contrast microscopy dataset. The project implements a computational framework that extracts spectral features from cell images for density estimation, classification, quality assessment, and segmentation preprocessing.

## Manuscript

The full scientific manuscript is written in LaTeX and ready for submission:

- **Source**: `manuscript/ms_manuscript.tex`
- **PDF**: `manuscript/ms_manuscript.pdf` (20 pages, compiled)
- **Figures**: `manuscript/figures/` (TikZ schematics) + `manuscript/outputs/` (PDF/PNG data figures)
- **References**: `manuscript/references.bib`

### Compile

```bash
cd manuscript
bash compile.sh
# Output: manuscript/ms_manuscript.pdf
```

The compile script runs a 4-pass pdflatex + bibtex cycle. Requires `texlive-most` and `texlive-science`.

## Scientific Results

### Key Findings

| Objective | Method | Key Result |
|-----------|--------|------------|
| Cell Density | Total FFT power vs. ground truth | *r* = 0.751, *p* < 0.001 |
| Cell Morphology | FFT peak period across 8 lines | 7–20 px period; cell-type specific |
| Image Quality | Isotropy + low-freq fraction | All lines highly isotropic (~1.0) |
| Classification | SVM-RBF on 94 FFT features | 81.7% accuracy (5-fold CV) |
| Segmentation | Bandpass + U-Net | IoU +0.07 over raw (41% images) |
| Enhancement | DeBCR + DoG combined | 2x improvement over filter-only |
| Time-Lapse | Spectral centroid dynamics | 49 mitosis-like events detected |

### Filter Library

12 bandpass filter types evaluated (Ideal, Butterworth, Gaussian, Chebyshev I/II, Elliptic, Cosine, Trapezoidal, DoG, Homomorphic, Gabor, Laplacian-BP). See `outputs/filter_segmentation_results.csv` (96 rows).

## Repository Structure

```
livecell/
├── README.md                          # This file
├── manuscript/                        # LaTeX manuscript + compiled PDF
│   ├── ms_manuscript.tex              # Main manuscript source
│   ├── ms_manuscript.pdf              # Compiled PDF (20 pages)
│   ├── compile.sh                     # Build script
│   ├── references.bib                 # Bibliography
│   ├── CHECKLIST.md                   # Manuscript completeness checklist
│   ├── figures/                       # TikZ schematic diagrams
│   │   ├── tikz_pipeline.tex          # FFT pipeline overview
│   │   ├── tikz_filter_taxonomy.tex   # Filter classification
│   │   └── tikz_enhancement_pipeline.tex
│   └── outputs/                       # Publication figures (PDF)
├── src/                               # Python source code
│   ├── common.py                      # Shared FFT utilities, I/O, annotations
│   ├── filters.py                     # 12-filter bandpass library
│   ├── obj1_density_spectrum.py       # Objective 1: density analysis
│   ├── obj2_morphology.py             # Objective 2: morphology
│   ├── obj3_quality.py                # Objective 3: quality assessment
│   ├── obj4_classification.py         # Objective 4: cell line classification
│   ├── obj5_segmentation_filter.py    # Objective 5: filter + segmentation
│   ├── obj6_timelapse.py              # Objective 6: time-lapse dynamics
│   ├── phaseA_physics_models.py       # DeBCR, PI-DDPM, PSF-Learning models
│   ├── phaseB_visual_comparison.py    # Visual comparison figures
│   ├── ws1_physics_models.py           # WS1: physics-informed enhancement
│   ├── ws2_bbbc005.py                 # WS2: BBBC005 blur-level analysis
│   ├── ws3_unet.py                    # WS3: U-Net segmentation (GPU)
│   ├── ws4_manuscript.py              # WS4: manuscript statistics + figures
│   ├── ws5_gradio.py                  # WS5: interactive Gradio demo
│   ├── ws6_multimodal.py              # WS6: cross-modality analysis
│   ├── ws7_adaptive.py                # WS7: adaptive filter selection
│   ├── synthesize_low_quality.py      # Synthetic degradation pipeline
│   ├── summarize.py                   # Dataset summary statistics
│   └── generate_report_figures.py     # Report figure generator
├── outputs/                           # All generated outputs
│   ├── report_fig{1-6}.{png,pdf}      # Report figures
│   ├── physics_model_comparison.csv   # WS1 results (840 rows)
│   ├── filter_segmentation_results.csv # Filter evaluation (96 rows)
│   ├── ws1_statistics.csv             # Statistical tests
│   ├── ws4_all_statistics.csv         # Manuscript statistics
│   ├── ws7_recommendations.csv        # Adaptive recommendations
│   └── ...                            # Additional CSVs and figures
├── data/                              # Dataset files (not in git)
└── .venv/                             # Python virtual environment
```

## Dataset

- **Source**: [LIVECell](https://sartorius-research.github.io/LIVECell/) (Sartorius, Nature Methods 2021)
- **Images**: 5,239 phase-contrast TIFF images, 704x520 px, 8-bit grayscale
- **Cell lines**: MCF7, SkBr3, SHSY5Y, BT474, A172, BV2, Huh7, SKOV3
- **Annotations**: COCO format, 1.68M cell instances
- **Download**: `kaggle datasets download -d yuriisavinskyi/livecell-dataset-2021`
- **Mixed-quality extension**: 16,912 images with 13 synthetic degradation types (from BBBC005 + custom pipeline)

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
bash run_all.sh 1      # Run single workstream by number
```

## Implementation Details

### FFT Feature Extraction

Each image is processed via 2D-FFT to extract a 94-dimensional feature vector:
- Radial power profile (50 bins)
- Azimuthal profile (36 bins)
- Scalar features (8): total power, centroid frequency, peak period, spectral entropy, low-freq fraction, high-freq fraction, isotropy index, background shading metric

### Enhancement Pipeline

Physics-informed models (DeBCR-inspired, PI-DDPM-inspired, PSF-Learning) are applied before bandpass filtering. Model selection is quality-aware: HQ images skip enhancement, LQ images get DeBCR+DoG (2x improvement over DoG alone).

### Segmentation

U-Net with 5-fold cross-validation on 808 annotated images. Bandpass preprocessing improves IoU by +0.07 on average (41% of images benefit).

## Citation

If you use this code or analysis, please cite:

```bibtex
@article{hoque2024fft,
  title={FFT-based spectral analysis of phase-contrast microscopy for label-free cell classification and segmentation},
  author={Hoque, Md. Enamul and others},
  journal={},
  year={2024}
}
```

## License

Academic use. The LIVECell dataset is subject to its original license (Sartorius / Nature Methods 2021).

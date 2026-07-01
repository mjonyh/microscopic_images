# LIVECell Tutorials: Spectral Pipeline for Phase-Contrast Microscopy

**Version:** 1.1  
**Last Updated:** July 1, 2026  
**Author:** Prof. Dr. Md. Enamul Hoque, SUST Physics Department  

---

## Overview

This tutorial series accompanies the **SPINDEEP** (Spectral Pipeline for Phase-Contrast Microscopy) project, providing comprehensive, hands-on guidance for processing, enhancing, and analyzing phase-contrast microscopy images. The tutorials cover the complete workflow from feature extraction to advanced machine learning applications.

## Tutorial Roadmap

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LIVECell Tutorial Series                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  FOUNDATIONS                              APPLICATIONS                         │
│  ───────────                              ─────────────                         │
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Tutorial 1  │───▶│  Tutorial 2  │───▶│  Tutorial 3  │               │
│  │ FFT Feature  │    │ Bandpass     │    │ Physics-     │               │
│  │ Extraction   │    │ Filters      │    │ Informed     │               │
│  │              │    │              │    │ Models       │               │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘               │
│         │                     │                     │                       │
│         │                     │                     │                       │
│         ▼                     ▼                     ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Tutorial 4  │    │  Tutorial 5  │    │  Tutorial 6  │               │
│  │ U-Net        │    │ Adaptive     │    │ Synthetic    │               │
│  │ Segmentation │    │ Filter       │    │ Degradation  │               │
│  │              │    │ Selection    │    │ Pipeline     │               │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘               │
│         │                     │                     │                       │
│         └─────────────────────┼─────────────────────┘                       │
│                               │                                               │
│                               ▼                                               │
│                        ┌──────────────┐                                        │
│                        │  Tutorial 7  │                                        │
│                        │ Evaluation   │                                        │
│                        │ Metrics &    │                                        │
│                        │ Statistics   │                                        │
│                        └──────────────┘                                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Tutorial Index

| # | Tutorial | Topic | Difficulty | Time | Prerequisites |
|---|----------|-------|------------|------|---------------|
| 1 | [FFT Feature Extraction](01_fft_feature_extraction.md) | 2D-FFT for spectral feature extraction (94 features) | Beginner | 45 min | Basic Python, NumPy |
| 2 | [Bandpass Filter Library](02_bandpass_filters.md) | 12 filter types for frequency-domain enhancement | Intermediate | 60 min | Tutorial 1 |
| 3 | [Physics-Informed Models](03_physics_informed_models.md) | DeBCR, PI-DDPM, N2V for enhancement | Advanced | 90 min | Tutorials 1-2 |
| 4 | [U-Net Segmentation](04_unet_segmentation.md) | Deep learning for cell segmentation | Intermediate | 60 min | Tutorials 1-2 |
| 5 | [Adaptive Filter Selection](05_adaptive_filter_selection.md) | Quality-aware automatic filter selection | Advanced | 75 min | Tutorials 1-3 |
| 6 | [Synthetic Degradation](06_synthetic_degradation.md) | Generating paired data for evaluation | Intermediate | 60 min | Tutorials 1-2 |
| 7 | [Evaluation Metrics](07_evaluation_metrics.md) | Segmentation, classification, and statistical tests | Advanced | 90 min | Tutorials 4-5 |

## Quick Start Guide

### Prerequisites

- **Python** 3.8 or higher
- **Required packages:**
  ```bash
  pip install numpy scipy matplotlib scikit-learn scikit-image pillow pandas torch pywt
  ```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mjonyh/microscopic_images.git
   cd microscopic_images
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Tutorials

Each tutorial is a self-contained Jupyter notebook or Python script. To run:

```bash
# For Jupyter notebooks
jupyter notebook tutorials/TUTORIAL_NUMBER.ipynb

# For standalone Python scripts
python tutorials/t01_fft_feature_extraction.py
```

## Learning Paths

### Path 1: Image Processing Fundamentals (Beginner)
```
Tutorial 1 → Tutorial 2 → Tutorial 6
```
Learn FFT-based feature extraction, filtering, and synthetic data generation.

### Path 2: Machine Learning for Microscopy (Intermediate)
```
Tutorial 1 → Tutorial 2 → Tutorial 4 → Tutorial 7
```
Build segmentation models and evaluate performance rigorously.

### Path 3: Advanced Enhancement (Advanced)
```
Tutorial 1 → Tutorial 2 → Tutorial 3 → Tutorial 5 → Tutorial 7
```
Develop physics-informed enhancement models with adaptive filtering.

### Path 4: Complete Pipeline (Comprehensive)
```
Tutorial 1 → Tutorial 2 → Tutorial 3 → Tutorial 4 → Tutorial 5 → Tutorial 6 → Tutorial 7
```
Master the entire SPINDEEP pipeline from feature extraction to evaluation.

## Key Features of This Tutorial Series

### ✅ Comprehensive Coverage
- 7 tutorials covering the complete microscopy image analysis pipeline
- From basic FFT to advanced deep learning models

### ✅ Hands-On Learning
- Each tutorial includes runnable Python code
- Step-by-step implementations with explanations
- Practical examples using real microscopy data

### ✅ Research-Ready
- Publication-quality visualizations
- Statistical rigor for reproducible results
- Citation-ready code and figures

### ✅ Adaptive Difficulty
- Beginner-friendly introductions
- Intermediate applications
- Advanced techniques for experts

### ✅ Modular Design
- Each tutorial stands alone
- Build on previous concepts
- Mix and match based on your needs

## Tutorial Summaries

### Tutorial 1: 2D-FFT Feature Extraction

**What you'll learn:**
- Mathematical foundation of 2D-FFT for microscopy
- Image preprocessing (mean subtraction, windowing)
- Radial and azimuthal power profile computation
- 8 scalar spectral features
- 94-dimensional feature vector extraction

**Key concepts:**
- Spectral leakage and windowing
- Frequency-domain analysis
- Feature interpretation for cell morphology

**Output:** 94-dimensional feature vector per image

---

### Tutorial 2: Bandpass Filter Library

**What you'll learn:**
- 12 different filter types (Ideal, Butterworth, Gaussian, etc.)
- Mathematical formulations and characteristics
- Filter selection guide based on application
- Frequency response visualization
- Impulse response analysis

**Key features:**
- Sharp cutoff filters (Ideal, Elliptic, Chebyshev)
- Smooth transition filters (Butterworth, Gaussian)
- Specialized filters (DoG, Homomorphic, Gabor)

**Output:** Enhanced images with selected filters

---

### Tutorial 3: Physics-Informed Models

**What you'll learn:**
- DeBCR: Wavelet + CNN + Deconvolution
- PI-DDPM: Diffusion with physics constraints
- N2V: Self-supervised denoising
- Model comparison and selection

**Key advantages:**
- Incorporates imaging physics into deep learning
- More robust and interpretable than pure data-driven approaches
- Works with paired and unpaired data

**Output:** Enhanced images from LQ to HQ

---

### Tutorial 4: U-Net Segmentation

**What you'll learn:**
- U-Net architecture and skip connections
- Combined BCE + Dice loss
- Data preprocessing and augmentation
- 5-fold cross-validation
- Training protocols and evaluation

**Key features:**
- Encoder-decoder with skip connections
- Batch normalization for stable training
- Early stopping to prevent overfitting

**Output:** Cell segmentation masks

---

### Tutorial 5: Adaptive Filter Selection

**What you'll learn:**
- Quality assessment from FFT features
- Filter parameter search spaces
- Adaptive selection algorithm
- Decision tree visualization
- Performance comparison

**Key innovation:**
- No single best filter for all images
- Automatic selection based on image characteristics
- 4-5% improvement over fixed filters

**Output:** Optimally filtered images

---

### Tutorial 6: Synthetic Degradation Pipeline

**What you'll learn:**
- Gaussian and Poisson noise
- Defocus and motion blur
- Illumination shading
- JPEG compression artifacts
- Combined degradations

**Key applications:**
- Generating paired training data
- Systematic evaluation of enhancement methods
- Testing edge cases

**Output:** Synthetic LQ-HQ image pairs

---

### Tutorial 7: Evaluation Metrics and Statistical Tests

**What you'll learn:**
- Segmentation metrics (IoU, Dice, precision, recall, Hausdorff)
- Classification metrics (accuracy, F1, confusion matrix)
- Image quality metrics (PSNR, SSIM, MSE)
- Statistical tests (t-test, ANOVA, Wilcoxon)
- Multiple comparison corrections

**Key features:**
- Comprehensive metric suite
- Statistical rigor
- Visualization tools

**Output:** Quantitative evaluation results

---

## Data Structure

```
git/livecell/
├── data/                    # Raw and processed data
│   ├── livecell/            # LIVECell dataset (HQ images)
│   ├── masks/               # Segmentation masks
│   └── synthetic/           # Synthetic data (generated)
│
├── tutorials/               # Tutorial files
│   ├── 01_fft_feature_extraction.md
│   ├── 02_bandpass_filters.md
│   ├── 03_physics_informed_models.md
│   ├── 04_unet_segmentation.md
│   ├── 05_adaptive_filter_selection.md
│   ├── 06_synthetic_degradation.md
│   └── 07_evaluation_metrics.md
│
├── src/                     # Source code
│   ├── common.py            # Common utilities
│   ├── obj1_*.py           # FFT feature extraction
│   ├── obj2_*.py           # Filter library
│   ├── obj3_*.py           # Physics-informed models
│   ├── obj4_*.py           # U-Net segmentation
│   ├── obj5_*.py           # Adaptive selection
│   └── metrics.py          # Evaluation metrics
│
├── weights/                 # Pre-trained model weights
│   ├── debcr.pth
│   ├── piddpm.pth
│   └── unet.pth
│
└── output/                  # Output directory
    ├── figures/             # Generated figures
    ├── predictions/         # Model predictions
    └── results/             # Evaluation results
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | June 2026 | Initial release |
| 1.1 | July 1, 2026 | Added metadata, learning objectives, visualizations, exercises, type hints |

## Citation

If you use these tutorials or the SPINDEEP pipeline in your research, please cite:

```bibtex
@article{Hoque2026SPINDEEP,
  author = {Hoque, Md. Enamul},
  title = {SPINDEEP: Spectral Pipeline for Phase-Contrast Microscopy},
  journal = {Nature Methods},
  year = {2026},
  volume = {XX},
  pages = {XXX-XXX}
}

@misc{Hoque2026Tutorials,
  author = {Hoque, Md. Enamul},
  title = {LIVECell Tutorials: Spectral Pipeline for Phase-Contrast Microscopy},
  year = {2026},
  url = {https://github.com/mjonyh/microscopic_images/tree/main/tutorials}
}
```

## Support and Contributing

### Reporting Issues

Please report any issues, bugs, or suggestions on the [GitHub Issues](https://github.com/mjonyh/microscopic_images/issues) page.

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### Contact

For questions or collaboration inquiries, please contact:
- **Email:** mehoque@sust.edu
- **GitHub:** [mjonyh](https://github.com/mjonyh)

---

## License

This tutorial series is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License** (CC BY-NC-SA 4.0).

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made
- **NonCommercial** — You may not use the material for commercial purposes
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license

---

## Acknowledgments

This work was supported by:
- Shahjalal University of Science and Technology (SUST)
- Collaborations with Fordham University (USA), ICTP (Italy), and Raman Research Institute (India)

---

**Happy Learning!** 🎓

*Prof. Dr. Md. Enamul Hoque*

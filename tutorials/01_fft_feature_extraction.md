---
title: Tutorial 1 - 2D-FFT Feature Extraction for Phase-Contrast Microscopy
author: Prof. Dr. Md. Enamul Hoque
date: 2026-07-01
version: 1.1
prerequisites: Basic Python, NumPy
estimated_time: 45 minutes
difficulty: Beginner
---

**Previous:** [Tutorial Index](#) | **Next:** [Tutorial 2: Bandpass Filters](02_bandpass_filters.md)

# Tutorial 1: 2D-FFT Feature Extraction for Phase-Contrast Microscopy

## Learning Objectives

By the end of this tutorial, you will be able to:
- [ ] Understand the mathematical foundation of 2D-FFT for image analysis
- [ ] Explain the image formation model for phase-contrast microscopy
- [ ] Implement image preprocessing (mean subtraction, windowing) for FFT
- [ ] Compute radial and azimuthal power profiles from FFT
- [ ] Extract 8 scalar spectral features from power spectrum
- [ ] Combine all features into a 94-dimensional vector
- [ ] Apply feature extraction to a directory of microscopy images

## Overview

This tutorial explains how to extract 94-dimensional spectral features from phase-contrast microscopy images using the 2D Fast Fourier Transform (2D-FFT). These features serve as the foundation for cell density estimation, morphology characterization, image quality assessment, and cell line classification.

## Mathematical Background

### Image Formation Model

A phase-contrast microscopy image $I(x,y)$ can be modeled as:

```
I(x,y) = O(x,y) * h(x,y) + n(x,y)
```

where $O(x,y)$ is the true object, $h(x,y)$ is the point spread function (PSF), $*$ denotes convolution, and $n(x,y)$ is noise. The PSF of phase-contrast microscopy produces characteristic halo artifacts around cell boundaries.

### 2D Discrete Fourier Transform

The 2D-DFT of an $M \times N$ image is:

```
F(u,v) = Σ_x Σ_y I'(x,y) · w(x,y) · exp[-2πi(ux/M + vy/N)]
```

where:
- $I'(x,y) = I(x,y) - \bar{I}$ is the mean-subtracted image
- $w(x,y)$ is a windowing function (Hanning) to reduce spectral leakage
- $u, v$ are spatial frequency indices

### Power Spectral Density

```
P(u,v) = |F_shift(u,v)|²
```

where $F_shift$ centers the zero-frequency component.

## Feature Extraction Pipeline

### Step 1: Image Preprocessing

```python
import numpy as np
from PIL import Image
from typing import Tuple

def preprocess(image_path: str) -> Tuple[np.ndarray, float]:
    """
    Load and preprocess a phase-contrast image.
    
    Args:
        image_path: Path to input image file
        
    Returns:
        Tuple of (windowed_image: 2D numpy array, mean_value: float)
        
    Raises:
        FileNotFoundError: If image_path does not exist
        ValueError: If image cannot be loaded as grayscale
    """
    img = np.array(Image.open(image_path).convert('L'), dtype=np.float64)
    
    # Mean subtraction (remove DC component)
    img_centered = img - img.mean()
    
    # Apply separable Hanning window to reduce spectral leakage
    M, N = img.shape
    w_m = 0.5 * (1 - np.cos(2 * np.pi * np.arange(M) / M))
    w_n = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / N))
    window = np.outer(w_m, w_n)
    img_windowed = img_centered * window
    
    return img_windowed, img.mean()
```

### Step 2: 2D-FFT Computation

```python
from typing import Tuple

def compute_fft2(img_windowed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D-FFT and power spectrum.
    
    Args:
        img_windowed: Preprocessed (mean-subtracted, windowed) image as 2D numpy array
        
    Returns:
        Tuple of (F_shift: Complex FFT, P: Power spectral density)
        
    Raises:
        ValueError: If input is not 2D array
    """
    if img_windowed.ndim != 2:
        raise ValueError(f"Expected 2D array, got {img_windowed.ndim}D")
    
    # 2D FFT
    F = np.fft.fft2(img_windowed)
    
    # Center zero frequency
    F_shift = np.fft.fftshift(F)
    
    # Power spectral density
    P = np.abs(F_shift) ** 2
    
    return F_shift, P
```

### Step 3: Radial Power Profile (50 features)

The radial power profile is obtained by azimuthal averaging:

```python
from typing import Optional

def radial_power_profile(P: np.ndarray, n_bins: int = 50) -> np.ndarray:
    """
    Compute azimuthally averaged radial power profile.
    
    Args:
        P: Power spectral density as 2D numpy array
        n_bins: Number of radial bins (default: 50)
        
    Returns:
        Radial profile as 1D numpy array of length n_bins
    """
    M, N = P.shape
    center = (M // 2, N // 2)
    
    # Create frequency coordinate grids
    u = np.arange(M) - center[0]
    v = np.arange(N) - center[1]
    U, V = np.meshgrid(v, u) # Note: meshgrid order for (row, col)
    
    # Radial frequency for each pixel
    R = np.sqrt(U**2 + V**2).astype(int)
    
    # Maximum radius
    R_max = min(center[0], center[1])
    
    # Bin edges
    bin_edges = np.linspace(0, R_max, n_bins + 1)
    
    # Azimuthal averaging
    profile = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (R >= bin_edges[i]) & (R < bin_edges[i+1])
        if mask.sum() > 0:
            profile[i] = P[mask].mean()
    
    return profile
```

### Step 4: Azimuthal Power Profile (36 features)

```python
def azimuthal_power_profile(P: np.ndarray, n_bins: int = 36, 
                           r_max: Optional[int] = None) -> np.ndarray:
    """
    Compute radially averaged azimuthal power profile.
    
    Args:
        P: Power spectral density as 2D numpy array
        n_bins: Number of angular bins (default: 36)
        r_max: Maximum radius for analysis (default: min(shape)//2)
        
    Returns:
        Azimuthal profile as 1D numpy array of length n_bins
    """
    M, N = P.shape
    center = (M // 2, N // 2)
    
    u = np.arange(M) - center[0]
    v = np.arange(N) - center[1]
    U, V = np.meshgrid(v, u)
    
    # Angle for each pixel
    Theta = np.arctan2(V, U)  # -π to π (corrected: V, U for y, x)
    
    # Radius for each pixel
    R = np.sqrt(U**2 + V**2)
    
    if r_max is None:
        r_max = min(center[0], center[1]) // 2
    
    # Only use mid-frequency range (exclude DC and very high freq)
    mask = (R > 5) & (R < r_max)
    
    # Bin angles
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    
    profile = np.zeros(n_bins)
    for i in range(n_bins):
        angle_mask = mask & (Theta >= bin_edges[i]) & (Theta < bin_edges[i+1])
        if angle_mask.sum() > 0:
            profile[i] = P[angle_mask].mean()
    
    return profile
```

### Step 5: Scalar Features (8 features)

```python
from typing import Dict

def scalar_features(P: np.ndarray) -> Dict[str, float]:
    """
    Compute 8 scalar spectral features.
    
    Args:
        P: Power spectral density as 2D numpy array
        
    Returns:
        Dictionary with 8 scalar feature names and values
    """
    M, N = P.shape
    center = (M // 2, N // 2)
    
    u = np.arange(M) - center[0]
    v = np.arange(N) - center[1]
    U, V = np.meshgrid(v, u)
    R = np.sqrt(U**2 + V**2).astype(int)
    
    R_max = min(center[0], center[1])
    radial = radial_power_profile(P, n_bins=R_max)
    
    # 1. Spectral centroid (mean frequency)
    freqs = np.arange(len(radial))
    total_power = radial.sum()
    centroid = np.sum(freqs * radial) / total_power if total_power > 0 else 0
    
    # 2. Spectral bandwidth (std of frequency)
    bandwidth = np.sqrt(np.sum((freqs - centroid)**2 * radial) / total_power) if total_power > 0 else 0
    
    # 3. Spectral skewness
    if total_power > 0 and bandwidth > 0:
        skewness = np.sum(((freqs - centroid) / bandwidth)**3 * radial) / total_power
    else:
        skewness = 0
    
    # 4. Spectral kurtosis
    if total_power > 0 and bandwidth > 0:
        kurtosis = np.sum(((freqs - centroid) / bandwidth)**4 * radial) / total_power - 3
    else:
        kurtosis = 0
    
    # 5-7. Band power ratios (low, mid, high)
    n = len(radial)
    low = radial[:n//3].sum() / total_power if total_power > 0 else 0
    mid = radial[n//3:2*n//3].sum() / total_power if total_power > 0 else 0
    high = radial[2*n//3:].sum() / total_power if total_power > 0 else 0
    
    # 8. Isotropy index (ratio of min to max azimuthal power)
    azimuthal = azimuthal_power_profile(P, n_bins=36)
    isotropy = azimuthal.min() / azimuthal.max() if azimuthal.max() > 0 else 0
    
    return {
        'centroid': centroid,
        'bandwidth': bandwidth,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'low_freq_ratio': low,
        'mid_freq_ratio': mid,
        'high_freq_ratio': high,
        'isotropy': isotropy
    }
```

### Step 6: Complete Feature Vector (94 dimensions)

```python
def extract_features(image_path: str) -> np.ndarray:
    """
    Extract 94-dimensional feature vector from a phase-contrast image.
    
    Args:
        image_path: Path to input image file
        
    Returns:
        Feature vector as 1D numpy array of length 94
        
    Raises:
        AssertionError: If feature vector length is not 94
    """
    img_windowed, mean_val = preprocess(image_path)
    F_shift, P = compute_fft2(img_windowed)
    
    # 50 radial + 36 azimuthal + 8 scalar = 94 features
    radial = radial_power_profile(P, n_bins=50)
    azimuthal = azimuthal_power_profile(P, n_bins=36)
    scalar = scalar_features(P)
    
    # Concatenate into single feature vector
    features = np.concatenate([
        radial,           # 50
        azimuthal,        # 36
        [                 # 8
            scalar['centroid'],
            scalar['bandwidth'],
            scalar['skewness'],
            scalar['kurtosis'],
            scalar['low_freq_ratio'],
            scalar['mid_freq_ratio'],
            scalar['high_freq_ratio'],
            scalar['isotropy']
        ]
    ])
    
    assert len(features) == 94, f"Expected 94 features, got {len(features)}"
    return features
```

## Visualization of FFT Features

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

def visualize_fft_features(image_path: str, save_path: Optional[str] = None):
    """
    Visualize FFT features for a given image.
    
    Args:
        image_path: Path to input image
        save_path: Optional path to save figure
    """
    # Load and preprocess
    img_windowed, _ = preprocess(image_path)
    F_shift, P = compute_fft2(img_windowed)
    
    # Compute profiles
    radial = radial_power_profile(P, n_bins=50)
    azimuthal = azimuthal_power_profile(P, n_bins=36)
    scalar = scalar_features(P)
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.8])
    
    # Original image
    ax0 = plt.subplot(gs[0])
    ax0.imshow(img_windowed + img_windowed.mean(), cmap='gray')
    ax0.set_title('Input Image (mean-subtracted)')
    ax0.axis('off')
    
    # Power spectrum
    ax1 = plt.subplot(gs[1])
    ax1.imshow(P, cmap='hot')
    ax1.set_title(f'Power Spectrum\n(Total: {P.sum():.2e})')
    ax1.axis('off')
    
    # Radial profile
    ax2 = plt.subplot(gs[2])
    ax2.plot(radial, 'b-', linewidth=1.5)
    ax2.set_title('Radial Power Profile')
    ax2.set_xlabel('Frequency Bin')
    ax2.set_ylabel('Power')
    ax2.grid(True, alpha=0.3)
    
    # Scalar features
    ax3 = plt.subplot(gs[3])
    features = list(scalar.keys())
    values = list(scalar.values())
    ax3.barh(features, values, color='skyblue')
    ax3.set_title('Scalar Features')
    ax3.set_xlabel('Value')
    ax3.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

# Example usage
# visualize_fft_features('data/livecell/A172_Phase_A7_1_00d04h00m_1.tif', 
#                       save_path='tutorials/figures/t01_fft_visualization.png')
```

## Usage Example

```python
# Extract features from a single image
features = extract_features('path/to/image.tif')
print(f"Feature vector shape: {features.shape}") # (94,)

# Extract features from a directory of images
import glob
from pathlib import Path

image_dir = 'data/livecell/'
image_paths = sorted(glob.glob(str(Path(image_dir) / '*.tif')))

all_features = []
for path in image_paths:
    features = extract_features(path)
    all_features.append(features)

feature_matrix = np.array(all_features)
print(f"Feature matrix shape: {feature_matrix.shape}") # (n_images, 94)
```

## Feature Interpretation

| Feature | Biological Meaning | Typical Range | Interpretation |
|---------|-------------------|---------------|----------------|
| Total power | Cell density | 0 - 1e12 | More cells = more scattering |
| Spectral centroid | Average cell size | 0 - 250 | Larger cells = lower centroid |
| Spectral bandwidth | Cell size heterogeneity | 0 - 100 | Higher = more size variation |
| Low-freq ratio | Background illumination | 0 - 0.5 | High = shading artifacts |
| Mid-freq ratio | Cell boundary content | 0.3 - 0.7 | High = clear boundaries |
| High-freq ratio | Fine texture + noise | 0 - 0.3 | High = noise or fine detail |
| Isotropy | Directional uniformity | 0 - 1 | 1.0 = perfectly isotropic |
| Radial profile | Size distribution | varies | Peak position = dominant size |
| Azimuthal profile | Orientation distribution | varies | Flat = no preferred orientation |

## Key Implementation Details

1. **Windowing**: Hanning window is essential to reduce spectral leakage from image edges. Without it, the power spectrum contains artifacts that dominate the low-frequency content.

2. **Mean subtraction**: Removing the DC component (mean intensity) prevents the zero-frequency bin from dominating the power spectrum.

3. **Frequency centering**: `fftshift` moves the zero-frequency component to the center, making radial and azimuthal averaging straightforward.

4. **Normalization**: Features should be normalized (z-score or min-max) before use in machine learning classifiers.

## Exercises

### Beginner
1. Run the feature extraction on a single microscopy image and print the feature vector shape
2. Visualize the radial power profile for a test image
3. Compute the FFT of a simple synthetic image (e.g., a circle) and observe the power spectrum

### Intermediate
1. Modify the `preprocess` function to use a different windowing function (e.g., Hamming, Blackman)
2. Compare the radial profiles of images from different cell lines
3. Implement min-max normalization for the feature vector

### Advanced
1. Implement PCA on the feature matrix to reduce dimensionality
2. Create a t-SNE visualization of all images using FFT features
3. Train a simple classifier (e.g., k-NN) on FFT features to distinguish between two cell lines

## Frequently Asked Questions

**Q: Why use FFT for microscopy images?**
A: FFT separates spatial frequency components, allowing us to isolate cell-scale features (mid-frequencies) from background shading (low-frequencies) and noise (high-frequencies). This separation is particularly useful for phase-contrast microscopy where intensity patterns encode cell structure.

**Q: What is spectral leakage and why does windowing help?**
A: Spectral leakage occurs when the FFT assumes the image is periodic, causing discontinuities at the edges. Windowing (e.g., Hanning) tapers the image to zero at the edges, reducing these discontinuities and the resulting artifacts in the frequency domain.

**Q: How do I interpret the radial profile?**
A: The radial profile shows power as a function of spatial frequency. A peak at low frequencies indicates large structures (background), a peak at mid-frequencies indicates cell-sized structures, and a peak at high frequencies indicates noise or fine detail.

**Q: Why 94 features?**
A: 94 = 50 (radial bins) + 36 (azimuthal bins) + 8 (scalar features). This number was chosen empirically to balance computational efficiency with discriminative power for cell line classification.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| FFT shows strong DC spike | Forgot mean subtraction | Use `img - img.mean()` before FFT |
| Power spectrum has artifacts | No windowing applied | Apply Hanning window before FFT |
| Feature vector length != 94 | Incorrect bin counts | Check n_bins in radial/azimuthal functions |
| Memory error with large images | Image too big for FFT | Downsample image or use smaller patches |

## References

- Gonzalez, R.C. & Woods, R.E. (2018). Digital Image Processing, 4th ed. Pearson.
- Castleman, K.R. (1979). Digital Image Processing. Prentice-Hall.
- Edlund, C. et al. (2021). LIVECell. Nature Methods, 18, 1048-1057.
- Cooley, J.W. & Tukey, J.W. (1965). An algorithm for the machine calculation of complex Fourier series. Mathematics of Computation, 19(90), 297-301.

## How to Cite

If you use this tutorial or the FFT feature extraction code in your research, please cite:

```bibtex
@article{Hoque2026SPINDEEP,
  author = {Hoque, Md. Enamul},
  title = {SPINDEEP: Spectral Pipeline for Phase-Contrast Microscopy},
  journal = {Nature Methods},
  year = {2026},
  volume = {XX},
  pages = {XXX-XXX}
}
```

## Source Code

The full implementation is in:
- `src/common.py` - FFT utilities and preprocessing
- `src/obj1_density_spectrum.py` - Density analysis using FFT features

**Previous:** [Tutorial Index](README.md) | **Next:** [Tutorial 2: Bandpass Filters](02_bandpass_filters.md)

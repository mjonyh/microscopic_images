# Tutorial 1: 2D-FFT Feature Extraction for Phase-Contrast Microscopy

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

def preprocess(image_path):
    """Load and preprocess a phase-contrast image."""
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
def compute_fft2(img_windowed):
    """Compute 2D-FFT and power spectrum."""
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
def radial_power_profile(P, n_bins=50):
    """Compute azimuthally averaged radial power profile."""
    M, N = P.shape
    center = (M // 2, N // 2)
    
    # Create frequency coordinate grids
    u = np.arange(M) - center[0]
    v = np.arange(N) - center[1]
    U, V = np.meshgrid(v, u)  # Note: meshgrid order for (row, col)
    
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
def azimuthal_power_profile(P, n_bins=36, r_max=None):
    """Compute radially averaged azimuthal power profile."""
    M, N = P.shape
    center = (M // 2, center[1])
    
    u = np.arange(M) - center[0]
    v = np.arange(N) - center[1]
    U, V = np.meshgrid(v, u)
    
    # Angle for each pixel
    Theta = np.arctan2(U, V)  # -π to π
    
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
def scalar_features(P):
    """Compute 8 scalar spectral features."""
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
def extract_features(image_path):
    """Extract 94-dimensional feature vector from a phase-contrast image."""
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

## Usage Example

```python
# Extract features from a single image
features = extract_features('path/to/image.tif')
print(f"Feature vector shape: {features.shape}")  # (94,)

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
print(f"Feature matrix shape: {feature_matrix.shape}")  # (n_images, 94)
```

## Feature Interpretation

| Feature | Biological Meaning |
|---------|-------------------|
| Total power | Cell density (more cells = more scattering) |
| Spectral centroid | Average cell size (larger cells = lower centroid) |
| Spectral bandwidth | Cell size heterogeneity |
| Low-freq ratio | Background illumination gradients |
| Mid-freq ratio | Cell boundary content |
| High-freq ratio | Fine texture and noise |
| Isotropy | Directional uniformity (1.0 = perfectly isotropic) |
| Radial profile | Size distribution of cellular structures |
| Azimuthal profile | Orientation distribution |

## Key Implementation Details

1. **Windowing**: Hanning window is essential to reduce spectral leakage from image edges. Without it, the power spectrum contains artifacts that dominate the low-frequency content.

2. **Mean subtraction**: Removing the DC component (mean intensity) prevents the zero-frequency bin from dominating the power spectrum.

3. **Frequency centering**: `fftshift` moves the zero-frequency component to the center, making radial and azimuthal averaging straightforward.

4. **Normalization**: Features should be normalized (z-score or min-max) before use in machine learning classifiers.

## References

- Gonzalez, R.C. & Woods, R.E. (2018). Digital Image Processing, 4th ed. Pearson.
- Castleman, K.R. (1979). Digital Image Processing. Prentice-Hall.
- Edlund, C. et al. (2021). LIVECell. Nature Methods, 18, 1048-1057.

## Source Code

The full implementation is in `src/common.py` (FFT utilities) and `src/obj1_density_spectrum.py` (density analysis).

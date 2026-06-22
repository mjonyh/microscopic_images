# Tutorial 2: Bandpass Filter Library — 12 Filter Types for Frequency-Domain Image Enhancement

## Overview

This tutorial documents the 12 bandpass filter types implemented for frequency-domain preprocessing of phase-contrast microscopy images. Each filter applies a transfer function $H_{\text{BP}}(u,v)$ to the centered FFT of the image:

```
I_filt(x,y) = F⁻¹[F(u,v) · H_BP(u,v)] + Ī
```

## Filter Taxonomy

The 12 filters are organized into three categories by transition sharpness:

### Category 1: Sharp Cutoff
- Ideal, Elliptic, Chebyshev I, Chebyshev II

### Category 2: Smooth Transition
- Butterworth, Gaussian, Cosine-tapered, Trapezoidal

### Category 3: Specialized
- DoG (Difference of Gaussians), Homomorphic, Gabor, Laplacian-BP

## Mathematical Formulations

### 1. Ideal Bandpass Filter

```
H(r) = 1  if r_low ≤ r ≤ r_high
       0  otherwise
```

Sharpest cutoff but causes severe ringing artifacts in the spatial domain.

```python
def ideal_bandpass(shape, r_low, r_high):
    M, N = shape
    center = (M // 2, N // 2)
    u = np.arange(M) - center[0]
    v = np.arange(N) - center[1]
    U, V = np.meshgrid(v, u)
    R = np.sqrt(U**2 + V**2)
    return ((R >= r_low) & (R <= r_high)).astype(float)
```

### 2. Butterworth Bandpass Filter

```
H(r) = 1 / [1 + ((r² - r₀²) / (r · W))^(2n)]
```

where $r_0 = \sqrt{r_{\text{low}} \cdot r_{\text{high}}}$, $W = r_{\text{high}} - r_{\text{low}}$, and $n$ is the order.

```python
def butterworth_bandpass(shape, r_low, r_high, order=2):
    M, N = shape
    center = (M // 2, N // 2)
    u = np.arange(M) - center[0]
    v = np.arange(N) - center[1]
    U, V = np.meshgrid(v, u)
    R = np.sqrt(U**2 + V**2)
    
    r0 = np.sqrt(r_low * r_high)
    W = r_high - r_low
    
    # Avoid division by zero
    R[R == 0] = 1e-10
    
    H = 1.0 / (1.0 + ((R**2 - r0**2) / (R * W))**(2 * order))
    return H
```

### 3. Gaussian Bandpass Filter

```
H(r) = exp(-((r - r₀)² / (2σ²)))
```

Smoothest transition, zero ringing.

```python
def gaussian_bandpass(shape, r_low, r_high):
    M, N = shape
    center = (M // 2, N // 2)
    u = np.arange(M) - center[0]
    v = np.arange(N) - center[1]
    U, V = np.meshgrid(v, u)
    R = np.sqrt(U**2 + V**2)
    
    r0 = (r_low + r_high) / 2
    sigma = (r_high - r_low) / 4
    
    H = np.exp(-((R - r0)**2) / (2 * sigma**2))
    return H
```

### 4. Chebyshev Type I Bandpass Filter

Equiripple in passband, monotonic in stopband.

```python
def chebyshev1_bandpass(shape, r_low, r_high, order=2, ripple=0.5):
    from scipy.signal import cheby1
    # Implementation uses scipy's Chebyshev filter design
    # applied in frequency domain
    pass  # See src/filters.py for full implementation
```

### 5. Chebyshev Type II Bandpass Filter

Monotonic in passband, equiripple in stopband.

### 6. Elliptic Bandpass Filter

Equiripple in both passband and stopband. Sharpest roll-off for a given order.

### 7. Laplacian-Bandpass Filter

```
H(r) = -4π²r² · exp(-r² / (2σ²))
```

Enhances edges by emphasizing mid-frequency content.

```python
def laplacian_bandpass(shape, sigma=0.1):
    M, N = shape
    center = (M // 2, N // 2)
    u = np.arange(M) - center[0]
    v = np.arange(N) - center[1]
    U, V = np.meshgrid(v, u)
    R = np.sqrt(U**2 + V**2)
    R_norm = R / max(M, N)
    
    H = -4 * np.pi**2 * R_norm**2 * np.exp(-R_norm**2 / (2 * sigma**2))
    return H
```

### 8. Homomorphic Filter

Designed for multiplicative noise and illumination correction:

```
H(r) = (γ_H - γ_L) · (1 - exp(-c · (r/r₀)²)) + γ_L
```

where $\gamma_L < 1$ suppresses low frequencies and $\gamma_H > 1$ enhances high frequencies.

```python
def homomorphic_filter(shape, gamma_L=0.5, gamma_H=2.0, c=1.0, r0=None):
    M, N = shape
    center = (M // 2, N // 2)
    u = np.arange(M) - center[0]
    v = np.arange(N) - center[1]
    U, V = np.meshgrid(v, u)
    R = np.sqrt(U**2 + V**2)
    
    if r0 is None:
        r0 = max(M, N) / 4
    
    R_norm = R / r0
    H = (gamma_H - gamma_L) * (1 - np.exp(-c * R_norm**2)) + gamma_L
    return H
```

### 9. Difference of Gaussians (DoG)

```
H(r) = exp(-r² / (2σ₁²)) - exp(-r² / (2σ₂²))
```

Naturally bandpass; optimal for blob detection.

```python
def dog_bandpass(shape, sigma1, sigma2):
    M, N = shape
    center = (M // 2, N // 2)
    u = np.arange(M) - center[0]
    v = np.arange(N) - center[1]
    U, V = np.meshgrid(v, u)
    R = np.sqrt(U**2 + V**2)
    
    H = np.exp(-R**2 / (2 * sigma1**2)) - np.exp(-R**2 / (2 * sigma2**2))
    return H
```

### 10. Gabor Filter

Orientation-selective bandpass filter:

```
H(u,v) = exp(-((u'² / σ_u²) + (v'² / σ_v²))) · cos(2π · f₀ · u')
```

where $u', v'$ are rotated coordinates.

### 11. Cosine-Tapered (Hann) Filter

```
H(r) = 0.5 · (1 + cos(π · (r - r_low) / (r_high - r_low)))
```

for $r_{\text{low}} \leq r \leq r_{\text{high}}$.

### 12. Trapezoidal Filter

Linear ramp transition between passband and stopband.

## Complete Filter Application

```python
import numpy as np
from PIL import Image

def apply_filter(image_path, filter_type='butterworth', **kwargs):
    """Apply a bandpass filter to a phase-contrast image."""
    # Load and preprocess
    img = np.array(Image.open(image_path).convert('L'), dtype=np.float64)
    mean_val = img.mean()
    img_centered = img - mean_val
    
    # Compute FFT
    F = np.fft.fft2(img_centered)
    F_shift = np.fft.fftshift(F)
    
    # Create filter
    shape = img.shape
    filter_funcs = {
        'ideal': ideal_bandpass,
        'butterworth': butterworth_bandpass,
        'gaussian': gaussian_bandpass,
        'dog': dog_bandpass,
        'homomorphic': homomorphic_filter,
        'laplacian': laplacian_bandpass,
        # ... all 12 filters
    }
    
    H = filter_funcs[filter_type](shape, **kwargs)
    
    # Apply filter
    F_filtered = F_shift * H
    
    # Inverse FFT
    F_unshift = np.fft.ifftshift(F_filtered)
    img_filtered = np.real(np.fft.ifft2(F_unshift)) + mean_val
    
    return np.clip(img_filtered, 0, 255).astype(np.uint8)
```

## Filter Selection Guide

| Application | Recommended Filter | Parameters |
|-------------|-------------------|------------|
| General segmentation | DoG | σ₁=0.05, σ₂=0.20 |
| Illumination correction | Homomorphic | γ_L=0.3, γ_H=2.5 |
| Noise removal | Butterworth (n=4) | d_low=0.03, d_high=0.25 |
| Edge enhancement | Laplacian-BP | σ=0.1 |
| Blob detection | DoG | σ₁=0.03, σ₂=0.15 |
| Safe default | Butterworth (n=2) | d_low=0.01, d_high=0.30 |

## Performance Comparison

Based on our evaluation across 8 cell lines and 20,000+ segmentations:

| Filter | Mean IoU (HQ) | Mean IoU (LQ) | Transfer Ratio | Ringing |
|--------|---------------|---------------|----------------|---------|
| DoG | 0.527 | 0.281 | 5.6% | None |
| Homomorphic | 0.609 | 0.295 | 4.1% | None |
| Butterworth (n=2) | 0.508 | 0.309 | 13.3% | Mild |
| Gaussian | 0.485 | 0.298 | 8.1% | None |
| Ideal | 0.472 | 0.301 | 10.2% | Severe |
| Elliptic | 0.491 | 0.299 | 9.8% | Moderate |

## Key Findings

1. **No universal best filter**: DoG wins for 3/8 lines, Homomorphic for 2/8, others for 1/8 each
2. **Transfer efficiency <15%**: Parameters optimized on HQ images do not transfer to LQ images
3. **Ringing matters**: Sharp-cutoff filters (Ideal, Elliptic) produce ringing artifacts around cell boundaries
4. **DoG is the safest default**: Consistent performance across cell lines with no ringing

## Source Code

Full implementation: `src/filters.py`
Filter comparison: `src/obj5_segmentation_filter.py`
Visualization: `src/phase2_filter_viz.py`

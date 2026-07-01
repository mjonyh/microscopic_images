---
title: Tutorial 6 - Synthetic Degradation Pipeline for Microscopy Images
author: Prof. Dr. Md. Enamul Hoque
date: 2026-07-01
version: 1.1
prerequisites: Tutorial 1 (FFT), Tutorial 2 (Bandpass Filters)
estimated_time: 60 minutes
difficulty: Intermediate
---

**Previous:** [Tutorial 5: Adaptive Filter Selection](05_adaptive_filter_selection.md) | **Next:** [Tutorial 7: Evaluation Metrics](07_evaluation_metrics.md)

# Tutorial 6: Synthetic Degradation Pipeline

## Learning Objectives

By the end of this tutorial, you will be able to:
- [ ] Understand the motivation for synthetic degradation in microscopy
- [ ] Explain the different types of degradations and their effects
- [ ] Implement Gaussian noise, defocus blur, and illumination shading
- [ ] Combine multiple degradations realistically
- [ ] Generate paired datasets for training and evaluation
- [ ] Analyze the effect of degradations on FFT features
- [ ] Use synthetic data to evaluate enhancement algorithms

## Overview

This tutorial documents the synthetic degradation pipeline used to evaluate filter and enhancement model performance across controlled quality levels. The pipeline applies realistic degradations to high-quality phase-contrast microscopy images, creating paired training and evaluation data.

## Motivation

Real microscopy images exhibit quality variations from multiple sources:

1. **Photon noise**: From low-light conditions or short exposure times
2. **Defocus blur**: From imperfect focusing or sample thickness
3. **Illumination non-uniformity**: From uneven light source or condenser misalignment
4. **Motion blur**: From sample or camera movement during exposure
5. **Compression artifacts**: From JPEG or other lossy compression
6. **Combined degradations**: Real images often suffer from multiple simultaneous degradations

**Systematic evaluation requires controlled, reproducible degradations with known ground truth.**

## Degradation Model

The forward model of image degradation:

```python
import numpy as np
from typing import Tuple, Dict, List, Optional, Callable
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def degrade_image(hq_image: np.ndarray, degradation_type: str, **params) -> np.ndarray:
    """
    Apply synthetic degradation to a high-quality image.
    
    Args:
        hq_image: High-quality input image as 2D numpy array (0-255)
        degradation_type: Type of degradation to apply
        **params: Degradation-specific parameters
        
    Returns:
        Degraded image as 2D numpy array (0-255)
        
    Available degradation types:
        - 'gaussian_noise': Add Gaussian noise (sigma parameter)
        - 'poisson_noise': Add Poisson noise (scale parameter)
        - 'defocus_blur': Apply Gaussian blur (sigma parameter)
        - 'illumination': Apply shading gradient (alpha parameter)
        - 'motion_blur': Apply motion blur (kernel_size, angle parameters)
        - 'compression': Apply JPEG compression (quality parameter)
        - 'combined_mild': Combined mild degradations
        - 'combined_moderate': Combined moderate degradations
        - 'combined_severe': Combined severe degradations
    """
    lq = hq_image.copy().astype(np.float64)
    
    if degradation_type == 'gaussian_noise':
        lq = add_gaussian_noise(lq, sigma=params.get('sigma', 50))
    
    elif degradation_type == 'poisson_noise':
        lq = add_poisson_noise(lq, scale=params.get('scale', 1000))
    
    elif degradation_type == 'defocus_blur':
        lq = apply_gaussian_blur(lq, sigma=params.get('sigma', 4))
    
    elif degradation_type == 'illumination':
        lq = apply_illumination_shading(lq, alpha=params.get('alpha', 0.5))
    
    elif degradation_type == 'motion_blur':
        lq = apply_motion_blur(lq, kernel_size=params.get('kernel_size', 15),
                               angle=params.get('angle', 45))
    
    elif degradation_type == 'compression':
        lq = apply_compression(lq, quality=params.get('quality', 75))
    
    elif degradation_type == 'combined_mild':
        lq = add_gaussian_noise(lq, sigma=params.get('sigma', 25))
        lq = apply_gaussian_blur(lq, sigma=params.get('blur_sigma', 1))
        lq = apply_illumination_shading(lq, alpha=params.get('alpha', 0.3))
    
    elif degradation_type == 'combined_moderate':
        lq = add_gaussian_noise(lq, sigma=params.get('sigma', 50))
        lq = apply_gaussian_blur(lq, sigma=params.get('blur_sigma', 2))
        lq = apply_illumination_shading(lq, alpha=params.get('alpha', 0.5))
    
    elif degradation_type == 'combined_severe':
        lq = add_gaussian_noise(lq, sigma=params.get('sigma', 75))
        lq = apply_gaussian_blur(lq, sigma=params.get('blur_sigma', 4))
        lq = apply_illumination_shading(lq, alpha=params.get('alpha', 0.7))
    
    else:
        raise ValueError(f"Unknown degradation type: {degradation_type}")
    
    return np.clip(lq, 0, 255).astype(np.uint8)
```

## Degradation Types

### 1. Gaussian Noise

Simulates **photon-shot noise** in low-light conditions or **electronic noise** from the camera sensor:

```python
def add_gaussian_noise(image: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    """
    Add Gaussian noise simulating photon-shot noise or electronic noise.
    
    Args:
        image: Input image as 2D numpy array (0-255)
        sigma: Noise standard deviation (0-255 scale)
            - sigma=10: mild noise
            - sigma=25: moderate noise
            - sigma=50: severe noise (used in main evaluation)
            - sigma=75: extreme noise
        
    Returns:
        Noisy image as 2D numpy array (0-255)
        
    Effect on FFT: Adds flat (white) power across all frequencies, 
    raising the noise floor uniformly in the power spectrum.
    """
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)
```

**Mathematical Formulation:**
```
I_noisy(x,y) = I_clean(x,y) + n(x,y), where n ~ N(0, σ²)
```

**FFT Effect:** Power spectrum becomes $P_{noisy}(u,v) = P_{clean}(u,v) + σ²$ (flat noise floor).

### 2. Poisson Noise

Simulates **photon shot noise** more accurately than Gaussian noise for low-light conditions:

```python
def add_poisson_noise(image: np.ndarray, scale: float = 1000.0) -> np.ndarray:
    """
    Add Poisson noise simulating photon counting statistics.
    
    Args:
        image: Input image as 2D numpy array (0-255)
        scale: Scaling factor for Poisson distribution
            - Higher scale = more photons = less relative noise
            - scale=1000: moderate photon count
            - scale=500: low photon count (high noise)
            - scale=2000: high photon count (low noise)
        
    Returns:
        Noisy image as 2D numpy array (0-255)
        
    Note: Poisson noise is signal-dependent. Brighter regions have 
    more noise (higher variance).
    """
    # Scale image to simulate photon counts
    scaled = image.astype(np.float64) * scale / 255.0
    
    # Generate Poisson noise
    noisy = np.random.poisson(scaled).astype(np.float64)
    
    # Scale back to 0-255 range
    noisy = noisy * 255.0 / scale
    
    return np.clip(noisy, 0, 255).astype(np.uint8)
```

**Mathematical Formulation:**
```
I_noisy(x,y) ~ Poisson(I_clean(x,y) * scale / 255) * 255 / scale
```

**FFT Effect:** Signal-dependent noise, higher variance in bright regions.

### 3. Defocus Blur

Simulates **out-of-focus imaging** caused by incorrect focusing or sample thickness:

```python
def apply_gaussian_blur(image: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """
    Apply Gaussian blur simulating defocus.
    
    Args:
        image: Input image as 2D numpy array (0-255)
        sigma: Blur standard deviation in pixels
            - sigma=1: mild defocus
            - sigma=2: moderate defocus
            - sigma=4: severe defocus (used in main evaluation)
            - sigma=8: extreme defocus
        
    Returns:
        Blurred image as 2D numpy array (0-255)
        
    Effect on FFT: Attenuates high frequencies. The optical transfer 
    function (OTF) is a Gaussian: H(u,v) = exp(-2π²σ²(u² + v²)).
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(image.astype(np.float64), sigma=sigma).astype(np.uint8)
```

**Mathematical Formulation:**
```
I_blur(x,y) = (I_clean * G_σ)(x,y)
where G_σ is a Gaussian kernel with standard deviation σ
```

**FFT Effect:** $F_{blur}(u,v) = F_{clean}(u,v) \cdot \exp(-2\pi^2\sigma^2(u^2 + v^2))$ (exponential high-frequency attenuation).

### 4. Illumination Shading

Simulates **uneven illumination** from light source or condenser misalignment:

```python
def apply_illumination_shading(image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Apply multiplicative illumination gradient.
    
    Args:
        image: Input image as 2D numpy array (0-255)
        alpha: Shading strength (0-1)
            - alpha=0.0: no shading
            - alpha=0.3: mild shading
            - alpha=0.5: moderate shading (used in main evaluation)
            - alpha=0.7: severe shading
            - alpha=1.0: maximum shading
        
    Returns:
        Shaded image as 2D numpy array (0-255)
        
    Effect on FFT: Adds strong low-frequency components (DC offset 
    and low-frequency gradients).
    """
    H, W = image.shape
    
    # Create gradient (top-left to bottom-right)
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    X, Y = np.meshgrid(x, y)
    
    # Compute shading mask
    shading = 1.0 - alpha * (X + Y) / 2
    
    # Apply multiplicative shading
    shaded = image.astype(np.float64) * shading
    
    return np.clip(shaded, 0, 255).astype(np.uint8)
```

**Mathematical Formulation:**
```
I_shade(x,y) = I_clean(x,y) * (1 - α * (x/W + y/H) / 2)
```

**FFT Effect:** Strong DC component and low-frequency gradients.

### 5. Motion Blur

Simulates **sample or camera movement** during exposure:

```python
def apply_motion_blur(image: np.ndarray, kernel_size: int = 15, 
                      angle: float = 45.0) -> np.ndarray:
    """
    Apply motion blur along a specified direction.
    
    Args:
        image: Input image as 2D numpy array (0-255)
        kernel_size: Size of motion blur kernel (pixels)
            - kernel_size=5: short motion
            - kernel_size=15: moderate motion (used in main evaluation)
            - kernel_size=25: long motion
        angle: Direction of motion in degrees (0-180)
            - 0: horizontal (left-right)
            - 90: vertical (up-down)
            - 45: diagonal (used in main evaluation)
        
    Returns:
        Motion-blurred image as 2D numpy array (0-255)
        
    Effect on FFT: Adds directional attenuation, creating streaks 
    perpendicular to motion direction.
    """
    from scipy.ndimage import convolve
    
    # Create motion blur kernel
    rad = np.deg2rad(angle)
    dx = np.cos(rad)
    dy = np.sin(rad)
    
    # Create line kernel
    x = np.linspace(-kernel_size//2, kernel_size//2, kernel_size)
    kernel = np.zeros((kernel_size, kernel_size))
    
    for i in range(kernel_size):
        j = int(i * dx + (kernel_size//2) * (1 - abs(dx)))
        k = int(i * dy + (kernel_size//2) * (1 - abs(dy)))
        if 0 <= j < kernel_size and 0 <= k < kernel_size:
            kernel[k, j] = 1.0
    
    # Normalize kernel
    kernel = kernel / kernel.sum()
    
    # Apply convolution
    blurred = convolve(image.astype(np.float64), kernel, mode='reflect')
    
    return np.clip(blurred, 0, 255).astype(np.uint8)
```

**Mathematical Formulation:**
```
I_motion(x,y) = (I_clean * M_{α,ks})(x,y)
where M is a motion blur kernel of size ks at angle α
```

**FFT Effect:** Directional attenuation: $F_{motion}(u,v) = F_{clean}(u,v) \cdot \text{sinc}(\pi k_s (u\cos\alpha + v\sin\alpha))$.

### 6. JPEG Compression Artifacts

Simulates **lossy JPEG compression** artifacts:

```python
def apply_compression(image: np.ndarray, quality: int = 75) -> np.ndarray:
    """
    Apply JPEG compression to simulate compression artifacts.
    
    Args:
        image: Input image as 2D numpy array (0-255)
        quality: JPEG quality factor (0-100)
            - quality=95: nearly lossless
            - quality=75: moderate compression (used in main evaluation)
            - quality=50: high compression
            - quality=10: extreme compression
        
    Returns:
        Compressed image as 2D numpy array (0-255)
        
    Effect on FFT: Adds blocking artifacts (periodic patterns) and 
    high-frequency loss.
    """
    from PIL import Image
    import io
    
    # Convert to PIL Image
    img_pil = Image.fromarray(image.astype(np.uint8))
    
    # Compress to JPEG
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG', quality=quality, optimize=True)
    
    # Decompress
    buffer.seek(0)
    compressed = Image.open(buffer)
    
    return np.array(compressed.convert('L'), dtype=np.uint8)
```

**FFT Effect:** Blocking artifacts appear as periodic spikes in frequency domain.

## Degradation Parameter Presets

We define presets for different quality levels:

```python
# Degradation presets for different quality levels
DEGRADATION_PRESETS = {
    'HQ': {
        'gaussian_noise': {'sigma': 10},
        'poisson_noise': {'scale': 2000},
        'defocus_blur': {'sigma': 0.5},
        'illumination': {'alpha': 0.1},
        'motion_blur': {'kernel_size': 3, 'angle': 45},
        'compression': {'quality': 95}
    },
    'MQ': {
        'gaussian_noise': {'sigma': 25},
        'poisson_noise': {'scale': 1000},
        'defocus_blur': {'sigma': 2},
        'illumination': {'alpha': 0.3},
        'motion_blur': {'kernel_size': 10, 'angle': 45},
        'compression': {'quality': 75}
    },
    'LQ': {
        'gaussian_noise': {'sigma': 50},
        'poisson_noise': {'scale': 500},
        'defocus_blur': {'sigma': 4},
        'illumination': {'alpha': 0.5},
        'motion_blur': {'kernel_size': 15, 'angle': 45},
        'compression': {'quality': 50}
    }
}

# Combined degradation presets
COMBINED_PRESETS = {
    'combined_mild': {
        'sigma': 25,
        'blur_sigma': 1,
        'alpha': 0.3
    },
    'combined_moderate': {
        'sigma': 50,
        'blur_sigma': 2,
        'alpha': 0.5
    },
    'combined_severe': {
        'sigma': 75,
        'blur_sigma': 4,
        'alpha': 0.7
    }
}
```

## Paired Dataset Generation

```python
def generate_paired_dataset(hq_dir: str, output_dir: str, 
                            degradation_types: List[str] = None,
                            n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate paired dataset with synthetic degradations.
    
    Args:
        hq_dir: Directory containing high-quality images
        output_dir: Directory to save degraded images
        degradation_types: List of degradation types to apply
        n_samples: Number of samples to generate per degradation type
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with metadata (image_path, hq_path, degradation_type, params)
    """
    import os
    import random
    from pathlib import Path
    
    # Default degradation types
    if degradation_types is None:
        degradation_types = [
            'gaussian_noise', 'poisson_noise', 'defocus_blur',
            'illumination', 'motion_blur', 'compression',
            'combined_mild', 'combined_moderate', 'combined_severe'
        ]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all HQ images
    hq_paths = sorted(Path(hq_dir).glob('*.tif')) + sorted(Path(hq_dir).glob('*.png'))
    
    if len(hq_paths) == 0:
        raise ValueError(f"No images found in {hq_dir}")
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate samples
    metadata = []
    
    for degradation_type in degradation_types:
        # Get parameters for this degradation
        if degradation_type in DEGRADATION_PRESETS:
            params = DEGRADATION_PRESETS[degradation_type]
        elif degradation_type in COMBINED_PRESETS:
            params = COMBINED_PRESETS[degradation_type]
        else:
            params = {}
        
        for i in range(n_samples):
            # Select random HQ image
            hq_idx = random.randint(0, len(hq_paths) - 1)
            hq_path = hq_paths[hq_idx]
            
            # Load image
            img = np.array(Image.open(hq_path).convert('L'), dtype=np.uint8)
            
            # Apply degradation
            lq_img = degrade_image(img, degradation_type, **params)
            
            # Save LQ image
            cell_line = Path(hq_path).stem.split('_')[0]
            lq_filename = f"{cell_line}_LQ_{degradation_type}_{i:04d}.tif"
            lq_path = Path(output_dir) / lq_filename
            Image.fromarray(lq_img).save(lq_path)
            
            # Store metadata
            metadata.append({
                'lq_path': str(lq_path),
                'hq_path': str(hq_path),
                'degradation_type': degradation_type,
                'params': params,
                'cell_line': cell_line,
                'quality_level': degradation_type.split('_')[0] if '_' in degradation_type else 'unknown'
            })
    
    # Create DataFrame
    df = pd.DataFrame(metadata)
    
    # Save metadata
    df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
    
    return df

def generate_train_test_split(hq_dir: str, output_dir: str,
                              test_size: float = 0.2,
                              seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate train/test split of paired dataset.
    
    Args:
        hq_dir: Directory containing high-quality images
        output_dir: Directory to save degraded images and metadata
        test_size: Fraction of data for test set
        seed: Random seed
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Generate full dataset
    df = generate_paired_dataset(hq_dir, output_dir, seed=seed)
    
    # Stratified split by degradation type
    from sklearn.model_selection import train_test_split
    
    train_idx, test_idx = train_test_split(
        range(len(df)),
        test_size=test_size,
        stratify=df['degradation_type'],
        random_state=seed
    )
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    return train_df, test_df
```

## Visualization of Degradations

```python
def visualize_degradations(hq_image: np.ndarray, 
                           degradation_types: List[str] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize the effect of different degradations on an image.
    
    Args:
        hq_image: High-quality input image
        degradation_types: List of degradation types to visualize
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    plt.style.use('seaborn-v0_8-paper')
    
    if degradation_types is None:
        degradation_types = [
            'gaussian_noise', 'poisson_noise', 'defocus_blur',
            'illumination', 'motion_blur', 'compression'
        ]
    
    n_degradations = len(degradation_types)
    n_cols = 3
    n_rows = (n_degradations + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(15, 5 * n_rows))
    
    # Original image
    ax = plt.subplot(n_rows, n_cols, 1)
    ax.imshow(hq_image, cmap='gray', vmin=0, vmax=255)
    ax.set_title('High-Quality\nOriginal')
    ax.axis('off')
    
    # Degraded images
    for idx, deg_type in enumerate(degradation_types):
        ax = plt.subplot(n_rows, n_cols, idx + 2)
        
        # Get default parameters
        if deg_type in DEGRADATION_PRESETS:
            params = DEGRADATION_PRESETS['MQ'][deg_type]
        elif deg_type in COMBINED_PRESETS:
            params = COMBINED_PRESETS['combined_moderate']
        else:
            params = {}
        
        # Apply degradation
        lq_img = degrade_image(hq_image, deg_type, **params)
        
        ax.imshow(lq_img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(deg_type.replace('_', ' ').title())
        ax.axis('off')
    
    plt.suptitle('Synthetic Degradations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def visualize_fft_degradations(hq_image: np.ndarray,
                               degradation_types: List[str] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize FFT power spectra of degraded images.
    
    Args:
        hq_image: High-quality input image
        degradation_types: List of degradation types to visualize
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    from tutorials.t01_fft_feature_extraction import compute_fft2
    
    plt.style.use('seaborn-v0_8-paper')
    
    if degradation_types is None:
        degradation_types = ['gaussian_noise', 'defocus_blur', 'illumination']
    
    n_degradations = len(degradation_types)
    fig = plt.figure(figsize=(15, 5 * n_degradations))
    
    for idx, deg_type in enumerate(degradation_types):
        # Original FFT
        ax1 = plt.subplot(n_degradations, 2, 2*idx + 1)
        img_centered = hq_image.astype(float) - hq_image.mean()
        M, N = img_centered.shape
        w_m = 0.5 * (1 - np.cos(2 * np.pi * np.arange(M) / M))
        w_n = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / N))
        window = np.outer(w_m, w_n)
        img_windowed = img_centered * window
        F_shift, P = compute_fft2(img_windowed)
        ax1.imshow(np.log1p(P), cmap='hot')
        ax1.set_title(f'Original FFT\n{deg_type}')
        ax1.axis('off')
        
        # Degraded FFT
        ax2 = plt.subplot(n_degradations, 2, 2*idx + 2)
        if deg_type in DEGRADATION_PRESETS:
            params = DEGRADATION_PRESETS['MQ'][deg_type]
        else:
            params = {}
        lq_img = degrade_image(hq_image, deg_type, **params)
        img_centered = lq_img.astype(float) - lq_img.mean()
        img_windowed = img_centered * window
        F_shift, P = compute_fft2(img_windowed)
        ax2.imshow(np.log1p(P), cmap='hot')
        ax2.set_title(f'Degraded FFT\n{deg_type}')
        ax2.axis('off')
    
    plt.suptitle('FFT Power Spectra: Original vs Degraded', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

# Example usage
# hq_image = np.array(Image.open('data/livecell/A172_Phase_A7_1_00d04h00m_1.tif').convert('L'))
# visualize_degradations(hq_image, save_path='tutorials/figures/t06_degradations.png')
# visualize_fft_degradations(hq_image, save_path='tutorials/figures/t06_fft_degradations.png')
```

## Usage Example

```python
# Example 1: Apply single degradation
from PIL import Image

# Load HQ image
hq_path = 'data/livecell/A172_Phase_A7_1_00d04h00m_1.tif'
hq_image = np.array(Image.open(hq_path).convert('L'))

# Apply Gaussian noise
noisy = add_gaussian_noise(hq_image, sigma=50)
Image.fromarray(noisy).save('output/noisy.png')

# Apply defocus blur
blurred = apply_gaussian_blur(hq_image, sigma=4)
Image.fromarray(blurred).save('output/blurred.png')

# Apply combined degradation
combined = degrade_image(hq_image, 'combined_severe')
Image.fromarray(combined).save('output/combined.png')

# Example 2: Generate paired dataset
train_df, test_df = generate_train_test_split(
    hq_dir='data/livecell/',
    output_dir='data/livecell_synthetic/',
    test_size=0.2,
    seed=42
)

print(f"Generated {len(train_df)} training samples and {len(test_df)} test samples")

# Example 3: Analyze degradation effect on FFT features
from tutorials.t01_fft_feature_extraction import extract_features

# Extract features from HQ and LQ images
features_hq = extract_features(hq_path)

# Generate LQ version
lq_image = degrade_image(hq_image, 'combined_moderate')
lq_path = 'output/temp_lq.png'
Image.fromarray(lq_image).save(lq_path)
features_lq = extract_features(lq_path)

# Compare features
print("Feature comparison (HQ vs LQ):")
print(f"  Centroid: {features_hq[86]:.2f} vs {features_lq[86]:.2f}")
print(f"  High-freq ratio: {features_hq[92]:.4f} vs {features_lq[92]:.4f}")
print(f"  Isotropy: {features_hq[93]:.4f} vs {features_lq[93]:.4f}")
```

## Degradation Effect on FFT Features

| Degradation Type | Total Power | Centroid | Bandwidth | High-Freq Ratio | Isotropy |
|------------------|-------------|----------|-----------|-----------------|----------|
| Original (HQ) | High | Medium | Medium | Low | High |
| Gaussian Noise | **↑↑** | → | → | **↑↑** | → |
| Defocus Blur | **↓** | **↓↓** | **↓** | **↓↓** | → |
| Illumination | **↓** | **↓** | **↓** | **↑** | → |
| Motion Blur | **↓** | **↓** | **↓** | **↓** | **↓** |
| Compression | **↓** | **↓** | **↓** | **↑** | **↓** |

**Key:** ↑ = increase, ↓ = decrease, → = no change, ↑↑ = large increase, ↓↓ = large decrease

## Performance Evaluation with Synthetic Data

### Training Enhancement Models

```python
# Example: Train U-Net on synthetic data
import torch
from tutorials.t04_unet_segmentation import UNet, LIVECellDataset, train_unet

# Create synthetic dataset
train_df, test_df = generate_train_test_split(
    hq_dir='data/livecell/',
    output_dir='data/livecell_synthetic/',
    test_size=0.2,
    seed=42
)

# Create PyTorch dataset
class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        lq = np.array(Image.open(row['lq_path']).convert('L'), dtype=np.float32)
        hq = np.array(Image.open(row['hq_path']).convert('L'), dtype=np.float32)
        lq = lq / 255.0
        hq = hq / 255.0
        return torch.from_numpy(lq).unsqueeze(0), torch.from_numpy(hq).unsqueeze(0)

# Create dataloaders
train_dataset = SyntheticDataset(train_df)
test_dataset = SyntheticDataset(test_df)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
train_unet(model, train_loader, test_loader, epochs=50, device=device)

# Evaluate
test_loss = 0.0
model.eval()
with torch.no_grad():
    for lq, hq in test_loader:
        lq, hq = lq.to(device), hq.to(device)
        pred = model(lq)
        loss = F.mse_loss(pred, hq)
        test_loss += loss.item()

print(f"Test MSE: {test_loss / len(test_loader):.6f}")
```

### Evaluating Filter Performance

```python
def evaluate_filter_performance(filter_func: callable, filter_params: dict,
                                 test_df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate filter performance on synthetic test data.
    
    Args:
        filter_func: Filter function to evaluate
        filter_params: Parameters for the filter
        test_df: Test DataFrame from generate_paired_dataset
        
    Returns:
        Dictionary with performance metrics
    """
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    
    psnrs = []
    ssims = []
    
    for _, row in test_df.iterrows():
        # Load images
        hq = np.array(Image.open(row['hq_path']).convert('L'), dtype=np.float64)
        lq = np.array(Image.open(row['lq_path']).convert('L'), dtype=np.float64)
        
        # Apply filter
        filtered = apply_filter(lq, filter_func, **filter_params)
        
        # Compute metrics
        psnr = peak_signal_noise_ratio(hq, filtered, data_range=255)
        ssim = structural_similarity(hq, filtered, data_range=255)
        
        psnrs.append(psnr)
        ssims.append(ssim)
    
    return {
        'mean_psnr': np.mean(psnrs),
        'std_psnr': np.std(psnrs),
        'mean_ssim': np.mean(ssims),
        'std_ssim': np.std(ssims)
    }

# Example usage
from tutorials.t02_bandpass_filters import butterworth_bandpass

# Evaluate Butterworth filter
results = evaluate_filter_performance(
    butterworth_bandpass,
    {'r_low': 0.02, 'r_high': 0.3, 'order': 2},
    test_df
)

print(f"Butterworth Filter Performance:")
print(f"  PSNR: {results['mean_psnr']:.2f} ± {results['std_psnr']:.2f}")
print(f"  SSIM: {results['mean_ssim']:.4f} ± {results['std_ssim']:.4f}")
```

## Key Implementation Details

1. **Normalization**: All degradations are applied to images in the 0-255 range, with clipping to prevent overflow/underflow.

2. **Parameter Ranges**: Degradation parameters were chosen based on analysis of real microscopy images to ensure realism.

3. **Reproducibility**: Random seeds are set for all random operations to ensure reproducible results.

4. **Memory Efficiency**: Images are processed one at a time to minimize memory usage.

5. **Metadata Tracking**: All generated samples are tracked with metadata for traceability.

## Exercises

### Beginner
1. Apply a single degradation type to a sample image and visualize the result
2. Generate a paired dataset with 10 samples for each degradation type
3. Visualize the FFT power spectrum of an image before and after degradation

### Intermediate
1. Implement a new degradation type (e.g., salt-and-pepper noise)
2. Compare the effect of different degradation types on FFT features
3. Evaluate a filter's performance on synthetic data

### Advanced
1. Train an enhancement model on synthetic data and test on real data
2. Implement a GAN to generate more realistic degradations
3. Create a domain adaptation approach for synthetic-to-real transfer

## Frequently Asked Questions

**Q: How realistic are synthetic degradations?**
A: Synthetic degradations are simplified models of real-world degradations. They capture the main effects (noise, blur, shading) but may not perfectly match real microscopy degradations. However, they are valuable for systematic evaluation and controlled experiments.

**Q: Why use synthetic data instead of real data?**
A: Synthetic data provides: (1) Known ground truth for quantitative evaluation, (2) Controlled degradation parameters for systematic analysis, (3) Unlimited data for training deep learning models, (4) Ability to test edge cases and rare degradations.

**Q: Can I combine multiple degradation types?**
A: Yes! The `degrade_image` function supports combined degradations. You can also create custom combinations by applying multiple degradation functions sequentially.

**Q: How do I choose degradation parameters?**
A: Start with the presets defined in `DEGRADATION_PRESETS`. For realistic evaluations, analyze your real data to determine appropriate parameter ranges. Use the MQ (moderate) preset for most experiments.

**Q: Why does defocus blur affect high frequencies?**
A: Defocus blur is a low-pass filter in the frequency domain. It attenuates high spatial frequencies, which correspond to fine details and edges in the image.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Degradation too strong | Parameters too high | Reduce sigma, alpha, or kernel_size |
| Degradation not visible | Parameters too low | Increase sigma, alpha, or kernel_size |
| Memory error | Too many large images | Process images one at a time, use smaller images |
| Slow performance | Large images or many samples | Use smaller batch sizes, process in parallel |
| Artifacts in degraded image | Incorrect parameter range | Check parameter values, ensure proper clipping |

## References

- Gonzalez, R.C. & Woods, R.E. (2018). Digital Image Processing, 4th ed. Pearson. (Chapter 5: Image Restoration)
- Burger, W. & Burge, M.J. (2009). Digital Image Processing: An Algorithmic Approach. Springer. (Chapter 5: Image Restoration)
- Pratt, W.K. (2007). Digital Image Processing, 4th ed. Wiley. (Chapter 7: Image Restoration and Reconstruction)
- Russ, J.C. (2016). The Image Processing Handbook, 7th ed. CRC Press. (Chapter 7: Image Enhancement)

## How to Cite

If you use the synthetic degradation pipeline in your research, please cite:

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
- `src/common.py` - Degradation utility functions
- `src/gen_synthetic_data.py` - Dataset generation scripts
- `tutorials/t06_synthetic_degradation.py` - Tutorial code

**Previous:** [Tutorial 5: Adaptive Filter Selection](05_adaptive_filter_selection.md) | **Next:** [Tutorial 7: Evaluation Metrics](07_evaluation_metrics.md)

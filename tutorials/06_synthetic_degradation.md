# Tutorial 6: Synthetic Degradation Pipeline for Microscopy Images

## Overview

This tutorial documents the synthetic degradation pipeline used to evaluate filter and enhancement model performance across controlled quality levels. The pipeline applies realistic degradations to high-quality phase-contrast microscopy images, creating paired training and evaluation data.

## Motivation

Real microscopy images exhibit quality variations from multiple sources:
- **Photon noise**: From low-light conditions or short exposure times
- **Defocus blur**: From imperfect focusing or sample thickness
- **Illumination non-uniformity**: From uneven light source or condenser misalignment
- **Combined degradations**: Real images often suffer from multiple simultaneous degradations

Systematic evaluation requires controlled, reproducible degradations with known ground truth.

## Degradation Model

The forward model of image degradation:

```python
def degrade_image(hq_image, degradation_type, **params):
    """
    Apply synthetic degradation to a high-quality image.
    
    Input: HQ image (clean, in-focus, uniform illumination)
    Output: LQ image (degraded)
    """
    lq = hq_image.copy()
    
    if degradation_type == 'gaussian_noise':
        lq = add_gaussian_noise(lq, sigma=params['sigma'])
    
    elif degradation_type == 'defocus_blur':
        lq = apply_gaussian_blur(lq, sigma=params['sigma'])
    
    elif degradation_type == 'illumination':
        lq = apply_illumination_shading(lq, alpha=params['alpha'])
    
    elif degradation_type == 'combined_mild':
        lq = add_gaussian_noise(lq, sigma=params['sigma'] / 2)
        lq = apply_gaussian_blur(lq, sigma=params['blur_sigma'] / 2)
        lq = apply_illumination_shading(lq, alpha=params['alpha'] / 2)
    
    return np.clip(lq, 0, 255).astype(np.uint8)
```

## Degradation Types

### 1. Gaussian Noise

Simulates photon-shot noise in low-light conditions:

```python
def add_gaussian_noise(image, sigma=50):
    """
    Add Gaussian noise simulating photon-shot noise.
    
    Parameters:
        sigma: Noise standard deviation (0-255 scale)
               sigma=10: mild noise
               sigma=25: moderate noise
               sigma=50: severe noise (used in main evaluation)
    """
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)
```

**Effect on FFT**: Adds flat (white) power across all frequencies, raising the noise floor uniformly in the power spectrum.

### 2. Defocus Blur

Simulates out-of-focus imaging:

```python
def apply_gaussian_blur(image, sigma=4):
    """
    Apply Gaussian blur simulating defocus.
    
    Parameters:
        sigma: Blur standard deviation in pixels
               sigma=1: mild defocus
               sigma=2: moderate defocus
               sigma=4: severe defocus (used in main evaluation)
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(image.astype(np.float64), sigma=sigma).astype(np.uint8)
```

**Effect on FFT**: Attenuates high frequencies. The optical transfer function (OTF) is a Gaussian: $H(u,v) = \exp(-2\pi^2\sigma^2(u^2 + v^2))$.

### 3. Illumination Shading

Simulates uneven illumination:

```python
def apply_illumination_shading(image, alpha=0.5):
    """
    Apply multiplicative illumination gradient.
    
    Parameters:
        alpha: Shading strength (0 = no shading, 1 = full gradient)
    """
    M, N = image.shape
    
    # Create smooth gradient (combination of linear and radial)
    y = np.linspace(-1, 1, M)
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    
    # Linear gradient + radial component
    gradient = 1 + alpha * (0.5 * X + 0.3 * (X**2 + Y**2))
    
    shaded = image.astype(np.float64) * gradient
    return np.clip(shaded, 0, 255).astype(np.uint8)
```

**Effect on FFT**: Adds power at very low frequencies (DC and near-DC components).

### 4. Combined Mild

Applies all three degradations at reduced strength:

```python
def apply_combined_mild(image, sigma_noise=25, sigma_blur=2, alpha=0.25):
    """Apply all three degradations at half strength."""
    degraded = add_gaussian_noise(image, sigma=sigma_noise)
    degraded = apply_gaussian_blur(degraded, sigma=sigma_blur)
    degraded = apply_illumination_shading(degraded, alpha=alpha)
    return degraded
```

## Complete Pipeline Implementation

```python
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter

class SyntheticDegradationPipeline:
    """
    Generate synthetic LQ images from HQ originals.
    
    Creates paired (HQ, LQ) datasets for training and evaluation
    of enhancement models and filter performance.
    """
    
    DEGRADATION_TYPES = [
        'gaussian_noise',
        'defocus_blur',
        'illumination_shading',
        'combined_mild'
    ]
    
    # Parameter values for each degradation type
    PARAM_VALUES = {
        'gaussian_noise': [10, 25, 50, 75],
        'defocus_blur': [1, 2, 4, 8],
        'illumination_shading': [0.3, 0.5, 0.7],
        'combined_mild': [(25, 2, 0.25)]  # (noise_sigma, blur_sigma, alpha)
    }
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
    
    def degrade(self, hq_image, degradation_type, **params):
        """Apply a single degradation type."""
        if degradation_type == 'gaussian_noise':
            return self._add_gaussian_noise(hq_image, params['sigma'])
        elif degradation_type == 'defocus_blur':
            return self._apply_blur(hq_image, params['sigma'])
        elif degradation_type == 'illumination_shading':
            return self._apply_shading(hq_image, params['alpha'])
        elif degradation_type == 'combined_mild':
            return self._apply_combined(hq_image, **params)
        else:
            raise ValueError(f"Unknown degradation: {degradation_type}")
    
    def generate_dataset(self, hq_images, output_dir):
        """
        Generate complete degraded dataset.
        
        For each HQ image, creates:
        - 4 noise levels
        - 4 blur levels
        - 3 shading levels
        - 1 combined mild
        
        Total: 12 LQ versions per HQ image
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, hq_path in enumerate(hq_images):
            hq = np.array(Image.open(hq_path).convert('L'))
            hq_name = Path(hq_path).stem
            
            for deg_type in self.DEGRADATION_TYPES:
                for params in self._param_combinations(deg_type):
                    lq = self.degrade(hq, deg_type, **params)
                    
                    # Save degraded image
                    param_str = '_'.join(f'{k}={v}' for k, v in params.items())
                    out_name = f"{hq_name}_{deg_type}_{param_str}.tif"
                    out_path = output_dir / out_name
                    Image.fromarray(lq).save(out_path)
                    
                    results.append({
                        'hq_path': hq_path,
                        'lq_path': str(out_path),
                        'degradation': deg_type,
                        'params': params
                    })
        
        return results
    
    def _add_gaussian_noise(self, img, sigma):
        noise = self.rng.normal(0, sigma, img.shape)
        return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    
    def _apply_blur(self, img, sigma):
        return gaussian_filter(img.astype(np.float64), sigma=sigma).astype(np.uint8)
    
    def _apply_shading(self, img, alpha):
        M, N = img.shape
        y = np.linspace(-1, 1, M)
        x = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(x, y)
        gradient = 1 + alpha * (0.5 * X + 0.3 * (X**2 + Y**2))
        return np.clip(img.astype(np.float64) * gradient, 0, 255).astype(np.uint8)
    
    def _apply_combined(self, img, sigma_noise=25, sigma_blur=2, alpha=0.25):
        degraded = self._add_gaussian_noise(img, sigma_noise)
        degraded = self._apply_blur(degraded, sigma_blur)
        degraded = self._apply_shading(degraded, alpha)
        return degraded
    
    def _param_combinations(self, degradation_type):
        """Generate parameter combinations for a degradation type."""
        param_names = {
            'gaussian_noise': ['sigma'],
            'defocus_blur': ['sigma'],
            'illumination_shading': ['alpha'],
            'combined_mild': ['sigma_noise', 'sigma_blur', 'alpha']
        }
        
        values = self.PARAM_VALUES[degradation_type]
        
        if degradation_type == 'combined_mild':
            return [dict(zip(param_names[degradation_type], v)) for v in values]
        else:
            return [dict(zip(param_names[degradation_type], [v])) for v in values]
```

## Usage

```python
# Initialize pipeline
pipeline = SyntheticDegradationPipeline(seed=42)

# Generate degraded dataset
hq_images = sorted(Path('data/livecell/').glob('*.tif'))
results = pipeline.generate_dataset(hq_images, 'data/synthetic_lq/')

print(f"Generated {len(results)} degraded images")
print(f"From {len(hq_images)} HQ images")

# Apply single degradation
hq = np.array(Image.open('image.tif').convert('L'))
lq_noise = pipeline.degrade(hq, 'gaussian_noise', sigma=50)
lq_blur = pipeline.degrade(hq, 'defocus_blur', sigma=4)
lq_shading = pipeline.degrade(hq, 'illumination_shading', alpha=0.5)
```

## Dataset Statistics

| Degradation | Parameter Values | Images per HQ | Total (803 HQ) |
|-------------|-----------------|---------------|-----------------|
| Gaussian noise | σ = 10, 25, 50, 75 | 4 | 3,212 |
| Defocus blur | σ = 1, 2, 4, 8 | 4 | 3,212 |
| Illumination | α = 0.3, 0.5, 0.7 | 3 | 2,409 |
| Combined mild | (25, 2, 0.25) | 1 | 803 |
| **Total** | | **12** | **9,636** |

Plus 19,200 real LQ images from BBBC005 for cross-dataset validation.

## Validation Against Real Degradations

To validate that synthetic degradations match real quality issues, we compared:
1. **FFT power spectra**: Synthetic noise adds flat power; real noise shows frequency-dependent structure
2. **Filter performance curves**: Similar degradation-performance relationships
3. **BBBC005 comparison**: Synthetic blur matches real defocus blur for σ < 15

Key finding: Synthetic degradations are representative of real quality issues for moderate degradation levels, but may not capture complex real-world artifacts at severe levels.

## Source Code

- Pipeline: `src/synthesize_low_quality.py`
- Dataset summary: `docs/DATASET_SUMMARY.md`
- BBBC005 analysis: `src/ws2_bbbc005.py`

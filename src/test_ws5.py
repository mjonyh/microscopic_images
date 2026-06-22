#!/usr/bin/env python3
"""Test WS5 model loading and pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ws5_gradio import load_models, process_pipeline
import numpy as np

print("Loading models...")
models = load_models(device="cpu")
print("Models loaded OK")

# Create a test image
test_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

print("\nTest 1: No enhancement, DoG filter, Otsu segmentation")
out_img, out_mask, info, fft = process_pipeline(
    test_img, "DoG", 0.02, 0.3, 2, "None", False, False, models
)
print(f"  Output: {out_img.shape}, Mask: {out_mask.shape}")
print(f"  Info: {info}")

print("\nTest 2: DeBCR enhancement, Butterworth filter, Otsu, FFT")
out_img, out_mask, info, fft = process_pipeline(
    test_img, "Butterworth", 0.01, 0.25, 2, "DeBCR", True, False, models
)
print(f"  Output: {out_img.shape}, Mask: {out_mask.shape}, FFT: {fft.shape if fft is not None else None}")
print(f"  Info: {info}")

print("\nTest 3: PI-DDPM enhancement, no filter, U-Net segmentation")
out_img, out_mask, info, fft = process_pipeline(
    test_img, "None", 0.02, 0.3, 2, "PI-DDPM", False, True, models
)
print(f"  Output: {out_img.shape}, Mask: {out_mask.shape}")
print(f"  Info: {info}")

print("\nAll tests passed!")

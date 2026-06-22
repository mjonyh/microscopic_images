#!/usr/bin/env python3
"""Quick test: just load models and process one image."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("Importing...", flush=True)
from phaseA_physics_models import DeBCRInspired, PIDDPMInspired
print("Physics models imported", flush=True)

# Test just DeBCR on a small image
model = DeBCRInspired(wavelet='db4', levels=3)
print("DeBCR created", flush=True)

test_img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
print("Running enhancement...", flush=True)
result = model.enhance(test_img)
print(f"Enhancement done: {result.shape}", flush=True)

# Now test full pipeline
from filters import apply_filter
from ws5_gradio import load_models, process_pipeline

print("\nLoading all models...", flush=True)
import torch
print(f"  PyTorch OK, CUDA: {torch.cuda.is_available()}", flush=True)

# Load UNet only (skip DeBCR in load_models for speed test)
from ws5_gradio import UNet
print("  Creating UNet...", flush=True)
unet = UNet()
print("  UNet created", flush=True)

print("\nAll OK", flush=True)

#!/usr/bin/env python3
"""Check filter and model imports for WS5."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from filters import apply_filter, FILTER_REGISTRY
print("Filters OK:", list(FILTER_REGISTRY.keys()))

from phaseA_physics_models import DeBCRInspired, PIDDPMInspired
print("Physics models OK")

from common import load_image, list_images, OUTPUT_DIR
print("Common OK")

# Check UNet
import torch
print(f"PyTorch OK, CUDA: {torch.cuda.is_available()}")

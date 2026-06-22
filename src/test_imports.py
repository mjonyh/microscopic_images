#!/usr/bin/env python3
"""Minimal import test."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

t = time.time()
print(f"t={time.time()-t:.1f}s: importing numpy", flush=True)
import numpy as np
print(f"t={time.time()-t:.1f}s: importing PIL", flush=True)
from PIL import Image
print(f"t={time.time()-t:.1f}s: importing scipy", flush=True)
from scipy import ndimage
print(f"t={time.time()-t:.1f}s: importing pywt", flush=True)
import pywt
print(f"t={time.time()-t:.1f}s: importing torch", flush=True)
import torch
print(f"t={time.time()-t:.1f}s: importing common", flush=True)
from common import load_image, list_images, load_annotations, OUTPUT_DIR
print(f"t={time.time()-t:.1f}s: importing filters", flush=True)
from filters import apply_filter, FILTER_REGISTRY
print(f"t={time.time()-t:.1f}s: importing phaseA", flush=True)
from phaseA_physics_models import DeBCRInspired, PIDDPMInspired
print(f"t={time.time()-t:.1f}s: ALL DONE", flush=True)

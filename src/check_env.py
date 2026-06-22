#!/usr/bin/env python3
"""Check environment for WS5."""
import importlib
import sys

packages = ["gradio", "torch", "numpy", "PIL", "skimage", "scipy", "pandas", "matplotlib"]
for pkg in packages:
    try:
        m = importlib.import_module(pkg)
        ver = getattr(m, "__version__", "?")
        print(f"  {pkg}: {ver}")
    except ImportError:
        print(f"  {pkg}: NOT INSTALLED")

# Check GPU
try:
    import torch
    print(f"\n  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except:
    pass

# Check model files
from pathlib import Path
models_dir = Path("outputs")
print("\n  Model files:")
for f in models_dir.glob("*.pth"):
    size = f.stat().st_size / (1024*1024)
    print(f"    {f.name}: {size:.1f} MB")

# GPU Utilization Plan

## Hardware Available
- **GPU 0**: NVIDIA GeForce GTX 1080 Ti (11 GB VRAM)
- **GPU 1**: NVIDIA GeForce GTX 1080 Ti (11 GB VRAM)
- **CUDA**: 13.0
- **PyTorch**: 2.6.0+cu124

## GPU-Accelerated Workstreams

### WS3: U-Net Segmentation (GPU 0)
- Training: 5-fold CV on 808 images
- Batch size: 4 (fits in 11 GB VRAM)
- Estimated time: ~30 min on GPU (vs ~3 hours on CPU)
- Model: ~50M parameters

### WS1: Enhancement Models (GPU 1)
- DeBCR: Wavelet + CNN (CPU-based, no GPU needed)
- PI-DDPM: Iterative refinement (CPU-based)
- N2V: Self-supervised (CPU-based)
- CARE: If TensorFlow available, can use GPU

### WS7: Adaptive Enhancement (CPU)
- Grid search over parameters
- Can be parallelized across cell lines using MPI

## Parallel Execution Strategy

```
GPU 0: WS3 U-Net training (5-fold CV)
GPU 1: WS1 CARE (if TF available) or idle
CPU:   WS1 DeBCR/N2V/PI-DDPM, WS7 adaptive, WS2 BBBC005
MPI:   WS4-7 can run in parallel when WS1-3 complete
```

## Expected Speedups
| Workstream | CPU Time | GPU Time | Speedup |
|------------|----------|----------|---------|
| WS3 U-Net | ~3 hours | ~30 min | 6× |
| WS1 CARE | N/A | ~10 min | — |
| WS7 Adaptive | ~2 hours | ~2 hours (CPU) | 1× |
| WS2 BBBC005 | ~3 hours | ~3 hours (CPU) | 1× |

## Memory Budget
- U-Net model: ~200 MB
- Training batch (4 images): ~50 MB
- Total GPU memory: ~300 MB out of 11 GB (3%)
- Plenty of room for larger batches or models

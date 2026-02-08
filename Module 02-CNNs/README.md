# Module 02: Convolutional Networks

> Build CNN primitives from scratch, then train a CIFAR-10 classifier.

---

## Overview

This module focuses on spatial modeling and efficient convolution implementations:
- Naive sliding-window convolutions
- Im2Col and vectorized kernels
- Pooling and stride logic
- Residual blocks and modern convolution variants
- Normalization alternatives for small-batch training

---

## Topics

| Topic | Name | Description | Duration |
|-------|------|-------------|----------|
| 11 | Conv2D Sliding Window | Naive conv with padding/stride | 2-3 hrs |
| 12 | Im2Col Vectorization | Convert conv into matrix multiplication | 2-3 hrs |
| 13 | Pooling & Strides | Max/avg pooling and backward masks | 2-3 hrs |
| 14 | ResNet Skip Connections | Residual blocks and projection shortcuts | 2-3 hrs |
| 15 | Modern Convolutions | Depthwise, pointwise, dilated convs | 2-3 hrs |
| 16 | Advanced Normalizations | LayerNorm, GroupNorm, InstanceNorm | 2-3 hrs |
| 17 | CIFAR-10 From Scratch | End-to-end image training milestone | 3-4 hrs |

---

## Learning Objectives

After this module, you should be able to:
1. Implement and debug Conv2D kernels in pure NumPy.
2. Explain accuracy/speed/memory tradeoffs between naive and im2col variants.
3. Implement residual blocks and normalization choices without framework helpers.
4. Train a small CNN on CIFAR-10 with reproducible preprocessing.

---

## Prerequisites

- Module 00 and Module 01 complete.
- Comfortable with tensor shapes `(N, C, H, W)`.
- Basic understanding of image channels and kernels.

---

## Directory Structure

```
Module 02-CNNs/
├── README.md
├── Topic 11-Conv2D-Sliding-Window/
├── Topic 12-Im2Col-Vectorization/
├── Topic 13-Pooling-Strides/
├── Topic 14-ResNet-Skip-Connections/
├── Topic 15-Modern-Convolutions/
├── Topic 16-Advanced-Normalizations/
└── Topic 17-CIFAR10-From-Scratch/
```

---

## Milestone

By Topic 17, you should have a full NumPy CNN training pipeline over CIFAR-10 data.

---

*Module 02 is complete when Topics 11-17 each have runnable starter code, tests, and hints.*

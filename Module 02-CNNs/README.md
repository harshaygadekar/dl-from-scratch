# Module 02: Convolutional Neural Networks

> Master spatial pattern recognition with CNNs.

---

## 📋 Overview

This module covers convolutional neural networks, the backbone of computer vision:
- Convolution operations
- Pooling layers
- Building complete CNN architectures
- Batch normalization

---

## 📚 Topics

| Topic | Name | Description | Duration |
|-------|------|-------------|----------|
| 09 | Convolution | im2col, convolution forward/backward | 3-4 hrs |
| 10 | Pooling | Max pool, average pool, strided | 2-3 hrs |
| 11 | CNN Architecture | LeNet-style networks | 3-4 hrs |
| 12 | Batch Normalization | Normalize activations | 2-3 hrs |
| 13 | Dropout | Regularization technique | 2-3 hrs |

---

## 🎯 Learning Objectives

After completing this module, you will:
1. Understand how convolutions extract spatial features
2. Implement efficient convolution using im2col
3. Build complete CNN architectures from scratch
4. Apply normalization and regularization techniques

---

## 🔧 Prerequisites

- ✅ Module 01: Neural Network Core
- ✅ Understanding of image representation
- ✅ Matrix operations and reshaping

---

## 📈 Difficulty Progression

```
Topic 09 (Conv)     ███████░░░ Hard
Topic 10 (Pool)     █████░░░░░ Medium
Topic 11 (CNN)      ██████░░░░ Medium-Hard
Topic 12 (BatchNorm)███████░░░ Hard
Topic 13 (Dropout)  ████░░░░░░ Easy-Medium
```

---

## ⏱️ Estimated Time

**Total**: 13-17 hours

---

## 🗂️ Directory Structure

```
Module 02-CNNs/
├── README.md           # This file
├── Topic 09-Convolution/
├── Topic 10-Pooling/
├── Topic 11-CNN-Architecture/
├── Topic 12-Batch-Normalization/
└── Topic 13-Dropout/
```

---

## 🏆 Module Milestone

By the end of this module, you should be able to:

```python
# Build a LeNet-style CNN
cnn = Sequential([
    Conv2d(1, 6, kernel_size=5, padding=2),
    ReLU(),
    MaxPool2d(2),
    Conv2d(6, 16, kernel_size=5),
    ReLU(),
    MaxPool2d(2),
    Flatten(),
    Linear(16 * 5 * 5, 120),
    ReLU(),
    Linear(120, 84),
    ReLU(),
    Linear(84, 10)
])

# Train on MNIST
for x, y in mnist_loader:
    logits = cnn(x)
    loss = cross_entropy(logits, y)
    # ... train
```

---

## 🔍 Key Interview Topics

- Why convolutions over fully connected layers?
- How does im2col work?
- What's the receptive field?
- Batch norm at train vs inference time

---

*"Convolutions see the world in patterns, not pixels."*

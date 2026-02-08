# Topic 08: Loss Functions

> **Goal**: Implement all common loss functions with gradients.
> **Time**: 2-3 hours | **Difficulty**: Medium

---

## 🎯 Learning Objectives

By the end of this topic, you will:
1. Implement MSE, MAE for regression
2. Implement Cross-Entropy for classification
3. Implement Binary Cross-Entropy
4. Understand when to use each loss

---

## 📋 Loss Functions Overview

| Name | Formula | Use Case |
|------|---------|----------|
| MSE | (y - ŷ)² | Regression |
| MAE | \|y - ŷ\| | Robust regression |
| Cross-Entropy | -Σ y log(p) | Multi-class |
| Binary CE | -y log(p) - (1-y)log(1-p) | Binary classification |
| Hinge | max(0, 1 - y·ŷ) | SVMs, margin-based |

---

## 📁 File Structure

```
Topic 08-Loss-Functions/
├── README.md
├── questions.md
├── intuition.md
├── math-refresh.md
├── hints/
│   ├── hint-1-mse-mae.md
│   ├── hint-2-cross-entropy.md
│   └── hint-3-binary-ce.md
├── solutions/
│   ├── level01_naive.py
│   ├── level02_vectorized.py
│   ├── level03_memory_efficient.py
│   └── level04_pytorch_reference.py
├── tests/
│   ├── test_basic.py
│   ├── test_edge.py
│   └── test_stress.py
└── visualization.py
```

---

## 🎮 Usage

```python
from losses import MSELoss, CrossEntropyLoss, BCELoss

# Regression
mse = MSELoss()
loss = mse.forward(predictions, targets)
grad = mse.backward()

# Multi-class classification
ce = CrossEntropyLoss()
loss = ce.forward(logits, one_hot_labels)
grad = ce.backward()

# Binary classification
bce = BCELoss()
loss = bce.forward(sigmoid_output, binary_labels)
grad = bce.backward()
```

---

## 🏆 Success Criteria

| Level | Requirement |
|-------|-------------|
| Level 1 | All forward passes work |
| Level 2 | All backward passes work |
| Level 3 | Numerically stable |
| Level 4 | Matches PyTorch losses |

---

## 🔗 Related Topics

- **Topic 06**: Backpropagation (uses loss gradients)
- **Topic 07**: Activation Functions (softmax + CE combined)

---

*"The loss function is your network's compass—it guides learning."*

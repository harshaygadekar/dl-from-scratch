# Topic 07: Activation Functions

> **Goal**: Implement all common activations with correct gradients.
> **Time**: 2-3 hours | **Difficulty**: Medium

---

## 🎯 Learning Objectives

By the end of this topic, you will:
1. Implement ReLU, LeakyReLU, ELU, GELU
2. Implement Sigmoid, Tanh, Softmax
3. Derive and implement gradients for each
4. Understand when to use each activation

---

## 📋 Activations Overview

| Name | Formula | Range | Use Case |
|------|---------|-------|----------|
| ReLU | max(0, x) | [0, ∞) | Hidden layers |
| LeakyReLU | max(αx, x) | (-∞, ∞) | Prevent dying ReLU |
| Sigmoid | 1/(1+e⁻ˣ) | (0, 1) | Binary output |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) | Zero-centered |
| Softmax | eˣⁱ/Σeˣʲ | (0, 1) | Multi-class |
| GELU | x·Φ(x) | (-0.17, ∞) | Transformers |

---

## 📁 File Structure

```
Topic 07-Activation-Functions/
├── README.md
├── questions.md
├── intuition.md
├── math-refresh.md
├── hints/
│   ├── hint-1-relu-family.md
│   ├── hint-2-sigmoid-tanh.md
│   └── hint-3-softmax.md
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
# All activations have forward and backward
from activations import ReLU, Sigmoid, Softmax

relu = ReLU()
y = relu.forward(x)
grad_x = relu.backward(grad_y)

# Softmax + CrossEntropy combined for efficiency
loss, grad = softmax_cross_entropy(logits, labels)
```

---

## 🏆 Success Criteria

| Level | Requirement |
|-------|-------------|
| Level 1 | All forward passes work |
| Level 2 | All backward passes work |
| Level 3 | Numerical gradient check passes |
| Level 4 | Matches PyTorch activations |

---

## 🔗 Related Topics

- **Topic 05**: MLP Forward Pass (uses activations)
- **Topic 06**: Backpropagation (gradient computation)
- **Topic 08**: Loss Functions (often combined with softmax)

---

*"The activation function is the 'spark' that makes neural networks non-linear."*

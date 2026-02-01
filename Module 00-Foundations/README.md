# Module 00: Foundations

> The survival module—master these before touching neural networks.

---

## 📋 Overview

This module covers the foundational building blocks you'll use throughout:
- Tensor operations and memory-efficient broadcasting
- Building an autograd engine from scratch
- Implementing optimization algorithms

**Without these fundamentals, the rest falls apart.**

---

## 📚 Topics

| Topic | Name | Description | Duration |
|-------|------|-------------|----------|
| 01 | Tensor Operations | Memory layouts, broadcasting, stride tricks | 2-3 hrs |
| 02 | Autograd Engine | Computational graphs, reverse-mode AD | 3-4 hrs |
| 03 | Optimizers | SGD, Momentum, Adam from equations | 2-3 hrs |

---

## 🎯 Learning Objectives

After completing this module, you will:
1. Understand how tensors are stored in memory
2. Master NumPy broadcasting rules
3. Build a working autograd engine with computational graphs
4. Implement SGD, Momentum, and Adam optimizers

---

## 🔧 Prerequisites

- ✅ Python proficiency (classes, decorators, generators)
- ✅ Linear algebra basics (vectors, matrices, dot products)
- ✅ Calculus I (derivatives, chain rule)

---

## 📈 Difficulty Progression

```
Topic 01 (Tensors)    ███░░░░░░░ Easy-Medium
Topic 02 (Autograd)   ██████░░░░ Medium-Hard
Topic 03 (Optimizers) ████░░░░░░ Medium
```

---

## ⏱️ Estimated Time

**Total**: 7-10 hours

---

## 🗂️ Directory Structure

```
Module 00-Foundations/
├── README.md
├── Topic 01-Tensor-Operations/
├── Topic 02-Autograd-Engine/
└── Topic 03-Optimizers/
```

---

## 🏆 Module Milestone

By the end of this module, you should be able to:

```python
# Create tensors with autograd support
from autograd import Tensor

x = Tensor([[1, 2], [3, 4]], requires_grad=True)
y = x @ x.T
z = y.sum()

# Compute gradients automatically
z.backward()
print(x.grad)  # Works!

# Use optimizers
from optimizers import Adam
optimizer = Adam([x], lr=0.01)
optimizer.step()
```

---

## 🚦 Getting Started

1. Start with [Topic 01: Tensor Operations](Topic%2001-Tensor-Operations/)
2. Master broadcasting before moving on
3. Topic 02 (Autograd) is the hardest—take your time!

---

*"If you can't implement autograd, you can't debug transformers."*

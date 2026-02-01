# Module 01: Neural Network Core

> Build foundational neural network components from scratch.

---

## 📋 Overview

This module covers the essential building blocks of neural networks:
- Linear layers and activation functions
- Loss functions
- Complete forward and backward passes
- A working MLP from scratch

---

## 📚 Topics

| Topic | Name | Description | Duration |
|-------|------|-------------|----------|
| 04 | Linear Layer | Weight matrices, biases, forward/backward | 2-3 hrs |
| 05 | Activations | ReLU, Sigmoid, Tanh, Softmax | 2-3 hrs |
| 06 | Loss Functions | MSE, Cross-Entropy, Binary CE | 2-3 hrs |
| 07 | MLP Assembly | Complete multi-layer perceptron | 3-4 hrs |
| 08 | Training Loop | Epochs, batches, validation | 2-3 hrs |

---

## 🎯 Learning Objectives

After completing this module, you will:
1. Understand how linear transformations work mathematically
2. Implement various activation functions with correct gradients
3. Build loss functions for regression and classification
4. Assemble a complete MLP that can learn

---

## 🔧 Prerequisites

- ✅ Module 00: Foundations (Tensor ops, Autograd, Optimizers)
- ✅ Understanding of matrix multiplication
- ✅ Basic calculus (chain rule)

---

## 📈 Difficulty Progression

```
Topic 04 (Linear)   ████░░░░░░ Medium
Topic 05 (Acts)     █████░░░░░ Medium
Topic 06 (Loss)     █████░░░░░ Medium
Topic 07 (MLP)      ██████░░░░ Medium-Hard
Topic 08 (Training) █████░░░░░ Medium
```

---

## ⏱️ Estimated Time

**Total**: 12-16 hours

---

## 🗂️ Directory Structure

```
Module 01-Neural-Network-Core/
├── README.md           # This file
├── Topic 04-Linear-Layer/
├── Topic 05-Activations/
├── Topic 06-Loss-Functions/
├── Topic 07-MLP-Assembly/
└── Topic 08-Training-Loop/
```

---

## 🏆 Module Milestone

By the end of this module, you should be able to:

```python
# Train a neural network on MNIST-like data
mlp = MLP(784, [256, 128, 10])
optimizer = Adam(mlp.parameters(), lr=0.001)

for epoch in range(10):
    for x_batch, y_batch in dataloader:
        logits = mlp(x_batch)
        loss = cross_entropy(logits, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}: Loss = {loss.data:.4f}")
```

---

*"The neural network is simple in concept but deep in possibilities."*

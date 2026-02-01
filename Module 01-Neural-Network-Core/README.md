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
| 04 | Single Layer Perceptron | Sigmoid, BCE loss, gradient descent | 2-3 hrs |
| 05 | MLP Forward Pass | Multi-layer networks, weight initialization | 2-3 hrs |
| 06 | Backpropagation | Chain rule, backward pass, gradients | 3-4 hrs |
| 07 | Activation Functions | ReLU, Sigmoid, Tanh, Softmax + gradients | 2-3 hrs |
| 08 | Loss Functions | MSE, Cross-Entropy, Binary CE | 2-3 hrs |
| 09 | Regularization | L2, Dropout, Batch Normalization | 2-3 hrs |
| 10 | End-to-End MNIST | Complete MLP, 95% accuracy target | 3-4 hrs |

---

## 🎯 Learning Objectives

After completing this module, you will:
1. Understand how linear transformations work mathematically
2. Implement various activation functions with correct gradients
3. Build loss functions for regression and classification
4. Assemble a complete MLP that can learn
5. Apply regularization techniques to prevent overfitting
6. Train a network to 95%+ accuracy on MNIST

---

## 🔧 Prerequisites

- ✅ Module 00: Foundations (Tensor ops, Autograd, Optimizers)
- ✅ Understanding of matrix multiplication
- ✅ Basic calculus (chain rule)

---

## 📈 Difficulty Progression

```
Topic 04 (Perceptron)   ████░░░░░░ Medium
Topic 05 (MLP)          █████░░░░░ Medium
Topic 06 (Backprop)     ██████░░░░ Medium-Hard
Topic 07 (Activations)  ████░░░░░░ Medium
Topic 08 (Losses)       ████░░░░░░ Medium
Topic 09 (Reg/BN)       █████░░░░░ Medium
Topic 10 (MNIST)        ██████░░░░ Medium-Hard
```

---

## ⏱️ Estimated Time

**Total**: 18-24 hours

---

## 🗂️ Directory Structure

```
Module 01-Neural-Network-Core/
├── README.md
├── Topic 04-Single-Layer-Perceptron/
├── Topic 05-MLP-Forward-Pass/
├── Topic 06-Backpropagation/
├── Topic 07-Activation-Functions/
├── Topic 08-Loss-Functions/
├── Topic 09-Regularization/
└── Topic 10-End-to-End-MNIST/
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

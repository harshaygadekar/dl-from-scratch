# Topic 10: End-to-End MNIST

> **Goal**: Train a complete MLP on MNIST to 95%+ accuracy.
> **Time**: 3-4 hours | **Difficulty**: Medium-Hard

---

## 🎯 Learning Objectives

By the end of this topic, you will:
1. Load and preprocess MNIST data
2. Build a complete MLP with all components from this module
3. Implement a training loop with batching and validation
4. Achieve 95%+ test accuracy

---

## 📋 The Challenge

Build an MLP using ONLY components from previous topics:
- Linear layers (Topic 05)
- Activations (Topic 07)
- Loss functions (Topic 08)
- Regularization (Topic 09)
- Optimizers (Module 00, Topic 03)

**Target**: 95%+ accuracy on MNIST test set.

---

## 📁 File Structure

```
Topic 10-End-to-End-MNIST/
├── README.md
├── questions.md
├── intuition.md
├── math-refresh.md
├── hints/
│   ├── hint-1-data-loading.md
│   ├── hint-2-model-architecture.md
│   └── hint-3-training-loop.md
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
from mnist_mlp import MLP, train, evaluate

# Build model
model = MLP(
    input_size=784,
    hidden_sizes=[256, 128],
    output_size=10,
    activation='relu',
    dropout=0.2
)

# Train
train(model, train_data, val_data, epochs=10, lr=0.001)

# Evaluate
accuracy = evaluate(model, test_data)
print(f"Test Accuracy: {accuracy:.2%}")
```

---

## 🏆 Success Criteria

| Level | Accuracy | Time |
|-------|----------|------|
| Level 1 | >90% | Any |
| Level 2 | >95% | <5 min |
| Level 3 | >97% | <2 min |
| Level 4 | Match PyTorch | Match |

---

## 🔗 Prerequisites

You must complete these topics first:
- Topic 05: MLP Forward Pass
- Topic 06: Backpropagation
- Topic 07: Activation Functions
- Topic 08: Loss Functions
- Topic 09: Regularization

---

*"This is your graduation project for Module 01—make it count!"*

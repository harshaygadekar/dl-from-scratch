# Topic 09: Regularization

> **Goal**: Implement L2 regularization, Dropout, and Batch Normalization.
> **Time**: 2-3 hours | **Difficulty**: Medium

---

## 🎯 Learning Objectives

By the end of this topic, you will:
1. Implement L2 (weight decay) regularization
2. Build Dropout with proper train/eval modes
3. Implement Batch Normalization with running statistics
4. Understand when and why to use each technique

---

## 📋 Techniques Overview

| Technique | Purpose | When to Use |
|-----------|---------|-------------|
| L2 Regularization | Prevent large weights | Always (as weight decay) |
| Dropout | Prevent co-adaptation | Hidden layers, large networks |
| Batch Normalization | Stabilize training | Deep networks, faster convergence |

---

## 📁 File Structure

```
Topic 09-Regularization/
├── README.md
├── questions.md
├── intuition.md
├── math-refresh.md
├── hints/
│   ├── hint-1-l2-regularization.md
│   ├── hint-2-dropout.md
│   └── hint-3-batch-norm.md
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
# L2 Regularization
loss = cross_entropy(logits, y) + 0.01 * l2_loss(model.parameters())

# Dropout
dropout = Dropout(p=0.5)
dropout.train()  # Enable dropout
h = dropout(h)
dropout.eval()   # Disable dropout

# Batch Normalization
bn = BatchNorm1d(num_features=64)
bn.train()  # Use batch stats
h = bn(h)
bn.eval()   # Use running stats
```

---

## 🏆 Success Criteria

| Level | Requirement |
|-------|-------------|
| Level 1 | L2, Dropout, BN forward pass |
| Level 2 | Correct backward passes |
| Level 3 | Train vs eval modes working |
| Level 4 | Matches PyTorch BatchNorm |

---

*"Regularization is the art of being just uncertain enough."*

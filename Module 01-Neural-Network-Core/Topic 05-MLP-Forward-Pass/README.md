# Topic 05: MLP Forward Pass

> **Goal**: Build a multi-layer perceptron with proper initialization.
> **Time**: 2-3 hours | **Difficulty**: Medium

---

## 🎯 Learning Objectives

By the end of this topic, you will:
1. Understand how layers stack to form deep networks
2. Implement Xavier and Kaiming initialization
3. Build forward pass through multiple layers
4. Understand activation function choices

---

## 📋 The Problem

Implement a Multi-Layer Perceptron (MLP) that can learn non-linear functions.

### Mathematical Model

```
Layer 1: h₁ = σ(W₁·x + b₁)
Layer 2: h₂ = σ(W₂·h₁ + b₂)
...
Output:  y = W_L·h_{L-1} + b_L
```

### Required Implementation

```python
class MLP:
    def __init__(self, layer_sizes, activation='relu'):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i+1]))
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = activation(layer(x))
        return self.layers[-1](x)  # No activation on output

class Linear:
    def __init__(self, in_features, out_features):
        self.W = initialize_weights(in_features, out_features)
        self.b = np.zeros(out_features)
```

---

## 🧠 Key Concepts

### 1. The Multi-Layer Architecture
```
Input    Hidden 1   Hidden 2   Output
 (3)       (4)        (4)       (2)

  x₁ ──┬──► h₁₁ ──┬──► h₂₁ ──┬──► y₁
       │         │         │
  x₂ ──┼──► h₁₂ ──┼──► h₂₂ ──┼──► y₂
       │         │         │
  x₃ ──┴──► h₁₃ ──┴──► h₂₃ ──┘
           ↓          ↓
          h₁₄        h₂₄
```

### 2. Weight Initialization Strategies
- **Xavier/Glorot**: Var(W) = 2/(n_in + n_out)
- **He/Kaiming**: Var(W) = 2/n_in (for ReLU)
- **LeCun**: Var(W) = 1/n_in

### 3. Activation Functions
- **ReLU**: max(0, x)
- **Sigmoid**: 1/(1 + e^(-x))
- **Tanh**: (e^x - e^(-x))/(e^x + e^(-x))

---

## 📁 File Structure

```
Topic 05-MLP-Forward-Pass/
├── README.md
├── questions.md
├── intuition.md
├── math-refresh.md
├── hints/
│   ├── hint-1-linear-layer.md
│   ├── hint-2-initialization.md
│   └── hint-3-activations.md
├── solutions/
│   ├── level01-naive.py
│   ├── level02-vectorized.py
│   ├── level03-memory-efficient.py
│   └── level04-pytorch-reference.py
├── tests/
│   ├── test_basic.py
│   ├── test_edge.py
│   └── test_stress.py
└── visualization.py
```

---

## 🎮 How to Use

### Step 1: Create an MLP
```python
# Network: 784 -> 256 -> 128 -> 10
mlp = MLP([784, 256, 128, 10], activation='relu')
```

### Step 2: Forward Pass
```python
# Single sample
x = np.random.randn(784)
output = mlp.forward(x)  # Shape: (10,)

# Batch
X = np.random.randn(32, 784)
outputs = mlp.forward(X)  # Shape: (32, 10)
```

---

## 🏆 Success Criteria

| Level | Requirement |
|-------|-------------|
| Level 1 | Single layer forward pass works |
| Level 2 | Multi-layer forward pass works |
| Level 3 | Xavier and Kaiming initialization |
| Level 4 | Matches PyTorch nn.Linear output |

---

## 🔗 Related Topics

- **Topic 04**: Single Layer Perceptron (foundation)
- **Topic 06**: Backpropagation (gradients for training)
- **Topic 07**: Activation Functions (deeper dive)

---

*"Depth enables learning hierarchical representations."*

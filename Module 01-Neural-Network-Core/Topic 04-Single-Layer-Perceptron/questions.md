# Topic 04: Interview Questions

Perceptron questions test your understanding of fundamentals!

---

## Q1: Derive the Sigmoid Gradient (Very Common)

**Difficulty**: Easy | **Time**: 3 min

Given σ(z) = 1/(1 + e^(-z)), derive dσ/dz.

<details>
<summary>Answer</summary>

```
σ(z) = 1/(1 + e^(-z)) = (1 + e^(-z))^(-1)

Using chain rule:
dσ/dz = -1 · (1 + e^(-z))^(-2) · (-e^(-z))
      = e^(-z) / (1 + e^(-z))²

Rewriting:
      = (1 + e^(-z) - 1) / (1 + e^(-z))²
      = 1/(1 + e^(-z)) - 1/(1 + e^(-z))²
      = σ(z) - σ(z)²
      = σ(z)(1 - σ(z))
```

**Key insight**: The sigmoid derivative is elegantly expressed in terms of itself:
```
σ'(z) = σ(z) · (1 - σ(z))
```

This makes computation efficient - just reuse the forward pass output!

</details>

---

## Q2: Binary Cross-Entropy Gradient (Common)

**Difficulty**: Medium | **Time**: 5 min

Derive ∂L/∂z where L = -[y·log(σ(z)) + (1-y)·log(1-σ(z))].

<details>
<summary>Answer</summary>

Using chain rule: ∂L/∂z = ∂L/∂σ · ∂σ/∂z

**Step 1**: Find ∂L/∂σ
```
L = -y·log(σ) - (1-y)·log(1-σ)
∂L/∂σ = -y/σ + (1-y)/(1-σ)
      = (-y(1-σ) + (1-y)σ) / (σ(1-σ))
      = (-y + yσ + σ - yσ) / (σ(1-σ))
      = (σ - y) / (σ(1-σ))
```

**Step 2**: Multiply by ∂σ/∂z = σ(1-σ)
```
∂L/∂z = (σ - y) / (σ(1-σ)) · σ(1-σ)
      = σ - y
      = ŷ - y
```

**Beautiful result**: The gradient is simply (prediction - target)!

This clean gradient is why sigmoid + BCE is used together.

</details>

---

## Q3: Why Not Use MSE for Classification? (Google)

**Difficulty**: Medium | **Time**: 5 min

What happens if you use Mean Squared Error instead of Cross-Entropy for binary classification?

<details>
<summary>Answer</summary>

**Problems with MSE for classification**:

1. **Saturated gradients**: When σ(z) ≈ 0 or 1, the sigmoid is flat. MSE gradient includes σ'(z) which is tiny, causing vanishing gradients.

2. **Non-convex loss surface**: MSE with sigmoid creates a non-convex optimization landscape with multiple local minima.

3. **Probabilistic interpretation**: Cross-entropy is the negative log-likelihood under a Bernoulli model. MSE has no such interpretation.

**Gradient comparison**:
```
MSE:  ∂L/∂z = (ŷ - y) · σ'(z) = (ŷ - y) · ŷ · (1 - ŷ)
BCE:  ∂L/∂z = (ŷ - y)
```

When ŷ ≈ 0 or ŷ ≈ 1:
- MSE gradient → 0 (vanishes!)
- BCE gradient → (ŷ - y) (stays meaningful)

**Example**: If y=1 and ŷ=0.01 (wrong prediction):
- MSE gradient: (0.01-1) × 0.01 × 0.99 ≈ -0.01 (tiny!)
- BCE gradient: 0.01 - 1 = -0.99 (strong signal!)

</details>

---

## Q4: Perceptron vs Logistic Regression (Basic)

**Difficulty**: Easy | **Time**: 3 min

What's the difference between a perceptron and logistic regression?

<details>
<summary>Answer</summary>

**Historical Perceptron** (Rosenblatt, 1958):
- Binary step activation: output = 1 if w·x + b > 0, else 0
- Perceptron learning rule: w += η(y - ŷ)x
- Can only learn linearly separable patterns
- No probabilistic interpretation

**Logistic Regression** (what we implement here):
- Sigmoid activation: output = σ(w·x + b) ∈ (0,1)
- Gradient descent with cross-entropy
- Outputs probabilities
- More gradual decision boundary

**Modern usage**: "Perceptron" often refers to a single neuron with sigmoid, which is really logistic regression. The distinction matters historically but less so in practice.

</details>

---

## Q5: XOR Problem (Famous)

**Difficulty**: Medium | **Time**: 5 min

Why can't a single-layer perceptron learn XOR? How would you fix it?

<details>
<summary>Answer</summary>

**The XOR truth table**:
```
x₁  x₂  y
0   0   0
0   1   1
1   0   1
1   1   0
```

**Why it fails**: XOR is not linearly separable. A single perceptron can only draw one straight line decision boundary.

**Visualization**:
```
x₂
│  1(0)      1(1)
│   ■         ●
│
│   ●         ■
│  0(1)      0(0)
└──────────────── x₁
```

No single line can separate ● from ■!

**Solution**: Add a hidden layer (Topic 05):
```python
# Two-layer network
h1 = σ(w1·x + b1)  # Hidden neuron 1
h2 = σ(w2·x + b2)  # Hidden neuron 2
y = σ(w3·[h1,h2] + b3)  # Output
```

The hidden layer learns to transform the space into one where XOR IS separable.

**Historical note**: The XOR problem was used by Minsky & Papert (1969) to argue against perceptrons, contributing to the first "AI winter". Multi-layer networks with backpropagation (1986) solved this.

</details>

---

## Q6: Implement from Scratch (Whiteboard)

**Difficulty**: Medium | **Time**: 10 min

Implement a perceptron with sigmoid activation. Include forward, loss computation, and gradient update.

<details>
<summary>Answer</summary>

```python
import numpy as np

class Perceptron:
    def __init__(self, input_dim):
        # Small random initialization
        self.w = np.random.randn(input_dim) * 0.01
        self.b = 0.0
        
        # For gradient accumulation
        self.dw = np.zeros(input_dim)
        self.db = 0.0
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def forward(self, x):
        z = np.dot(self.w, x) + self.b
        return self.sigmoid(z)
    
    def compute_loss(self, y_pred, y_true):
        # Binary cross-entropy with numerical stability
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def backward(self, x, y_true, y_pred):
        # Gradient of BCE + sigmoid = (y_pred - y_true)
        error = y_pred - y_true
        self.dw = error * x
        self.db = error
    
    def update(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db

# Training loop
def train(X, y, epochs=100, lr=0.1):
    model = Perceptron(X.shape[1])
    
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X)):
            y_pred = model.forward(X[i])
            loss = model.compute_loss(y_pred, y[i])
            model.backward(X[i], y[i], y_pred)
            model.update(lr)
            total_loss += loss
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss/len(X):.4f}")
    
    return model
```

</details>

---

## 🎯 Interview Tips

1. **Always clip sigmoid inputs**: Mention numerical stability
2. **Know the gradient formula**: ∂L/∂z = ŷ - y (for BCE + sigmoid)
3. **Explain XOR**: Shows you understand limitations
4. **Mention the bias term**: Don't forget it!

---

*"The perceptron is simple, but its lessons are profound."*

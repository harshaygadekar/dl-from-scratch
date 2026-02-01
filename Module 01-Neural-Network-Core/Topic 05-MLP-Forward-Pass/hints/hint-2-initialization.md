# Hint 2: Weight Initialization

Proper initialization prevents vanishing/exploding activations.

---

## Xavier (Glorot) Initialization

For tanh/sigmoid networks:
```python
def xavier_init(n_in, n_out):
    std = np.sqrt(2.0 / (n_in + n_out))
    return np.random.randn(n_in, n_out) * std
```

---

## Kaiming (He) Initialization

For ReLU networks:
```python
def kaiming_init(n_in, n_out):
    std = np.sqrt(2.0 / n_in)
    return np.random.randn(n_in, n_out) * std
```

---

## Bias Initialization

Always initialize biases to zero:
```python
b = np.zeros(n_out)
```

---

## When to Use What

| Activation | Initialization |
|------------|----------------|
| Sigmoid | Xavier |
| Tanh | Xavier |
| ReLU | Kaiming |
| LeakyReLU | Kaiming |

---

*Next: [Hint 3 - Activations](hint-3-activations.md)*

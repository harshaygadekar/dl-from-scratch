# Hint 3: Activation Functions

Apply non-linearity between layers.

---

## ReLU (Recommended)

```python
def relu(x):
    return np.maximum(0, x)
```

---

## Sigmoid

```python
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))
```

---

## Tanh

```python
def tanh(x):
    return np.tanh(x)
```

---

## Assembling the MLP

```python
class MLP:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
    
    def forward(self, x):
        h = x
        # Apply activation after each layer except the last
        for i, layer in enumerate(self.layers):
            h = layer.forward(h)
            if i < len(self.layers) - 1:
                h = relu(h)  # No activation on output
        return h
```

---

## Key Points

1. Apply activation between layers, not after the last layer
2. Output layer produces raw logits
3. Softmax (for classification) is applied during loss computation

---

*You now have all the pieces to build an MLP!*

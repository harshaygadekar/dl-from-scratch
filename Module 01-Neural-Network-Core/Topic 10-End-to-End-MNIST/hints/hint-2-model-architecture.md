# Hint 2: Model Architecture

---

## MLP Class

```python
class MLP:
    def __init__(self, sizes=[784, 256, 128, 10]):
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                self.layers.append(ReLU())
                self.layers.append(Dropout(0.2))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'W'):
                params.extend([layer.W, layer.b])
        return params
```

---

## Weight Initialization

```python
# Xavier/He initialization
W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
```

---

*Next: [Hint 3 - Training Loop](hint-3-training-loop.md)*

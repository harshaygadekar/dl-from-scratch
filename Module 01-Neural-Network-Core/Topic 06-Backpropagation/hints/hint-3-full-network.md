# Hint 3: Full Network Backward

Putting it all together for a complete network.

---

## Network Structure

```
x → Linear1 → ReLU → Linear2 → Softmax → Loss
```

---

## Forward Pass (with caching)

```python
class MLP:
    def forward(self, x, y_true):
        # Store activations for backward
        self.z1 = self.linear1.forward(x)
        self.a1 = self.relu.forward(self.z1)
        self.z2 = self.linear2.forward(self.a1)
        self.probs = softmax(self.z2)
        self.loss = cross_entropy(self.probs, y_true)
        return self.loss
```

---

## Backward Pass

```python
def backward(self, y_true):
    # Start from loss
    # Softmax + CE gradient
    grad = self.probs - y_true  # (batch, classes)
    
    # Linear2 backward
    grad = self.linear2.backward(grad)
    
    # ReLU backward
    grad = self.relu.backward(grad)
    
    # Linear1 backward
    grad = self.linear1.backward(grad)
    
    return grad  # Gradient w.r.t. input (optional)
```

---

## Update Weights

```python
def update(self, lr):
    self.linear1.W -= lr * self.linear1.grad_W
    self.linear1.b -= lr * self.linear1.grad_b
    self.linear2.W -= lr * self.linear2.grad_W
    self.linear2.b -= lr * self.linear2.grad_b
```

---

## Complete Training Loop

```python
for epoch in range(epochs):
    loss = mlp.forward(X, y)
    mlp.backward(y)
    mlp.update(lr)
    print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

---

*You now have all pieces to implement backpropagation!*

# Hint 1: L2 Regularization

---

## Weight Decay

Add penalty to loss:
```python
def l2_loss(weights, lambda_=0.01):
    return 0.5 * lambda_ * sum(np.sum(w**2) for w in weights)

total_loss = data_loss + l2_loss(model.weights)
```

---

## Gradient Update

```python
# Standard gradient with L2
grad_w = data_grad + lambda_ * w

# Or equivalently (weight decay)
w = w - lr * data_grad - lr * lambda_ * w
w = (1 - lr * lambda_) * w - lr * data_grad
```

---

*Next: [Hint 2 - Dropout](hint-2-dropout.md)*

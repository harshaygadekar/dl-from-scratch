# Topic 10: Math Refresh

---

## Complete Forward Pass

```
x: (batch, 784)
h1 = ReLU(x @ W1 + b1)     # (batch, 256)
h2 = ReLU(h1 @ W2 + b2)    # (batch, 128)
logits = h2 @ W3 + b3      # (batch, 10)
probs = softmax(logits)    # (batch, 10)
loss = cross_entropy(probs, y)
```

---

## Complete Backward Pass

```
dlogits = probs - y                    # (batch, 10)
dW3 = h2.T @ dlogits
db3 = dlogits.sum(axis=0)
dh2 = dlogits @ W3.T

dh2_pre = dh2 * (h2 > 0)               # ReLU backward
dW2 = h1.T @ dh2_pre
db2 = dh2_pre.sum(axis=0)
dh1 = dh2_pre @ W2.T

dh1_pre = dh1 * (h1 > 0)               # ReLU backward
dW1 = x.T @ dh1_pre
db1 = dh1_pre.sum(axis=0)
```

---

## Accuracy

```
predictions = argmax(logits, axis=1)
accuracy = mean(predictions == labels)
```

---

*"The math is straightforward—it's the engineering that's hard."*

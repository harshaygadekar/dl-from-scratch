# Topic 10: Intuition Guide

The full picture—putting it all together.

---

## 🧠 The Training Loop

```
Data → Forward → Loss → Backward → Update → Repeat
```

This is the heartbeat of deep learning.

---

## MNIST: The Hello World of ML

- 28×28 grayscale images of digits 0-9
- 60,000 training, 10,000 test
- Simple enough to train on CPU
- Complex enough to be meaningful

---

## The Architecture

```
Input (784) → Dense (256) → ReLU → Dropout
           → Dense (128) → ReLU → Dropout
           → Dense (10) → Softmax → Prediction
```

Why this architecture?
- 784 = 28 × 28 (flattened image)
- Two hidden layers capture patterns
- 10 outputs = 10 digit classes

---

## Training Dynamics

**Epoch 1**: Random guessing (~10% accuracy)
**Epoch 5**: Learning patterns (~85% accuracy)
**Epoch 10**: Refined (~95% accuracy)

Watch the loss decrease and accuracy increase!

---

## Common Pitfalls

1. **Forgetting to normalize**: Pixel values 0-255 → 0-1
2. **Wrong mode**: Train vs eval for dropout/batchnorm
3. **Not shuffling**: Same order each epoch = patterns
4. **Too high LR**: Loss explodes
5. **Too low LR**: Takes forever

---

*"When you can train MNIST from scratch, you understand neural networks."*

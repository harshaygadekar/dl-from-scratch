# Topic 10: Interview Questions

End-to-end training questions are common!

---

## Q1: Walk Through a Training Loop (Every Interview)

**Difficulty**: Medium | **Time**: 5 min

<details>
<summary>Answer</summary>

```python
for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        # 1. Forward pass
        logits = model(x_batch)
        loss = cross_entropy(logits, y_batch)
        
        # 2. Backward pass
        loss.backward()
        
        # 3. Update weights
        optimizer.step()
        
        # 4. Zero gradients
        optimizer.zero_grad()
    
    # 5. Validation
    model.eval()
    val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch}: Val Acc = {val_acc:.2%}")
```

**Key points**:
1. Switch between train/eval modes
2. Zero gradients before each batch
3. Forward → Backward → Update order
4. Validate after each epoch

</details>

---

## Q2: Why Batch Training? (Common)

**Difficulty**: Easy | **Time**: 3 min

<details>
<summary>Answer</summary>

**Three reasons**:

1. **Memory**: Can't fit entire dataset in GPU memory
2. **Noise**: Mini-batch gradients add regularization
3. **Speed**: Parallelism within each batch

**Batch size tradeoffs**:
- Smaller: More noise, may generalize better
- Larger: Stable gradients, faster per epoch
- Typical: 32-256

</details>

---

## Q3: Learning Rate Scheduling (Google, Meta)

**Difficulty**: Medium | **Time**: 5 min

<details>
<summary>Answer</summary>

**Common schedules**:

1. **Step decay**: lr × 0.1 every N epochs
2. **Exponential**: lr × γ^epoch
3. **Cosine annealing**: lr × (1 + cos(π × t/T)) / 2
4. **Warmup + decay**: Linear warmup, then decay

```python
# Warmup + cosine decay
def get_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
```

</details>

---

## Q4: Debugging Poor Training (Practical)

**Difficulty**: Medium | **Time**: 5 min

<details>
<summary>Answer</summary>

**Checklist**:

1. **Loss not decreasing?**
   - Learning rate too high/low
   - Check gradient magnitudes
   - Verify data loading

2. **Training good, validation bad?**
   - Overfitting → more regularization
   - Data augmentation

3. **NaN loss?**
   - Learning rate too high
   - Numerical instability
   - Check for log(0)

4. **Accuracy stuck at random?**
   - Check labels
   - Verify forward pass
   - Wrong loss function

</details>

---

## Q5: MNIST Specific Tips

**Difficulty**: Easy | **Time**: 3 min

<details>
<summary>Answer</summary>

**For 95%+ accuracy**:
- Input normalization: x = x / 255 - 0.5
- Architecture: [784, 256, 128, 10]
- ReLU activation
- Dropout 0.2
- Adam optimizer, lr=0.001
- 10-20 epochs

**For 97%+**:
- Add BatchNorm
- Data augmentation (small rotations)
- Larger model

</details>

---

*"Training a model end-to-end tests all your knowledge—this is the real interview test."*

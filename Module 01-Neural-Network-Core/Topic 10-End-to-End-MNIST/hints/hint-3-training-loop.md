# Hint 3: Training Loop

---

## Full Training Loop

```python
def train(model, x_train, y_train, x_val, y_val, epochs=10, lr=0.001):
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = SoftmaxCrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for x_batch, y_batch in get_batches(x_train, y_train):
            # Forward
            logits = model.forward(x_batch)
            loss = loss_fn.forward(logits, y_batch)
            epoch_loss += loss
            
            # Backward
            grad = loss_fn.backward()
            model.backward(grad)
            
            # Update
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation
        model.eval()
        val_acc = evaluate(model, x_val, y_val)
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Val Acc={val_acc:.2%}")
```

---

## Evaluation

```python
def evaluate(model, x, y):
    logits = model.forward(x)
    preds = np.argmax(logits, axis=1)
    labels = np.argmax(y, axis=1)
    return np.mean(preds == labels)
```

---

*You're ready to train MNIST!*

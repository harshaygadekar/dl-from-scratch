"""
Topic 04: Single Layer Perceptron - Level 03 Memory-Efficient Implementation

Production-quality perceptron with:
- Learning rate scheduling
- Early stopping
- Regularization
- Memory-efficient batch processing
"""

import numpy as np
from typing import Tuple, List, Optional, Dict


class Perceptron:
    """
    Memory-efficient perceptron with production features.
    
    Features:
    - L2 regularization
    - Learning rate decay
    - Early stopping
    - Gradient accumulation
    """
    
    def __init__(self, input_dim: int, l2_lambda: float = 0.0):
        """
        Initialize perceptron.
        
        Args:
            input_dim: Number of features
            l2_lambda: L2 regularization strength
        """
        # Xavier initialization
        self.w = np.random.randn(input_dim) * np.sqrt(2.0 / input_dim)
        self.b = 0.0
        self.l2_lambda = l2_lambda
        
        # Gradient accumulators
        self.dw = np.zeros(input_dim)
        self.db = 0.0
        self._grad_count = 0
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        positive = z >= 0
        negative = ~positive
        result = np.empty_like(z, dtype=np.float64)
        result[positive] = 1 / (1 + np.exp(-z[positive]))
        exp_z = np.exp(z[negative])
        result[negative] = exp_z / (1 + exp_z)
        return result
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass."""
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """BCE loss with L2 regularization."""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        l2_term = 0.5 * self.l2_lambda * np.sum(self.w ** 2)
        return bce + l2_term
    
    def accumulate_gradients(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Accumulate gradients for gradient accumulation strategy.
        
        Useful when batch_size * accumulation_steps > GPU memory.
        """
        error = y_pred - y_true
        self.dw += np.dot(error, X) + self.l2_lambda * self.w
        self.db += np.sum(error)
        self._grad_count += len(y_true)
    
    def step(self, lr: float):
        """Apply accumulated gradients."""
        if self._grad_count > 0:
            self.w -= lr * (self.dw / self._grad_count)
            self.b -= lr * (self.db / self._grad_count)
            self.dw.fill(0)
            self.db = 0.0
            self._grad_count = 0
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return (self.forward(X) >= 0.5).astype(int)
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        return np.mean(self.predict(X) == y)


class LRScheduler:
    """Learning rate schedulers."""
    
    @staticmethod
    def constant(initial_lr: float, epoch: int) -> float:
        return initial_lr
    
    @staticmethod
    def step_decay(initial_lr: float, epoch: int, 
                   decay_rate: float = 0.5, step_size: int = 20) -> float:
        return initial_lr * (decay_rate ** (epoch // step_size))
    
    @staticmethod
    def exponential_decay(initial_lr: float, epoch: int, 
                          decay_rate: float = 0.95) -> float:
        return initial_lr * (decay_rate ** epoch)
    
    @staticmethod
    def cosine_annealing(initial_lr: float, epoch: int, 
                         total_epochs: int) -> float:
        return initial_lr * (1 + np.cos(np.pi * epoch / total_epochs)) / 2


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train(X_train: np.ndarray, y_train: np.ndarray,
          X_val: Optional[np.ndarray] = None, 
          y_val: Optional[np.ndarray] = None,
          epochs: int = 100,
          lr: float = 0.1,
          batch_size: int = 32,
          l2_lambda: float = 0.01,
          patience: int = 10,
          scheduler: str = 'cosine') -> Tuple[Perceptron, Dict]:
    """
    Train with production features.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (optional)
        epochs: Max epochs
        lr: Initial learning rate
        batch_size: Mini-batch size
        l2_lambda: L2 regularization
        patience: Early stopping patience
        scheduler: LR scheduler type
    
    Returns:
        Trained model and history dict
    """
    model = Perceptron(X_train.shape[1], l2_lambda=l2_lambda)
    early_stopping = EarlyStopping(patience=patience)
    
    # Select scheduler
    schedulers = {
        'constant': LRScheduler.constant,
        'step': lambda lr, e: LRScheduler.step_decay(lr, e),
        'exponential': lambda lr, e: LRScheduler.exponential_decay(lr, e),
        'cosine': lambda lr, e: LRScheduler.cosine_annealing(lr, e, epochs)
    }
    get_lr = schedulers.get(scheduler, LRScheduler.constant)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    
    for epoch in range(epochs):
        current_lr = get_lr(lr, epoch)
        indices = np.random.permutation(len(X_train))
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            
            y_pred = model.forward(X_batch)
            epoch_loss += model.compute_loss(y_pred, y_batch)
            n_batches += 1
            
            model.accumulate_gradients(X_batch, y_batch, y_pred)
            model.step(current_lr)
        
        train_loss = epoch_loss / n_batches
        train_acc = model.accuracy(X_train, y_train)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['lr'].append(current_lr)
        
        # Validation
        if X_val is not None:
            val_pred = model.forward(X_val)
            val_loss = model.compute_loss(val_pred, y_val)
            val_acc = model.accuracy(X_val, y_val)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            msg = f"Epoch {epoch:3d}: lr={current_lr:.6f}, train_loss={train_loss:.4f}, train_acc={train_acc:.2%}"
            if X_val is not None:
                msg += f", val_loss={val_loss:.4f}, val_acc={val_acc:.2%}"
            print(msg)
    
    return model, history


def generate_data(n_samples: int = 1000, noise: float = 0.5, seed: int = 42):
    """Generate data with controlled noise level."""
    np.random.seed(seed)
    
    X_pos = np.random.randn(n_samples // 2, 2) * noise + np.array([2, 2])
    X_neg = np.random.randn(n_samples // 2, 2) * noise + np.array([-2, -2])
    
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))
    
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]


def demo():
    """Demonstrate production-quality perceptron."""
    print("=" * 60)
    print("Single Layer Perceptron - Level 03 (Memory-Efficient)")
    print("=" * 60)
    
    # Generate train/val split
    X, y = generate_data(1200, noise=0.8)
    X_train, y_train = X[:1000], y[:1000]
    X_val, y_val = X[1000:], y[1000:]
    
    print(f"\nTrain: {len(X_train)} samples, Val: {len(X_val)} samples")
    
    # Train with all features
    print("\nTraining with L2 regularization & cosine annealing...")
    model, history = train(
        X_train, y_train, X_val, y_val,
        epochs=100, lr=0.1, batch_size=32,
        l2_lambda=0.01, patience=15, scheduler='cosine'
    )
    
    print(f"\nFinal train accuracy: {model.accuracy(X_train, y_train):.2%}")
    print(f"Final val accuracy: {model.accuracy(X_val, y_val):.2%}")


if __name__ == "__main__":
    demo()

"""
Topic 04: Perceptron Visualization

Visualize decision boundaries, training progress, and loss surfaces.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def sigmoid(z):
    """Numerically stable sigmoid."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def generate_data(n_samples=200, seed=42):
    """Generate linearly separable data."""
    np.random.seed(seed)
    X_pos = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    X_neg = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]


def plot_decision_boundary(w, b, X, y, title="Decision Boundary"):
    """Plot the perceptron decision boundary."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Compute predictions
    Z = sigmoid(xx * w[0] + yy * w[1] + b)
    
    # Plot contour
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.4, cmap=cmap_light)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    # Plot data points
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='o', 
               s=50, label='Class 0', edgecolors='k')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='s', 
               s=50, label='Class 1', edgecolors='k')
    
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    
    return fig


def plot_training_progress(history):
    """Plot loss and accuracy during training."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(len(history['loss']))
    
    # Loss
    axes[0].plot(epochs, history['loss'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['accuracy'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training Accuracy', fontsize=14)
    axes[1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_loss_surface(X, y, w_range=(-5, 5), b_range=(-5, 5), resolution=50):
    """Plot the loss surface for a perceptron."""
    fig = plt.figure(figsize=(12, 5))
    
    w0_vals = np.linspace(*w_range, resolution)
    w1_vals = np.linspace(*w_range, resolution)
    W0, W1 = np.meshgrid(w0_vals, w1_vals)
    
    # Compute loss for each (w0, w1) combination (b=0)
    losses = np.zeros_like(W0)
    for i in range(resolution):
        for j in range(resolution):
            z = X[:, 0] * W0[i, j] + X[:, 1] * W1[i, j]
            y_pred = sigmoid(z)
            eps = 1e-15
            y_pred = np.clip(y_pred, eps, 1 - eps)
            loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            losses[i, j] = loss
    
    # 2D contour
    ax1 = fig.add_subplot(121)
    cs = ax1.contourf(W0, W1, losses, levels=30, cmap='viridis')
    plt.colorbar(cs, ax=ax1)
    ax1.set_xlabel('w₀', fontsize=12)
    ax1.set_ylabel('w₁', fontsize=12)
    ax1.set_title('Loss Surface (b=0)', fontsize=14)
    
    # 3D surface
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(W0, W1, losses, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('w₀')
    ax2.set_ylabel('w₁')
    ax2.set_zlabel('Loss')
    ax2.set_title('Loss Surface 3D', fontsize=14)
    
    plt.tight_layout()
    return fig


def plot_sigmoid():
    """Visualize sigmoid function and its derivative."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    z = np.linspace(-6, 6, 200)
    sigma = sigmoid(z)
    dsigma = sigma * (1 - sigma)
    
    # Sigmoid
    axes[0].plot(z, sigma, 'b-', linewidth=2, label='σ(z)')
    axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('z', fontsize=12)
    axes[0].set_ylabel('σ(z)', fontsize=12)
    axes[0].set_title('Sigmoid Function', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Derivative
    axes[1].plot(z, dsigma, 'g-', linewidth=2, label="σ'(z)")
    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('z', fontsize=12)
    axes[1].set_ylabel("σ'(z)", fontsize=12)
    axes[1].set_title('Sigmoid Derivative', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def demo():
    """Run all visualizations."""
    print("Topic 04: Perceptron Visualizations")
    print("=" * 40)
    
    # Generate data
    X, y = generate_data(200)
    
    # 1. Sigmoid function
    print("\n1. Plotting sigmoid function...")
    fig1 = plot_sigmoid()
    plt.savefig('sigmoid_visualization.png', dpi=150, bbox_inches='tight')
    print("   Saved: sigmoid_visualization.png")
    
    # 2. Decision boundary (untrained)
    print("\n2. Plotting initial (random) decision boundary...")
    np.random.seed(42)
    w = np.random.randn(2) * 0.01
    b = 0.0
    fig2 = plot_decision_boundary(w, b, X, y, "Initial Decision Boundary")
    plt.savefig('initial_boundary.png', dpi=150, bbox_inches='tight')
    print("   Saved: initial_boundary.png")
    
    # 3. Train and track progress
    print("\n3. Training and plotting progress...")
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(100):
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)
        
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(y * np.log(y_pred_clipped) + (1-y) * np.log(1 - y_pred_clipped))
        accuracy = np.mean((y_pred >= 0.5) == y)
        
        history['loss'].append(loss)
        history['accuracy'].append(accuracy)
        
        error = y_pred - y
        dw = np.dot(error, X) / len(X)
        db = np.mean(error)
        
        w -= 0.5 * dw
        b -= 0.5 * db
    
    fig3 = plot_training_progress(history)
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    print("   Saved: training_progress.png")
    
    # 4. Final decision boundary
    print("\n4. Plotting final decision boundary...")
    fig4 = plot_decision_boundary(w, b, X, y, f"Final Decision Boundary (Acc: {history['accuracy'][-1]:.1%})")
    plt.savefig('final_boundary.png', dpi=150, bbox_inches='tight')
    print("   Saved: final_boundary.png")
    
    # 5. Loss surface
    print("\n5. Plotting loss surface...")
    fig5 = plot_loss_surface(X, y)
    plt.savefig('loss_surface.png', dpi=150, bbox_inches='tight')
    print("   Saved: loss_surface.png")
    
    print("\n" + "=" * 40)
    print("All visualizations saved!")
    plt.show()


if __name__ == "__main__":
    demo()

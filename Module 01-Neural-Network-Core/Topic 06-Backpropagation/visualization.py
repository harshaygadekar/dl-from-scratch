"""
Topic 06: Backpropagation Visualization

Visualize gradient flow, loss landscapes, and training dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_gradient_flow():
    """Visualize how gradients flow through a network."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    layers = ['Input', 'Linear', 'ReLU', 'Linear', 'ReLU', 'Linear', 'Softmax', 'Loss']
    n = len(layers)
    
    for i, name in enumerate(layers):
        color = 'lightblue' if i % 2 == 0 else 'lightgreen'
        if name in ['Loss', 'Softmax']:
            color = 'lightyellow'
        
        rect = plt.Rectangle((i, 0), 0.8, 1, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(i + 0.4, 0.5, name, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(i + 0.4, 0.2, f'Layer {i}', ha='center', va='center', fontsize=8, alpha=0.6)
        
        if i < n - 1:
            ax.annotate('', xy=(i+0.9, 0.7), xytext=(i+1, 0.7),
                       arrowprops=dict(arrowstyle='->', color='blue', lw=2))
            ax.annotate('', xy=(i+1, 0.3), xytext=(i+0.9, 0.3),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    ax.text(n/2, 1.2, 'Forward Pass (blue)', ha='center', color='blue', fontsize=12)
    ax.text(n/2, -0.3, 'Backward Pass (red)', ha='center', color='red', fontsize=12)
    
    ax.set_xlim(-0.5, n + 0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Gradient Flow Through Neural Network', fontsize=14, fontweight='bold')
    
    return fig


def visualize_training_dynamics():
    """Simulate and visualize training."""
    np.random.seed(42)
    
    # Simulate training metrics
    epochs = 100
    train_loss = 2.5 * np.exp(-0.03 * np.arange(epochs)) + 0.1 + 0.05 * np.random.randn(epochs)
    val_loss = 2.5 * np.exp(-0.025 * np.arange(epochs)) + 0.2 + 0.08 * np.random.randn(epochs)
    train_acc = 1 - np.exp(-0.05 * np.arange(epochs)) + 0.05 * np.random.randn(epochs)
    train_acc = np.clip(train_acc, 0, 1)
    
    grad_norms = 1.0 * np.exp(-0.02 * np.arange(epochs)) + 0.1 * np.random.randn(epochs)
    grad_norms = np.abs(grad_norms)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(train_loss, 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(val_loss, 'r--', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(train_acc, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].semilogy(grad_norms, 'm-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Gradient Norm (log)')
    axes[1, 0].set_title('Gradient Magnitude Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    lrs = [0.001, 0.01, 0.1]
    for lr in lrs:
        loss = 2.5 * np.exp(-lr * np.arange(epochs)) + 0.1
        if lr == 0.1:
            loss = loss + 0.3 * np.sin(0.5 * np.arange(epochs))
        axes[1, 1].plot(loss, label=f'lr={lr}', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Effect of Learning Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def demo():
    """Run all visualizations."""
    print("Topic 06: Backpropagation Visualizations")
    print("=" * 40)
    
    print("\n1. Gradient flow diagram...")
    fig1 = visualize_gradient_flow()
    plt.savefig('gradient_flow.png', dpi=150, bbox_inches='tight')
    
    print("2. Training dynamics...")
    fig2 = visualize_training_dynamics()
    plt.savefig('training_dynamics.png', dpi=150, bbox_inches='tight')
    
    print("\nAll visualizations saved!")
    plt.show()


if __name__ == "__main__":
    demo()

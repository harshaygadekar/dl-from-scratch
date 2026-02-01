"""
Topic 07: Activation Functions Visualization
"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_all_activations():
    """Plot all activation functions."""
    x = np.linspace(-5, 5, 200)
    
    relu = np.maximum(0, x)
    leaky_relu = np.where(x > 0, x, 0.1 * x)
    elu = np.where(x > 0, x, np.exp(x) - 1)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    
    # GELU approximation
    gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    activations = [
        ('ReLU', relu, 'tab:blue'),
        ('LeakyReLU (α=0.1)', leaky_relu, 'tab:orange'),
        ('ELU', elu, 'tab:green'),
        ('Sigmoid', sigmoid, 'tab:red'),
        ('Tanh', tanh, 'tab:purple'),
        ('GELU', gelu, 'tab:brown'),
    ]
    
    for ax, (name, y, color) in zip(axes.flat, activations):
        ax.plot(x, y, color=color, linewidth=2.5)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 5)
    
    plt.suptitle('Activation Functions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_gradients():
    """Plot gradients of activations."""
    x = np.linspace(-5, 5, 200)
    
    relu_grad = (x > 0).astype(float)
    sigmoid = 1 / (1 + np.exp(-x))
    sigmoid_grad = sigmoid * (1 - sigmoid)
    tanh = np.tanh(x)
    tanh_grad = 1 - tanh ** 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(x, relu_grad, 'b-', linewidth=2.5)
    axes[0].set_title('ReLU Gradient', fontsize=14, fontweight='bold')
    axes[0].set_ylim(-0.1, 1.5)
    
    axes[1].plot(x, sigmoid_grad, 'r-', linewidth=2.5)
    axes[1].set_title('Sigmoid Gradient', fontsize=14, fontweight='bold')
    axes[1].axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='max=0.25')
    axes[1].legend()
    
    axes[2].plot(x, tanh_grad, 'purple', linewidth=2.5)
    axes[2].set_title('Tanh Gradient', fontsize=14, fontweight='bold')
    
    for ax in axes:
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel("f'(x)")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Activation Gradients', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def demo():
    print("Topic 07: Activation Functions Visualization")
    print("=" * 40)
    
    fig1 = visualize_all_activations()
    plt.savefig('activations.png', dpi=150, bbox_inches='tight')
    
    fig2 = visualize_gradients()
    plt.savefig('gradients.png', dpi=150, bbox_inches='tight')
    
    print("Saved: activations.png, gradients.png")
    plt.show()


if __name__ == "__main__":
    demo()

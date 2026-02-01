"""
Topic 08: Loss Functions Visualization
"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_regression_losses():
    """Compare MSE, MAE, and Huber."""
    errors = np.linspace(-3, 3, 200)
    
    mse = errors ** 2
    mae = np.abs(errors)
    huber = np.where(np.abs(errors) <= 1, 0.5 * errors**2, np.abs(errors) - 0.5)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(errors, mse, 'b-', linewidth=2.5, label='MSE')
    ax.plot(errors, mae, 'r-', linewidth=2.5, label='MAE')
    ax.plot(errors, huber, 'g-', linewidth=2.5, label='Huber (δ=1)')
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Error (y_pred - y_true)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Regression Loss Functions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 5)
    
    return fig


def visualize_cross_entropy():
    """Visualize cross-entropy loss."""
    p = np.linspace(0.01, 0.99, 200)
    
    loss_y1 = -np.log(p)
    loss_y0 = -np.log(1 - p)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(p, loss_y1, 'b-', linewidth=2.5, label='y=1 (true class)')
    ax.plot(p, loss_y0, 'r-', linewidth=2.5, label='y=0 (wrong class)')
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Cross-Entropy Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 5)
    
    return fig


def demo():
    print("Topic 08: Loss Functions Visualization")
    print("=" * 40)
    
    fig1 = visualize_regression_losses()
    plt.savefig('regression_losses.png', dpi=150, bbox_inches='tight')
    
    fig2 = visualize_cross_entropy()
    plt.savefig('cross_entropy.png', dpi=150, bbox_inches='tight')
    
    print("Saved: regression_losses.png, cross_entropy.png")
    plt.show()


if __name__ == "__main__":
    demo()

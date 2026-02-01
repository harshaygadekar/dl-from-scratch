"""Topic 09: Regularization Visualization"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_dropout():
    """Show dropout effect."""
    np.random.seed(42)
    x = np.ones((10, 10))
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for idx, p in enumerate([0.0, 0.3, 0.5]):
        mask = np.random.rand(10, 10) > p
        out = x * mask / max(1 - p, 0.01)
        
        axes[idx].imshow(out, cmap='RdYlGn', vmin=0, vmax=2)
        axes[idx].set_title(f'Dropout p={p}')
        axes[idx].axis('off')
    
    plt.suptitle('Dropout Effect', fontsize=14, fontweight='bold')
    return fig


def visualize_batchnorm():
    """Show BatchNorm effect."""
    np.random.seed(42)
    
    x = np.random.randn(1000) * 5 + 10
    x_norm = (x - x.mean()) / x.std()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].hist(x, bins=50, alpha=0.7, color='red')
    axes[0].set_title(f'Before BN\nμ={x.mean():.1f}, σ={x.std():.1f}')
    
    axes[1].hist(x_norm, bins=50, alpha=0.7, color='green')
    axes[1].set_title(f'After BN\nμ={x_norm.mean():.2f}, σ={x_norm.std():.2f}')
    
    plt.suptitle('Batch Normalization Effect', fontsize=14, fontweight='bold')
    return fig


def demo():
    print("Topic 09: Regularization Visualization")
    
    fig1 = visualize_dropout()
    plt.savefig('dropout_viz.png', dpi=150, bbox_inches='tight')
    
    fig2 = visualize_batchnorm()
    plt.savefig('batchnorm_viz.png', dpi=150, bbox_inches='tight')
    
    print("Saved visualizations")
    plt.show()


if __name__ == "__main__":
    demo()

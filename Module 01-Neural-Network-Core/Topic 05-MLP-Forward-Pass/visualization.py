"""
Topic 05: MLP Visualization

Visualize initialization effects, activation distributions, and network architecture.
"""

import numpy as np
import matplotlib.pyplot as plt


def relu(x): return np.maximum(0, x)


def visualize_initialization_comparison():
    """Compare different initialization schemes."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    n_in, n_out = 512, 512
    np.random.seed(42)
    
    inits = {
        'Too Small (0.001)': 0.001,
        'Too Large (1.0)': 1.0,
        'Xavier': np.sqrt(2.0 / (n_in + n_out)),
        'Kaiming': np.sqrt(2.0 / n_in),
        'LeCun': np.sqrt(1.0 / n_in)
    }
    
    for idx, (name, std) in enumerate(inits.items()):
        ax = axes[idx // 3, idx % 3]
        W = np.random.randn(n_in, n_out) * std
        
        ax.hist(W.flatten(), bins=50, density=True, alpha=0.7, color='steelblue')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_title(f'{name}\nstd={std:.4f}, var={W.var():.6f}')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Density')
    
    axes[1, 2].axis('off')
    plt.tight_layout()
    return fig


def visualize_activation_propagation():
    """Show how activations propagate through layers."""
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    
    configs = [
        ('Good Init (Kaiming)', np.sqrt(2.0 / 256)),
        ('Bad Init (Too Small)', 0.01)
    ]
    
    for row, (config_name, init_std) in enumerate(configs):
        np.random.seed(42)
        x = np.random.randn(1000, 256)
        h = x.copy()
        
        for col in range(5):
            W = np.random.randn(256, 256) * init_std
            h = np.dot(h, W)
            h = relu(h)
            
            ax = axes[row, col]
            ax.hist(h.flatten(), bins=50, density=True, alpha=0.7, 
                   color='steelblue' if row == 0 else 'coral')
            ax.set_title(f'Layer {col+1}\nmean={h.mean():.2f}, std={h.std():.2f}')
            ax.set_xlabel('Activation')
            if col == 0:
                ax.set_ylabel(config_name)
    
    plt.suptitle('Activation Propagation: Good vs Bad Initialization', fontsize=14)
    plt.tight_layout()
    return fig


def visualize_network_architecture():
    """Draw a simple network diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    layers = [4, 6, 6, 3]
    colors = ['#3498db', '#2ecc71', '#2ecc71', '#e74c3c']
    
    max_neurons = max(layers)
    layer_spacing = 3
    
    for l, (n_neurons, color) in enumerate(zip(layers, colors)):
        y_offset = (max_neurons - n_neurons) / 2
        
        for n in range(n_neurons):
            circle = plt.Circle((l * layer_spacing, n + y_offset), 0.3, 
                               color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            
            # Draw connections to next layer
            if l < len(layers) - 1:
                next_n_neurons = layers[l + 1]
                next_y_offset = (max_neurons - next_n_neurons) / 2
                for next_n in range(next_n_neurons):
                    ax.plot([l * layer_spacing + 0.3, (l + 1) * layer_spacing - 0.3],
                           [n + y_offset, next_n + next_y_offset],
                           'gray', alpha=0.3, linewidth=0.5)
    
    # Labels
    labels = ['Input\n(4)', 'Hidden 1\n(6)', 'Hidden 2\n(6)', 'Output\n(3)']
    for l, label in enumerate(labels):
        ax.text(l * layer_spacing, -1.5, label, ha='center', fontsize=12)
    
    ax.set_xlim(-1, (len(layers) - 1) * layer_spacing + 1)
    ax.set_ylim(-2.5, max_neurons + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('MLP Architecture: 4 → 6 → 6 → 3', fontsize=14)
    
    return fig


def demo():
    """Run all visualizations."""
    print("Topic 05: MLP Visualizations")
    print("=" * 40)
    
    print("\n1. Initialization comparison...")
    fig1 = visualize_initialization_comparison()
    plt.savefig('init_comparison.png', dpi=150, bbox_inches='tight')
    
    print("2. Activation propagation...")
    fig2 = visualize_activation_propagation()
    plt.savefig('activation_propagation.png', dpi=150, bbox_inches='tight')
    
    print("3. Network architecture...")
    fig3 = visualize_network_architecture()
    plt.savefig('network_architecture.png', dpi=150, bbox_inches='tight')
    
    print("\nAll visualizations saved!")
    plt.show()


if __name__ == "__main__":
    demo()

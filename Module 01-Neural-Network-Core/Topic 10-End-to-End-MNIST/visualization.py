"""Topic 10: End-to-End MNIST Visualization"""

import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves():
    """Simulate and plot training curves."""
    epochs = np.arange(1, 21)
    
    # Simulated curves
    train_loss = 2.5 * np.exp(-epochs * 0.3) + 0.1
    val_loss = 2.5 * np.exp(-epochs * 0.25) + 0.15
    train_acc = 1 - 0.9 * np.exp(-epochs * 0.4)
    val_acc = 1 - 0.9 * np.exp(-epochs * 0.35) - 0.02
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss
    axes[0].plot(epochs, train_loss, 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, val_loss, 'r--', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, train_acc * 100, 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, val_acc * 100, 'r--', label='Validation', linewidth=2)
    axes[1].axhline(y=95, color='g', linestyle=':', label='95% Target')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('MNIST Training Progress', fontsize=14, fontweight='bold')
    return fig


def plot_sample_predictions():
    """Plot sample MNIST predictions (fake data)."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    
    for i, ax in enumerate(axes.flat):
        # Fake digit image
        img = np.random.rand(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Pred: {i % 10}')
        ax.axis('off')
    
    plt.suptitle('Sample MNIST Predictions', fontsize=14, fontweight='bold')
    return fig


def demo():
    print("Topic 10: MNIST Visualization")
    
    fig1 = plot_training_curves()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    
    fig2 = plot_sample_predictions()
    plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
    
    print("Saved training_curves.png and sample_predictions.png")
    plt.show()


if __name__ == "__main__":
    demo()

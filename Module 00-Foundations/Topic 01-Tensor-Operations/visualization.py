#!/usr/bin/env python3
"""
Visualization Tools for Topic 01: Tensor Operations

Interactive visualizations to help understand:
- Memory layouts
- Stride patterns
- Broadcasting behavior
"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_memory_layout(arr: np.ndarray, title: str = "Memory Layout"):
    """
    Visualize how an array is laid out in memory.
    
    Shows both the logical view and the flat memory view.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Logical view
    ax1 = axes[0]
    ax1.set_title(f"Logical View\nShape: {arr.shape}, Strides: {arr.strides}")
    
    if arr.ndim == 1:
        ax1.bar(range(len(arr)), arr.flatten())
        ax1.set_xlabel("Index")
        ax1.set_ylabel("Value")
    elif arr.ndim == 2:
        im = ax1.imshow(arr, cmap='viridis', aspect='auto')
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax1.text(j, i, f'{arr[i,j]:.1f}', ha='center', va='center', 
                        color='white' if arr[i,j] < arr.mean() else 'black')
        ax1.set_xlabel("Column")
        ax1.set_ylabel("Row")
        plt.colorbar(im, ax=ax1)
    
    # Memory view
    ax2 = axes[1]
    flat = arr.flatten() if arr.flags['C_CONTIGUOUS'] else arr.ravel()
    ax2.bar(range(len(flat)), flat, color='steelblue')
    ax2.set_title(f"Memory View (Flat)\nTotal elements: {arr.size}")
    ax2.set_xlabel("Memory Position")
    ax2.set_ylabel("Value")
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_broadcasting(a: np.ndarray, b: np.ndarray):
    """
    Visualize how two arrays broadcast together.
    """
    try:
        result = a + b
    except ValueError as e:
        print(f"Cannot broadcast: {e}")
        return None
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original a
    ax1 = axes[0]
    ax1.set_title(f"Array A\nShape: {a.shape}")
    if a.ndim == 1:
        ax1.bar(range(len(a)), a)
    else:
        ax1.imshow(a.reshape(-1, a.shape[-1]) if a.ndim > 2 else a, 
                   cmap='Blues', aspect='auto')
    
    # Original b
    ax2 = axes[1]
    ax2.set_title(f"Array B\nShape: {b.shape}")
    if b.ndim == 1:
        ax2.bar(range(len(b)), b)
    else:
        ax2.imshow(b.reshape(-1, b.shape[-1]) if b.ndim > 2 else b,
                   cmap='Reds', aspect='auto')
    
    # Broadcast shapes
    ax3 = axes[2]
    ax3.axis('off')
    ax3.set_title("Broadcasting")
    
    text = f"A: {a.shape}\n"
    text += f"B: {b.shape}\n"
    text += "─" * 20 + "\n"
    text += f"Result: {result.shape}"
    ax3.text(0.5, 0.5, text, ha='center', va='center', fontsize=12,
             family='monospace', transform=ax3.transAxes)
    
    # Result
    ax4 = axes[3]
    ax4.set_title(f"Result (A + B)\nShape: {result.shape}")
    if result.ndim == 1:
        ax4.bar(range(len(result)), result)
    else:
        im = ax4.imshow(result.reshape(-1, result.shape[-1]) if result.ndim > 2 else result,
                        cmap='Greens', aspect='auto')
        plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    return fig


def visualize_strides(arr: np.ndarray):
    """
    Visualize how strides affect element access.
    """
    print("="*50)
    print("Stride Visualization")
    print("="*50)
    print(f"Shape: {arr.shape}")
    print(f"Strides: {arr.strides}")
    print(f"Item size: {arr.itemsize} bytes")
    print(f"Is C-contiguous: {arr.flags['C_CONTIGUOUS']}")
    print(f"Is F-contiguous: {arr.flags['F_CONTIGUOUS']}")
    print()
    
    if arr.ndim == 2:
        rows, cols = arr.shape
        print("Memory offset for each element (in items):")
        print()
        
        for i in range(min(rows, 4)):
            for j in range(min(cols, 6)):
                offset = (i * arr.strides[0] + j * arr.strides[1]) // arr.itemsize
                print(f"[{i},{j}]→{offset:3d}  ", end="")
            print()
    
    print()


def visualize_view_vs_copy():
    """
    Demonstrate the difference between views and copies.
    """
    print("="*50)
    print("View vs Copy Demonstration")
    print("="*50)
    
    # Create base array
    original = np.arange(12).reshape(3, 4)
    print(f"Original array:\n{original}\n")
    
    # Views
    print("VIEWS (share memory):")
    print("-" * 30)
    
    view1 = original[::2, :]  # Slicing
    view2 = original.T        # Transpose
    view3 = original.reshape(4, 3)  # Reshape
    
    print(f"original[::2, :]   shares memory: {np.shares_memory(original, view1)}")
    print(f"original.T          shares memory: {np.shares_memory(original, view2)}")
    print(f"original.reshape()  shares memory: {np.shares_memory(original, view3)}")
    
    # Copies
    print("\nCOPIES (independent memory):")
    print("-" * 30)
    
    copy1 = original.copy()
    copy2 = original[[0, 2], :]  # Fancy indexing
    copy3 = original + 0  # Arithmetic creates copy
    
    print(f"original.copy()     shares memory: {np.shares_memory(original, copy1)}")
    print(f"original[[0,2], :]  shares memory: {np.shares_memory(original, copy2)}")
    print(f"original + 0        shares memory: {np.shares_memory(original, copy3)}")
    
    print("\n" + "="*50)


def sliding_window_demo():
    """
    Demonstrate sliding window with stride tricks.
    """
    from numpy.lib.stride_tricks import as_strided
    
    print("="*50)
    print("Sliding Window Demo (Zero-Copy)")
    print("="*50)
    
    x = np.arange(10)
    print(f"Original array: {x}")
    print(f"Original strides: {x.strides}")
    print()
    
    window_size = 4
    n = len(x)
    output_len = n - window_size + 1
    
    new_shape = (output_len, window_size)
    new_strides = (x.strides[0], x.strides[0])
    
    windows = as_strided(x, shape=new_shape, strides=new_strides)
    
    print(f"Window size: {window_size}")
    print(f"New shape: {new_shape}")
    print(f"New strides: {new_strides}")
    print()
    print("Sliding windows:")
    print(windows)
    print()
    print(f"Shares memory with original: {np.shares_memory(x, windows)}")
    
    print("\n" + "="*50)


def main():
    """Run all visualizations."""
    print("\n" + "="*60)
    print("Topic 01: Tensor Operations Visualization")
    print("="*60 + "\n")
    
    # 1. View vs Copy
    visualize_view_vs_copy()
    
    # 2. Strides
    arr = np.arange(12).reshape(3, 4)
    visualize_strides(arr)
    
    # 3. Sliding window
    sliding_window_demo()
    
    # 4. Graphical visualizations (if matplotlib available)
    try:
        # Memory layout
        fig1 = visualize_memory_layout(arr, "2D Array Memory Layout")
        
        # Broadcasting
        a = np.array([[1], [2], [3]])  # (3, 1)
        b = np.array([10, 20, 30])     # (3,)
        fig2 = visualize_broadcasting(a, b)
        
        plt.show()
        print("\n✅ Visualizations complete!")
        
    except Exception as e:
        print(f"Graphical visualization skipped: {e}")
        print("Text-based visualizations completed successfully.")


if __name__ == "__main__":
    main()

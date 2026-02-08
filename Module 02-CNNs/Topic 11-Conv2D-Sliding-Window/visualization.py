"""
Topic 11: Conv2D Visualization

Interactive visualization of convolution operations.
Generates HTML files showing sliding window behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def visualize_convolution_step(input_img, kernel, output, step, save_path=None):
    """
    Visualize a single step of convolution.
    
    Args:
        input_img: Input image (H, W)
        kernel: Kernel (K, K)
        output: Output accumulator
        step: Current step number (0-indexed)
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    K = kernel.shape[0]
    H, W = input_img.shape
    H_out = H - K + 1
    W_out = W - K + 1
    
    # Current position
    h_idx = step // W_out
    w_idx = step % W_out
    
    # Plot input with highlighted window
    ax1 = axes[0]
    ax1.imshow(input_img, cmap='viridis', aspect='auto')
    rect = patches.Rectangle((w_idx - 0.5, h_idx - 0.5), K, K, 
                             linewidth=3, edgecolor='red', facecolor='none')
    ax1.add_patch(rect)
    ax1.set_title(f'Input\nWindow at ({h_idx}, {w_idx})')
    ax1.set_xlabel('Width')
    ax1.set_ylabel('Height')
    
    # Plot kernel
    ax2 = axes[1]
    im = ax2.imshow(kernel, cmap='RdBu_r', aspect='auto')
    ax2.set_title('Kernel')
    plt.colorbar(im, ax=ax2)
    
    # Plot output accumulator
    ax3 = axes[2]
    im = ax3.imshow(output, cmap='viridis', aspect='auto', vmin=0)
    # Highlight current position
    if h_idx < output.shape[0] and w_idx < output.shape[1]:
        ax3.scatter([w_idx], [h_idx], c='red', s=200, marker='x', linewidths=3)
    ax3.set_title(f'Output (Step {step+1}/{H_out*W_out})')
    plt.colorbar(im, ax=ax3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_convolution_animation(input_img, kernel, output_dir='conv_frames'):
    """
    Create frames for convolution animation.
    
    Args:
        input_img: Input image (H, W)
        kernel: Kernel (K, K)
        output_dir: Directory to save frames
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    K = kernel.shape[0]
    H, W = input_img.shape
    H_out = H - K + 1
    W_out = W - K + 1
    
    # Initialize output
    output = np.zeros((H_out, W_out))
    
    # Generate frames
    step = 0
    for h in range(H_out):
        for w in range(W_out):
            # Compute convolution for this position
            window = input_img[h:h+K, w:w+K]
            output[h, w] = np.sum(window * kernel)
            
            # Save frame
            frame_path = output_path / f'frame_{step:03d}.png'
            visualize_convolution_step(input_img, kernel, output, step, frame_path)
            step += 1
    
    print(f"Created {step} frames in {output_dir}/")
    print(f"To create GIF: convert -delay 20 -loop 0 {output_dir}/frame_*.png conv_animation.gif")


def plot_kernel_weights(kernel, title="Kernel Weights", save_path=None):
    """
    Visualize kernel weights as a heatmap.
    
    Args:
        kernel: Kernel array (K, K) or (C_out, C_in, K, K)
        title: Plot title
        save_path: Path to save figure
    """
    if kernel.ndim == 4:
        # Multi-channel kernel, show first output channel, first input channel
        kernel = kernel[0, 0]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(kernel, cmap='RdBu_r', aspect='equal')
    
    # Add text annotations
    K = kernel.shape[0]
    for i in range(K):
        for j in range(K):
            text = ax.text(j, i, f'{kernel[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title(title)
    ax.set_xlabel('Kernel Width')
    ax.set_ylabel('Kernel Height')
    plt.colorbar(im, ax=ax)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compare_convolutions(input_img, kernel, implementations):
    """
    Compare different convolution implementations.
    
    Args:
        input_img: Input image (1, 1, H, W)
        kernel: Kernel (1, 1, K, K)
        implementations: Dict of {name: function}
    """
    fig, axes = plt.subplots(2, len(implementations) + 1, figsize=(5*(len(implementations)+1), 10))
    
    # Input
    axes[0, 0].imshow(input_img[0, 0], cmap='viridis')
    axes[0, 0].set_title('Input')
    axes[1, 0].axis('off')
    
    # Kernel
    axes[0, 0].text(-0.5, -0.5, f'Kernel:\n{kernel[0, 0]}', 
                   transform=axes[0, 0].transAxes, fontsize=8)
    
    # Run each implementation
    for idx, (name, conv_fn) in enumerate(implementations.items(), 1):
        start = time.time()
        output = conv_fn(input_img, kernel, stride=1, padding=0)
        elapsed = time.time() - start
        
        # Plot output
        axes[0, idx].imshow(output[0, 0], cmap='viridis')
        axes[0, idx].set_title(f'{name}\nTime: {elapsed:.4f}s')
        
        # Plot difference from first implementation
        if idx > 1:
            first_output = list(implementations.values())[0](input_img, kernel, stride=1, padding=0)
            diff = np.abs(output[0, 0] - first_output[0, 0])
            im = axes[1, idx].imshow(diff, cmap='hot')
            axes[1, idx].set_title(f'Difference\nMax: {np.max(diff):.2e}')
            plt.colorbar(im, ax=axes[1, idx])
        else:
            axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def create_interactive_html(save_path='assets/conv_visualization.html'):
    """
    Create an interactive HTML visualization of convolution.
    
    This generates a standalone HTML file with JavaScript that demonstrates
    the sliding window behavior interactively.
    """
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Conv2D Interactive Visualization</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .container {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 30px;
        }
        .panel {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .panel h3 {
            margin-top: 0;
            color: #555;
        }
        .grid {
            display: grid;
            gap: 2px;
            margin: 10px 0;
        }
        .cell {
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid #ddd;
            font-size: 12px;
            font-weight: bold;
            transition: all 0.3s;
        }
        .cell.input { background-color: #e3f2fd; }
        .cell.kernel { background-color: #ffebee; }
        .cell.output { background-color: #e8f5e9; }
        .cell.highlight { 
            background-color: #ff9800; 
            transform: scale(1.1);
            z-index: 10;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        button {
            padding: 10px 20px;
            margin: 0 5px;
            font-size: 16px;
            cursor: pointer;
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
        }
        button:hover {
            background-color: #1976D2;
        }
        button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .info {
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            background: #fff3cd;
            border-radius: 4px;
        }
        .formula {
            font-family: 'Courier New', monospace;
            background: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <h1>🔍 Conv2D Sliding Window Visualization</h1>
    
    <div class="info">
        <strong>How it works:</strong> The kernel slides over the input image. At each position, 
        we multiply the kernel values with the corresponding input values and sum them up.
        <br><br>
        <div class="formula">
            output[i,j] = Σ<sub>m</sub> Σ<sub>n</sub> input[i+m, j+n] × kernel[m, n]
        </div>
    </div>
    
    <div class="controls">
        <button id="prevBtn" onclick="previousStep()">← Previous</button>
        <button id="playBtn" onclick="togglePlay()">▶ Play</button>
        <button id="nextBtn" onclick="nextStep()">Next →</button>
        <button id="resetBtn" onclick="reset()">Reset</button>
    </div>
    
    <div class="container">
        <div class="panel">
            <h3>Input Image (5×5)</h3>
            <div id="inputGrid" class="grid"></div>
        </div>
        
        <div class="panel">
            <h3>Kernel (3×3)</h3>
            <div id="kernelGrid" class="grid"></div>
        </div>
        
        <div class="panel">
            <h3>Output (3×3)</h3>
            <div id="outputGrid" class="grid"></div>
        </div>
    </div>
    
    <div class="info">
        <strong>Current Step:</strong> <span id="stepInfo">0 / 9</span>
        <br>
        <strong>Position:</strong> <span id="posInfo">(0, 0)</span>
        <br>
        <strong>Calculation:</strong> <span id="calcInfo">-</span>
    </div>

    <script>
        // Define input, kernel, and expected output
        const input = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25]
        ];
        
        const kernel = [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ];
        
        const output = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ];
        
        let currentStep = 0;
        let isPlaying = false;
        let playInterval;
        
        const totalSteps = 9; // 3x3 output
        const outputSize = 3;
        
        function createGrid(elementId, data, type) {
            const grid = document.getElementById(elementId);
            const rows = data.length;
            const cols = data[0].length;
            
            grid.style.gridTemplateColumns = `repeat(${cols}, 40px)`;
            grid.innerHTML = '';
            
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    const cell = document.createElement('div');
                    cell.className = `cell ${type}`;
                    cell.textContent = data[i][j];
                    cell.id = `${type}_${i}_${j}`;
                    grid.appendChild(cell);
                }
            }
        }
        
        function updateHighlight() {
            // Clear previous highlights
            document.querySelectorAll('.cell').forEach(cell => {
                cell.classList.remove('highlight');
            });
            
            // Calculate position
            const outRow = Math.floor(currentStep / outputSize);
            const outCol = currentStep % outputSize;
            
            // Highlight input window
            for (let i = 0; i < 3; i++) {
                for (let j = 0; j < 3; j++) {
                    const inRow = outRow + i;
                    const inCol = outCol + j;
                    const cell = document.getElementById(`input_${inRow}_${inCol}`);
                    if (cell) cell.classList.add('highlight');
                }
            }
            
            // Highlight kernel
            for (let i = 0; i < 3; i++) {
                for (let j = 0; j < 3; j++) {
                    const cell = document.getElementById(`kernel_${i}_${j}`);
                    if (cell) cell.classList.add('highlight');
                }
            }
            
            // Highlight output position
            const outCell = document.getElementById(`output_${outRow}_${outCol}`);
            if (outCell) outCell.classList.add('highlight');
            
            // Calculate and display
            let sum = 0;
            let calc = '';
            for (let i = 0; i < 3; i++) {
                for (let j = 0; j < 3; j++) {
                    const inVal = input[outRow + i][outCol + j];
                    const kVal = kernel[i][j];
                    sum += inVal * kVal;
                    if (calc) calc += ' + ';
                    calc += `${inVal}×${kVal}`;
                }
            }
            calc += ` = ${sum}`;
            
            // Update output
            output[outRow][outCol] = sum;
            const outCellContent = document.getElementById(`output_${outRow}_${outCol}`);
            if (outCellContent) outCellContent.textContent = sum;
            
            // Update info
            document.getElementById('stepInfo').textContent = `${currentStep + 1} / ${totalSteps}`;
            document.getElementById('posInfo').textContent = `(${outRow}, ${outCol})`;
            document.getElementById('calcInfo').textContent = calc;
            
            // Update buttons
            document.getElementById('prevBtn').disabled = currentStep === 0;
            document.getElementById('nextBtn').disabled = currentStep === totalSteps - 1;
        }
        
        function nextStep() {
            if (currentStep < totalSteps - 1) {
                currentStep++;
                updateHighlight();
            }
        }
        
        function previousStep() {
            if (currentStep > 0) {
                currentStep--;
                // Reset output values ahead
                for (let i = currentStep + 1; i < totalSteps; i++) {
                    const row = Math.floor(i / outputSize);
                    const col = i % outputSize;
                    output[row][col] = 0;
                    const cell = document.getElementById(`output_${row}_${col}`);
                    if (cell) cell.textContent = '0';
                }
                updateHighlight();
            }
        }
        
        function togglePlay() {
            const btn = document.getElementById('playBtn');
            if (isPlaying) {
                clearInterval(playInterval);
                btn.textContent = '▶ Play';
                isPlaying = false;
            } else {
                if (currentStep >= totalSteps - 1) {
                    reset();
                }
                playInterval = setInterval(() => {
                    if (currentStep < totalSteps - 1) {
                        nextStep();
                    } else {
                        togglePlay();
                    }
                }, 1000);
                btn.textContent = '⏸ Pause';
                isPlaying = true;
            }
        }
        
        function reset() {
            currentStep = 0;
            // Clear output
            for (let i = 0; i < 3; i++) {
                for (let j = 0; j < 3; j++) {
                    output[i][j] = 0;
                }
            }
            // Recreate grids
            init();
            if (isPlaying) togglePlay();
        }
        
        function init() {
            createGrid('inputGrid', input, 'input');
            createGrid('kernelGrid', kernel, 'kernel');
            createGrid('outputGrid', output, 'output');
            updateHighlight();
        }
        
        // Initialize
        init();
    </script>
</body>
</html>'''
    
    # Ensure assets directory exists
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write HTML file
    with open(save_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Interactive visualization saved to: {save_path}")
    print(f"Open this file in a web browser to explore Conv2D interactively.")


if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("Conv2D Visualization Generator")
    print("=" * 60)
    
    # Create example data
    np.random.seed(42)
    
    # Simple example for animation
    input_small = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])
    
    kernel_edge = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])
    
    # Create output directory
    output_dir = Path(__file__).parent / "conv_visuals"
    output_dir.mkdir(exist_ok=True)
    
    print("\n1. Creating kernel visualization...")
    plot_kernel_weights(kernel_edge, "Vertical Edge Detector", 
                       output_dir / "kernel_edge.png")
    print("✅ Saved: kernel_edge.png")
    
    print("\n2. Creating interactive HTML visualization...")
    create_interactive_html(Path(__file__).parent / "assets" / "conv_visualization.html")
    
    print("\n3. Creating convolution animation frames...")
    create_convolution_animation(input_small, kernel_edge, 
                                 output_dir / "animation_frames")
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"\nFiles saved to: {output_dir}/")
    print("\nTo view:")
    print("  - Open assets/conv_visualization.html in a web browser")
    print("  - View PNG files in conv_visuals/")
    print("=" * 60)

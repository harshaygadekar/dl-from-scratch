#!/usr/bin/env python3
"""
Visualization Tools for Topic 03: Optimizers

Interactive visualizations to help understand:
- Optimizer convergence paths
- Loss landscape navigation
- Learning rate effects
"""

import math
import sys
from pathlib import Path
from typing import List, Tuple

# Add solutions to path
sys.path.insert(0, str(Path(__file__).parent / "solutions"))

try:
    from level02_vectorized import SGD, Adam, AdamW, Value
except ImportError:
    from level01_naive import SGD, Value
    Adam = None
    AdamW = None


def create_loss_landscape_ascii():
    """
    ASCII visualization of a loss landscape with optimizer paths.
    """
    print("\n" + "="*60)
    print("Loss Landscape Visualization (ASCII)")
    print("="*60)
    
    # Simple 1D visualization: f(x) = x²
    print("\nf(x) = x² - Finding the minimum")
    print("-" * 50)
    
    scale = 5.0
    height = 10
    width = 40
    
    def f(x):
        return x ** 2
    
    # Draw landscape
    print("\nLoss")
    print("  ^")
    for row in range(height, -1, -1):
        y = row * (scale / height)
        line = "  |"
        for col in range(width):
            x = (col - width/2) * (2 * scale / width)
            fx = f(x)
            if abs(fx - y * scale) < scale / height:
                line += "*"
            elif fx < y * scale:
                line += " "
            else:
                line += " "
        print(line)
    print("  +" + "-" * width + "-> x")
    print("  " + " " * (width//2 - 2) + "min")


def compare_optimizers_1d():
    """
    Compare optimizer convergence on 1D problem.
    """
    print("\n" + "="*60)
    print("Optimizer Comparison: f(x) = x²")
    print("="*60)
    
    # Settings
    x_init = 5.0
    lr = 0.1
    steps = 15
    
    optimizers = [("SGD", lambda p: SGD(p, lr=lr))]
    
    if Adam is not None:
        optimizers.append(("Adam", lambda p: Adam(p, lr=0.5)))
    
    try:
        optimizers.append(("SGD+Momentum", lambda p: SGD(p, lr=lr, momentum=0.9)))
    except TypeError:
        pass
    
    results = {}
    
    for name, opt_fn in optimizers:
        x = Value(x_init)
        opt = opt_fn([x])
        
        trajectory = [x.data]
        for _ in range(steps):
            loss = x ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            trajectory.append(x.data)
        
        results[name] = trajectory
    
    # ASCII plot
    print("\nTrajectory (x values over steps):")
    print("-" * 50)
    
    min_x = -1
    max_x = x_init + 0.5
    width = 50
    
    for step in range(steps + 1):
        line = f"Step {step:2d}: "
        
        for name, traj in results.items():
            x_val = traj[step]
            # Map to column position
            col = int((x_val - min_x) / (max_x - min_x) * (width - 1))
            col = max(0, min(width - 1, col))
            
            marker = name[0]  # First letter
            line_chars = list(" " * width)
            line_chars[col] = marker
            
            # Add to display (simplified - just show one)
            if name == list(results.keys())[0]:
                break
        
        print(line + "".join(line_chars))
    
    # Table format
    print("\n" + "-" * 50)
    print(f"{'Step':<6}" + "".join(f"{name:<12}" for name in results.keys()))
    print("-" * 50)
    
    for step in range(min(6, steps + 1)):
        row = f"{step:<6}"
        for name, traj in results.items():
            row += f"{traj[step]:<12.4f}"
        print(row)
    
    print("...")
    row = f"{steps:<6}"
    for name, traj in results.items():
        row += f"{traj[steps]:<12.4f}"
    print(row)


def visualize_momentum():
    """
    Visualize how momentum accumulates.
    """
    print("\n" + "="*60)
    print("Momentum Visualization")
    print("="*60)
    
    print("""
Momentum helps in two ways:
1. Accelerates in consistent direction
2. Dampens oscillations

Without momentum (oscillation):
    ←→←→←→←→  (bouncing between walls)
    
With momentum (smooth progress):
    ────────→  (steady movement)

Example: Narrow valley
""")
    
    # Demonstrate with numbers
    x = Value(5.0)
    try:
        opt = SGD([x], lr=0.1, momentum=0.9)
        
        print("With momentum=0.9:")
        print("-" * 40)
        print(f"{'Step':<6}{'x':<10}{'Velocity':<12}{'Gradient':<10}")
        print("-" * 40)
        
        for step in range(10):
            loss = x ** 2
            opt.zero_grad()
            loss.backward()
            
            vel = opt.velocity[0] if opt.velocity[0] else 0
            print(f"{step:<6}{x.data:<10.4f}{vel:<12.4f}{x.grad:<10.4f}")
            
            opt.step()
    except TypeError:
        print("(Momentum not implemented in current version)")


def visualize_adam_adaptation():
    """
    Visualize Adam's adaptive learning rates.
    """
    if Adam is None:
        print("\n(Adam not available for visualization)")
        return
    
    print("\n" + "="*60)
    print("Adam Adaptive Learning Rates")
    print("="*60)
    
    print("""
Adam maintains two running averages:
- m: exponential average of gradients (like momentum)
- v: exponential average of squared gradients (scaling)

Effective update: lr * m / sqrt(v)

Large v → smaller steps (noisy or inconsistent gradients)
Small v → larger steps (consistent gradients)
""")
    
    x = Value(5.0)
    opt = Adam([x], lr=0.5)
    
    print(f"{'Step':<6}{'x':<10}{'m':<12}{'v':<12}{'Effective LR'}")
    print("-" * 55)
    
    for step in range(10):
        loss = x ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # Get internal state
        m = opt.m[0]
        v = opt.v[0]
        
        # Bias correction
        t = opt.t
        m_hat = m / (1 - 0.9 ** t)
        v_hat = v / (1 - 0.999 ** t)
        
        # Effective learning rate
        eff_lr = 0.5 / (math.sqrt(v_hat) + 1e-8)
        
        print(f"{step:<6}{x.data:<10.4f}{m:<12.6f}{v:<12.6f}{eff_lr:<.4f}")


def learning_rate_comparison():
    """
    Show effect of different learning rates.
    """
    print("\n" + "="*60)
    print("Learning Rate Comparison")
    print("="*60)
    
    learning_rates = [0.01, 0.1, 0.5, 1.0, 2.0]
    steps = 10
    
    print(f"\nf(x) = x², starting at x=5.0")
    print("-" * 50)
    print(f"{'LR':<8}" + "".join(f"Step {i:<5}" for i in range(steps)))
    print("-" * 50)
    
    for lr in learning_rates:
        x = Value(5.0)
        opt = SGD([x], lr=lr)
        
        row = f"{lr:<8}"
        for step in range(steps):
            loss = x ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            val = x.data
            if abs(val) > 1000:
                row += "DIV    "
            else:
                row += f"{val:<7.2f}"
        
        print(row)
    
    print("-" * 50)
    print("DIV = diverged (value > 1000)")
    print("\nNote: LR=2.0 causes divergence! LR < 1/L is required.")


def main():
    """Run all visualizations."""
    print("\n" + "="*60)
    print("Topic 03: Optimizer Visualization Demo")
    print("="*60)
    
    create_loss_landscape_ascii()
    compare_optimizers_1d()
    visualize_momentum()
    visualize_adam_adaptation()
    learning_rate_comparison()
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
    print("""
Key Takeaways:
1. SGD is simple but can be slow
2. Momentum accelerates convergence in consistent directions
3. Adam adapts learning rates per-parameter
4. Learning rate choice is critical - too high = divergence
""")


if __name__ == "__main__":
    main()

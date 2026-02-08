"""
Topic 03: Level 01 - Naive Optimizers (SGD Only)

The simplest possible optimizer implementation.
Focuses on clarity and understanding the core concept.

Goal: Understand the basic parameter update mechanism.
"""

import importlib.util
from pathlib import Path


def _load_autograd_value():
    """Load Topic 02 Value class without colliding with local module names."""
    root = Path(__file__).resolve().parents[3]
    autograd_solutions = (
        root / "Module 00-Foundations" / "Topic 02-Autograd-Engine" / "solutions"
    )
    candidates = [
        autograd_solutions / "level02_vectorized.py",
        autograd_solutions / "level01_naive.py",
        root / "utils" / "autograd_stub.py",
    ]

    for idx, module_path in enumerate(candidates):
        if not module_path.exists():
            continue
        spec = importlib.util.spec_from_file_location(
            f"_topic03_autograd_{idx}", module_path
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "Value"):
            return module.Value

    raise ImportError("Could not load Value class for optimizers")


Value = _load_autograd_value()


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, parameters, lr=0.01):
        """
        Args:
            parameters: Iterable of Value objects to optimize
            lr: Learning rate
        """
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        """Update all parameters. Must be implemented by subclasses."""
        raise NotImplementedError

    def zero_grad(self):
        """Reset all gradients to zero."""
        for p in self.parameters:
            p.grad = 0.0


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    Update rule:
        θ = θ - lr × ∂L/∂θ

    Example:
        x = Value(5.0)
        optimizer = SGD([x], lr=0.1)

        for _ in range(100):
            loss = x ** 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    """

    def step(self):
        """Apply gradient descent update."""
        for p in self.parameters:
            p.data -= self.lr * p.grad


# ==================== Demo ====================


def demo():
    print("=" * 50)
    print("Level 01: Vanilla SGD Demo")
    print("=" * 50)

    # Problem: Minimize f(x) = x²
    print("\nProblem: Minimize f(x) = x²")
    print("Starting at x = 5.0")
    print("-" * 40)

    x = Value(5.0)
    optimizer = SGD([x], lr=0.1)

    history = []
    for step in range(20):
        loss = x**2

        history.append((x.data, loss.data))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step < 10 or step % 5 == 0:
            print(f"Step {step:2d}: x = {x.data:8.4f}, loss = {loss.data:10.4f}")

    print("-" * 40)
    print(f"Final: x = {x.data:.6f} (expected: 0)")

    # 2D problem
    print("\n" + "=" * 50)
    print("2D Problem: Minimize f(x,y) = x² + y²")
    print("Starting at (3.0, 4.0)")
    print("-" * 40)

    x = Value(3.0)
    y = Value(4.0)
    optimizer = SGD([x, y], lr=0.1)

    for step in range(20):
        loss = x**2 + y**2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step < 5 or step % 5 == 0:
            print(
                f"Step {step:2d}: (x,y) = ({x.data:6.3f}, {y.data:6.3f}), loss = {loss.data:8.4f}"
            )

    print("-" * 40)
    print(f"Final: ({x.data:.6f}, {y.data:.6f})")

    print("\n✅ SGD working correctly!")


if __name__ == "__main__":
    demo()

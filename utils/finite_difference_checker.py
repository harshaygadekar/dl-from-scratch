"""
Finite Difference Gradient Checker

Validates autograd implementations by comparing against numerical gradients.
This is the gold standard test for checking if your backprop is correct.

Usage:
    from utils.finite_difference_checker import check_gradients, numerical_gradient
    
    def f(x):
        return x ** 2
    
    is_correct = check_gradients(f, x=2.0, analytic_grad=4.0)
"""

import numpy as np
from typing import Callable, Union


def numerical_gradient(f: Callable, x: Union[float, np.ndarray], 
                       eps: float = 1e-5) -> Union[float, np.ndarray]:
    """
    Compute numerical gradient using central difference.
    
    The central difference formula is:
        f'(x) ≈ [f(x + eps) - f(x - eps)] / (2 * eps)
    
    This is more accurate than forward difference because errors cancel.
    
    Args:
        f: Function to differentiate (must return a scalar)
        x: Point at which to evaluate gradient
        eps: Step size for finite difference (default: 1e-5)
    
    Returns:
        Numerical gradient approximation (same shape as x)
    """
    if isinstance(x, (int, float)):
        return (f(x + eps) - f(x - eps)) / (2 * eps)
    
    # Handle arrays element by element
    x = np.array(x, dtype=np.float64)
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        original = x[idx]
        
        # f(x + eps)
        x[idx] = original + eps
        fx_plus = f(x)
        
        # f(x - eps)
        x[idx] = original - eps
        fx_minus = f(x)
        
        # Central difference
        grad[idx] = (fx_plus - fx_minus) / (2 * eps)
        
        # Restore original value
        x[idx] = original
        
        it.iternext()
    
    return grad


def check_gradients(f: Callable, x: Union[float, np.ndarray], 
                    analytic_grad: Union[float, np.ndarray],
                    eps: float = 1e-5, tol: float = 1e-5,
                    verbose: bool = True) -> bool:
    """
    Check if analytic gradient matches numerical gradient.
    
    Args:
        f: Function to check (must return a scalar)
        x: Input point
        analytic_grad: Your computed gradient
        eps: Step size for finite difference
        tol: Tolerance for relative error
        verbose: Print detailed output
    
    Returns:
        True if gradients match within tolerance
    """
    numerical_grad = numerical_gradient(f, x, eps)
    
    # Compute relative error
    if isinstance(analytic_grad, (int, float)):
        analytic_grad = float(analytic_grad)
        numerical_grad = float(numerical_grad)
        error = abs(analytic_grad - numerical_grad)
        denominator = max(abs(numerical_grad), abs(analytic_grad), 1e-8)
        relative_error = error / denominator
    else:
        analytic_grad = np.array(analytic_grad)
        error = np.linalg.norm(analytic_grad - numerical_grad)
        denominator = max(np.linalg.norm(numerical_grad), np.linalg.norm(analytic_grad), 1e-8)
        relative_error = error / denominator
    
    passed = relative_error < tol
    
    if verbose:
        print(f"Numerical gradient:  {numerical_grad}")
        print(f"Analytic gradient:   {analytic_grad}")
        print(f"Absolute error:      {error:.2e}")
        print(f"Relative error:      {relative_error:.2e}")
        print(f"Tolerance:           {tol:.2e}")
        print(f"Status: {'✅ PASSED' if passed else '❌ FAILED'}")
    
    return passed


def gradient_check_weights(forward_fn: Callable, 
                           x: np.ndarray, 
                           W: np.ndarray,
                           dW_analytic: np.ndarray,
                           eps: float = 1e-5,
                           tol: float = 1e-5) -> bool:
    """
    Check gradients for weight matrix in a layer.
    
    Args:
        forward_fn: Function that takes (x, W) and returns scalar loss
        x: Input data
        W: Weight matrix
        dW_analytic: Your computed gradient for W
        eps: Finite difference step
        tol: Tolerance
    
    Returns:
        True if all gradients match
    """
    print(f"Checking gradients for weight matrix of shape {W.shape}")
    
    # Compute numerical gradient for W
    W = W.astype(np.float64)
    dW_numerical = np.zeros_like(W)
    
    it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        original = W[idx]
        
        W[idx] = original + eps
        loss_plus = forward_fn(x, W)
        
        W[idx] = original - eps
        loss_minus = forward_fn(x, W)
        
        dW_numerical[idx] = (loss_plus - loss_minus) / (2 * eps)
        W[idx] = original
        
        it.iternext()
    
    # Compare
    error = np.linalg.norm(dW_analytic - dW_numerical)
    denominator = max(np.linalg.norm(dW_numerical), np.linalg.norm(dW_analytic), 1e-8)
    relative_error = error / denominator
    
    passed = relative_error < tol
    
    print(f"Relative error: {relative_error:.2e}")
    print(f"Status: {'✅ PASSED' if passed else '❌ FAILED'}")
    
    if not passed:
        print("\nSample differences (first 5):")
        diff = np.abs(dW_analytic - dW_numerical)
        flat_diff = diff.flatten()
        sorted_indices = np.argsort(flat_diff)[::-1]
        for i in range(min(5, len(sorted_indices))):
            idx = np.unravel_index(sorted_indices[i], diff.shape)
            print(f"  [{idx}]: analytic={dW_analytic[idx]:.6f}, numerical={dW_numerical[idx]:.6f}")
    
    return passed


if __name__ == "__main__":
    print("Testing finite_difference_checker...\n")
    print("="*50)
    
    # Test 1: Simple quadratic
    print("\nTest 1: f(x) = x² at x=3")
    print("-"*30)
    def f1(x):
        return x ** 2
    
    passed1 = check_gradients(f1, 3.0, 6.0)  # d/dx(x²) = 2x = 6
    
    # Test 2: Sigmoid
    print("\nTest 2: f(x) = sigmoid(x) at x=0")
    print("-"*30)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def f2(x):
        return sigmoid(x)
    
    # Derivative of sigmoid at 0 is 0.25
    passed2 = check_gradients(f2, 0.0, 0.25)
    
    # Test 3: Vector input
    print("\nTest 3: f(x) = sum(x²) at x=[1, 2, 3]")
    print("-"*30)
    def f3(x):
        return np.sum(x ** 2)
    
    x = np.array([1.0, 2.0, 3.0])
    analytic = 2 * x  # Gradient of sum(x²) is 2x
    passed3 = check_gradients(f3, x, analytic)
    
    # Test 4: Matrix input
    print("\nTest 4: f(W) = sum(W²) for 2x2 matrix")
    print("-"*30)
    def f4(W):
        return np.sum(W ** 2)
    
    W = np.array([[1.0, 2.0], [3.0, 4.0]])
    analytic_W = 2 * W
    passed4 = check_gradients(f4, W, analytic_W)
    
    print("\n" + "="*50)
    all_passed = passed1 and passed2 and passed3 and passed4
    print(f"Overall: {'✅ All tests passed!' if all_passed else '❌ Some tests failed'}")

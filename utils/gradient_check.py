"""
Gradient Check Utility for Neural Network Layers

Compares analytic gradients against numerical gradients for:
- Weight matrices
- Bias vectors
- Layer inputs

This is essential for debugging backprop implementations.
"""

import numpy as np
from typing import Callable, Dict, Tuple, Optional


def gradient_check_layer(forward_fn: Callable, 
                         backward_fn: Callable,
                         inputs: Dict[str, np.ndarray],
                         params: Dict[str, np.ndarray],
                         eps: float = 1e-5,
                         tol: float = 1e-5,
                         verbose: bool = True) -> Dict[str, bool]:
    """
    Check gradients for all parameters in a layer.
    
    Args:
        forward_fn: Forward pass function
                    Signature: forward_fn(inputs, params) -> output (scalar or array)
        backward_fn: Backward pass function  
                    Signature: backward_fn(inputs, params, grad_out) -> dict of gradients
        inputs: Dictionary of input arrays (e.g., {'x': x_data})
        params: Dictionary of parameter arrays (e.g., {'W': W, 'b': b})
        eps: Finite difference step size
        tol: Tolerance for relative error
        verbose: Print detailed output
    
    Returns:
        Dictionary mapping param names to pass/fail status
    """
    results = {}
    
    # Get analytic gradients
    output = forward_fn(inputs, params)
    
    # Create grad_output (all ones for checking)
    if isinstance(output, np.ndarray):
        grad_output = np.ones_like(output)
    else:
        grad_output = 1.0
    
    analytic_grads = backward_fn(inputs, params, grad_output)
    
    # Check each parameter
    for name, param in params.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"Checking gradient for: {name} (shape: {param.shape})")
            print("="*50)
        
        # Compute numerical gradient
        numerical_grad = np.zeros_like(param, dtype=np.float64)
        param_copy = param.astype(np.float64)
        
        it = np.nditer(param_copy, flags=['multi_index'], op_flags=['readwrite'])
        
        while not it.finished:
            idx = it.multi_index
            original = param_copy[idx]
            
            # f(x + eps)
            param_copy[idx] = original + eps
            params_plus = {**params, name: param_copy.copy()}
            output_plus = forward_fn(inputs, params_plus)
            loss_plus = np.sum(output_plus * grad_output) if isinstance(output_plus, np.ndarray) else output_plus
            
            # f(x - eps)
            param_copy[idx] = original - eps
            params_minus = {**params, name: param_copy.copy()}
            output_minus = forward_fn(inputs, params_minus)
            loss_minus = np.sum(output_minus * grad_output) if isinstance(output_minus, np.ndarray) else output_minus
            
            # Restore
            param_copy[idx] = original
            numerical_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            
            it.iternext()
        
        # Compare gradients
        analytic = analytic_grads[name]
        error = np.linalg.norm(analytic - numerical_grad)
        denominator = max(np.linalg.norm(numerical_grad), np.linalg.norm(analytic), 1e-8)
        relative_error = error / denominator
        
        passed = relative_error < tol
        results[name] = passed
        
        if verbose:
            print(f"Relative error: {relative_error:.2e}")
            print(f"Status: {'✅ PASSED' if passed else '❌ FAILED'}")
            
            if not passed:
                # Show biggest differences
                diff = np.abs(analytic - numerical_grad)
                max_idx = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"\nMax difference at index {max_idx}:")
                print(f"  Analytic:  {analytic[max_idx]:.6f}")
                print(f"  Numerical: {numerical_grad[max_idx]:.6f}")
    
    return results


def check_all_passed(results: Dict[str, bool]) -> bool:
    """Check if all gradient checks passed."""
    return all(results.values())


# Example usage with a simple linear layer
if __name__ == "__main__":
    print("Gradient Check Utility Demo\n")
    print("Testing a simple linear layer: y = Wx + b\n")
    
    # Define a simple linear layer
    def linear_forward(inputs, params):
        x = inputs['x']
        W = params['W']
        b = params['b']
        return x @ W + b
    
    def linear_backward(inputs, params, grad_output):
        x = inputs['x']
        W = params['W']
        
        # Analytic gradients
        dW = x.T @ grad_output
        db = np.sum(grad_output, axis=0)
        
        return {'W': dW, 'b': db}
    
    # Create test data
    np.random.seed(42)
    batch_size = 4
    in_features = 3
    out_features = 2
    
    inputs = {'x': np.random.randn(batch_size, in_features)}
    params = {
        'W': np.random.randn(in_features, out_features),
        'b': np.random.randn(out_features)
    }
    
    # Run gradient check
    results = gradient_check_layer(
        linear_forward, 
        linear_backward, 
        inputs, 
        params
    )
    
    print("\n" + "="*50)
    if check_all_passed(results):
        print("✅ All gradient checks passed!")
    else:
        print("❌ Some gradient checks failed!")
        for name, passed in results.items():
            print(f"  {name}: {'✅' if passed else '❌'}")

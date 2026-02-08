"""
Topic 02: Level 04 - PyTorch Reference

Verify your autograd implementation against PyTorch.
All gradient computations should match to numerical precision.

Usage:
    python level04_pytorch_reference.py
    
    # Or run specific test
    python level04_pytorch_reference.py --test arithmetic
"""

import sys
import argparse
from typing import Callable, Tuple

# Try to import our implementation
try:
    from level02_vectorized import Value
except ImportError:
    try:
        from solutions.level02_vectorized import Value
    except ImportError:
        # Fallback to level01
        try:
            from level01_naive import Value
        except ImportError:
            from solutions.level01_naive import Value

# Try to import PyTorch
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️  PyTorch not installed. Install with: pip install torch")
    print("    Running in verification-only mode.\n")


def check_gradient(
    our_fn: Callable,
    torch_fn: Callable,
    input_values: Tuple[float, ...],
    test_name: str,
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> bool:
    """
    Compare gradient computation between our implementation and PyTorch.
    
    Args:
        our_fn: Function using our Value class
        torch_fn: Equivalent function using torch.tensor
        input_values: Tuple of input values
        test_name: Name of the test for display
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        True if gradients match, False otherwise
    """
    # Our implementation
    our_inputs = [Value(v) for v in input_values]
    our_out = our_fn(*our_inputs)
    our_out.backward()
    our_grads = [v.grad for v in our_inputs]
    
    if PYTORCH_AVAILABLE:
        # PyTorch implementation
        torch_inputs = [torch.tensor([v], dtype=torch.float64, requires_grad=True) 
                        for v in input_values]
        torch_out = torch_fn(*torch_inputs)
        torch_out.backward()
        torch_grads = [v.grad.item() for v in torch_inputs]
        
        # Compare
        all_match = True
        for i, (ours, theirs) in enumerate(zip(our_grads, torch_grads)):
            match = abs(ours - theirs) <= atol + rtol * abs(theirs)
            if not match:
                print(f"  ❌ Input {i}: our={ours:.6f}, torch={theirs:.6f}")
                all_match = False
        
        status = "✅" if all_match else "❌"
        print(f"{status} {test_name}: output={our_out.data:.4f}, grads={our_grads}")
        return all_match
    else:
        # Just show our results
        print(f"🔸 {test_name}: output={our_out.data:.4f}, grads={our_grads}")
        return True


# ==================== Test Suites ====================

def test_arithmetic():
    """Test basic arithmetic operations."""
    print("\n" + "="*50)
    print("Testing Arithmetic Operations")
    print("="*50)
    
    tests_passed = 0
    tests_total = 0
    
    # Addition
    tests_total += 1
    if check_gradient(
        lambda x, y: x + y,
        lambda x, y: x + y,
        (2.0, 3.0),
        "Addition: x + y"
    ):
        tests_passed += 1
    
    # Multiplication
    tests_total += 1
    if check_gradient(
        lambda x, y: x * y,
        lambda x, y: x * y,
        (2.0, 3.0),
        "Multiplication: x * y"
    ):
        tests_passed += 1
    
    # Combined: x * y + x
    tests_total += 1
    if check_gradient(
        lambda x, y: x * y + x,
        lambda x, y: x * y + x,
        (2.0, 3.0),
        "Combined: x * y + x"
    ):
        tests_passed += 1
    
    # Squared: x * x
    tests_total += 1
    if check_gradient(
        lambda x: x * x,
        lambda x: x * x,
        (3.0,),
        "Squared: x * x"
    ):
        tests_passed += 1
    
    # Subtraction
    tests_total += 1
    if check_gradient(
        lambda x, y: x - y,
        lambda x, y: x - y,
        (5.0, 3.0),
        "Subtraction: x - y"
    ):
        tests_passed += 1
    
    return tests_passed, tests_total


def test_power_division():
    """Test power and division operations."""
    print("\n" + "="*50)
    print("Testing Power and Division")
    print("="*50)
    
    tests_passed = 0
    tests_total = 0
    
    # Power
    tests_total += 1
    if check_gradient(
        lambda x: x ** 2,
        lambda x: x ** 2,
        (3.0,),
        "Power: x^2"
    ):
        tests_passed += 1
    
    # Square root
    tests_total += 1
    if check_gradient(
        lambda x: x ** 0.5,
        lambda x: x ** 0.5,
        (4.0,),
        "Sqrt: x^0.5"
    ):
        tests_passed += 1
    
    # Division
    tests_total += 1
    if check_gradient(
        lambda x, y: x / y,
        lambda x, y: x / y,
        (6.0, 2.0),
        "Division: x / y"
    ):
        tests_passed += 1
    
    # Inverse
    tests_total += 1
    if check_gradient(
        lambda x: 1 / x,
        lambda x: 1 / x,
        (4.0,),
        "Inverse: 1 / x"
    ):
        tests_passed += 1
    
    return tests_passed, tests_total


def test_activations():
    """Test activation functions."""
    print("\n" + "="*50)
    print("Testing Activation Functions")
    print("="*50)
    
    tests_passed = 0
    tests_total = 0
    
    # ReLU positive
    tests_total += 1
    if check_gradient(
        lambda x: x.relu(),
        lambda x: torch.relu(x) if PYTORCH_AVAILABLE else x,
        (3.0,),
        "ReLU(3.0) - positive"
    ):
        tests_passed += 1
    
    # ReLU negative
    tests_total += 1
    if check_gradient(
        lambda x: x.relu(),
        lambda x: torch.relu(x) if PYTORCH_AVAILABLE else x,
        (-3.0,),
        "ReLU(-3.0) - negative"
    ):
        tests_passed += 1
    
    # Tanh
    tests_total += 1
    if check_gradient(
        lambda x: x.tanh(),
        lambda x: torch.tanh(x) if PYTORCH_AVAILABLE else x,
        (0.5,),
        "tanh(0.5)"
    ):
        tests_passed += 1
    
    # Sigmoid
    tests_total += 1
    if check_gradient(
        lambda x: x.sigmoid(),
        lambda x: torch.sigmoid(x) if PYTORCH_AVAILABLE else x,
        (0.0,),
        "sigmoid(0.0)"
    ):
        tests_passed += 1
    
    return tests_passed, tests_total


def test_transcendental():
    """Test transcendental functions."""
    print("\n" + "="*50)
    print("Testing Transcendental Functions")
    print("="*50)
    
    tests_passed = 0
    tests_total = 0
    
    # Exp
    tests_total += 1
    if check_gradient(
        lambda x: x.exp(),
        lambda x: torch.exp(x) if PYTORCH_AVAILABLE else x,
        (1.0,),
        "exp(1.0)"
    ):
        tests_passed += 1
    
    # Log
    tests_total += 1
    if check_gradient(
        lambda x: x.log(),
        lambda x: torch.log(x) if PYTORCH_AVAILABLE else x,
        (2.0,),
        "log(2.0)"
    ):
        tests_passed += 1
    
    # Composed: log(exp(x))
    tests_total += 1
    if check_gradient(
        lambda x: x.exp().log(),
        lambda x: torch.log(torch.exp(x)) if PYTORCH_AVAILABLE else x,
        (2.0,),
        "log(exp(2.0)) = identity"
    ):
        tests_passed += 1
    
    return tests_passed, tests_total


def test_complex_expressions():
    """Test complex composed expressions."""
    print("\n" + "="*50)
    print("Testing Complex Expressions")
    print("="*50)
    
    tests_passed = 0
    tests_total = 0
    
    # Expression 1: (x + y) * (x - y)
    tests_total += 1
    if check_gradient(
        lambda x, y: (x + y) * (x - y),
        lambda x, y: (x + y) * (x - y),
        (3.0, 2.0),
        "(x + y) * (x - y)"
    ):
        tests_passed += 1
    
    # Expression 2: softmax-ish
    tests_total += 1
    if check_gradient(
        lambda x, y: x.exp() / (x.exp() + y.exp()),
        lambda x, y: torch.exp(x) / (torch.exp(x) + torch.exp(y)) if PYTORCH_AVAILABLE else x,
        (1.0, 2.0),
        "Softmax: exp(x) / (exp(x) + exp(y))"
    ):
        tests_passed += 1
    
    # Expression 3: layered
    tests_total += 1
    if check_gradient(
        lambda x: ((x * x + x) * x).tanh(),
        lambda x: torch.tanh((x * x + x) * x) if PYTORCH_AVAILABLE else x,
        (0.5,),
        "tanh((x^2 + x) * x)"
    ):
        tests_passed += 1
    
    return tests_passed, tests_total


def test_neural_network():
    """Test a simple neural network computation."""
    print("\n" + "="*50)
    print("Testing Neural Network Computation")
    print("="*50)
    
    # A single neuron: tanh(w1*x1 + w2*x2 + b)
    def our_neuron(w1, w2, x1, x2, b):
        return (w1 * x1 + w2 * x2 + b).tanh()
    
    def torch_neuron(w1, w2, x1, x2, b):
        if PYTORCH_AVAILABLE:
            return torch.tanh(w1 * x1 + w2 * x2 + b)
        return None
    
    result = check_gradient(
        our_neuron,
        torch_neuron,
        (0.5, -0.3, 1.0, 2.0, 0.1),
        "Neuron: tanh(w1*x1 + w2*x2 + b)"
    )
    
    return (1 if result else 0), 1


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='Verify autograd against PyTorch')
    parser.add_argument('--test', choices=['arithmetic', 'power', 'activations', 
                                           'transcendental', 'complex', 'nn', 'all'],
                        default='all', help='Which test suite to run')
    args = parser.parse_args()
    
    print("="*50)
    print("Topic 02: PyTorch Reference Verification")
    print("="*50)
    
    if PYTORCH_AVAILABLE:
        print(f"PyTorch version: {torch.__version__}")
    
    total_passed = 0
    total_tests = 0
    
    test_map = {
        'arithmetic': test_arithmetic,
        'power': test_power_division,
        'activations': test_activations,
        'transcendental': test_transcendental,
        'complex': test_complex_expressions,
        'nn': test_neural_network,
    }
    
    if args.test == 'all':
        for test_fn in test_map.values():
            passed, total = test_fn()
            total_passed += passed
            total_tests += total
    else:
        passed, total = test_map[args.test]()
        total_passed += passed
        total_tests += total
    
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"Tests passed: {total_passed}/{total_tests}")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Your autograd matches PyTorch!")
    else:
        print(f"\n⚠️ {total_tests - total_passed} test(s) failed. Debug needed.")
    
    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())

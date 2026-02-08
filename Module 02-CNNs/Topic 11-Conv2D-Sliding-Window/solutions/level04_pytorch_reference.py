"""
Level 04: PyTorch Reference Implementation

Topic 11: Conv2D Sliding Window

This is the ground truth implementation using PyTorch.
Use it to verify your NumPy implementations are correct.

Note: This file requires PyTorch to be installed:
    pip install torch
"""

import numpy as np

def check_pytorch():
    """Check if PyTorch is available."""
    try:
        import torch
        import torch.nn as nn
        return True
    except ImportError:
        return False


def conv2d_pytorch(input_np, weight_np, bias_np=None, stride=1, padding=0):
    """
    Conv2D using PyTorch (ground truth).
    
    Args:
        input_np: NumPy array (N, C_in, H, W)
        weight_np: NumPy array (C_out, C_in, K, K)
        bias_np: NumPy array (C_out,) or None
        stride: int or tuple
        padding: int or tuple
    
    Returns:
        NumPy array (N, C_out, H_out, W_out)
    """
    import torch
    import torch.nn.functional as F
    
    # Convert to PyTorch tensors
    input_torch = torch.from_numpy(input_np)
    weight_torch = torch.from_numpy(weight_np)
    bias_torch = torch.from_numpy(bias_np) if bias_np is not None else None
    
    # Perform convolution
    output_torch = F.conv2d(input_torch, weight_torch, bias_torch, 
                            stride=stride, padding=padding)
    
    # Convert back to NumPy
    return output_torch.numpy()


def generate_golden_outputs():
    """
    Generate pre-computed outputs for testing.
    
    Run this to create reference outputs that the test suite can use.
    """
    if not check_pytorch():
        print("PyTorch not installed. Install with: pip install torch")
        return
    
    import torch
    
    print("Generating golden outputs for Conv2D tests...")
    print("=" * 60)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    test_cases = [
        {
            'name': 'basic_3x3',
            'input_shape': (2, 3, 5, 5),
            'weight_shape': (4, 3, 3, 3),
            'stride': 1,
            'padding': 0,
        },
        {
            'name': 'same_conv',
            'input_shape': (1, 1, 7, 7),
            'weight_shape': (2, 1, 3, 3),
            'stride': 1,
            'padding': 1,
        },
        {
            'name': 'strided',
            'input_shape': (1, 1, 8, 8),
            'weight_shape': (1, 1, 3, 3),
            'stride': 2,
            'padding': 1,
        },
        {
            'name': 'large_kernel',
            'input_shape': (1, 3, 32, 32),
            'weight_shape': (8, 3, 5, 5),
            'stride': 1,
            'padding': 2,
        },
        {
            'name': 'with_bias',
            'input_shape': (2, 4, 8, 8),
            'weight_shape': (6, 4, 3, 3),
            'stride': 1,
            'padding': 1,
            'use_bias': True,
        },
    ]
    
    golden_outputs = {}
    
    for test in test_cases:
        name = test['name']
        print(f"\nGenerating: {name}")
        
        # Create random inputs
        input_np = np.random.randn(*test['input_shape']).astype(np.float32)
        weight_np = np.random.randn(*test['weight_shape']).astype(np.float32)
        bias_np = None
        if test.get('use_bias', False):
            bias_np = np.random.randn(test['weight_shape'][0]).astype(np.float32)
        
        # Compute PyTorch output
        output_np = conv2d_pytorch(input_np, weight_np, bias_np, 
                                   test['stride'], test['padding'])
        
        # Store in dictionary
        golden_outputs[name] = {
            'input': input_np,
            'weight': weight_np,
            'bias': bias_np,
            'output': output_np,
            'stride': test['stride'],
            'padding': test['padding'],
        }
        
        print(f"  Input: {input_np.shape}")
        print(f"  Output: {output_np.shape}")
    
    # Save to file
    output_file = Path(__file__).parent / "golden_outputs.npy"
    np.save(output_file, golden_outputs)
    print(f"\n✅ Saved golden outputs to: {output_file}")
    
    return golden_outputs


def verify_implementation(conv_fn, name="Your Implementation"):
    """
    Verify a NumPy implementation against PyTorch.
    
    Args:
        conv_fn: Function with signature conv_fn(input, weight, bias, stride, padding)
        name: Name of implementation for printing
    """
    if not check_pytorch():
        print("❌ PyTorch not installed. Cannot verify.")
        print("   Install with: pip install torch")
        return False
    
    import torch
    
    print("=" * 60)
    print(f"Verification: {name}")
    print("=" * 60)
    
    np.random.seed(123)
    torch.manual_seed(123)
    
    test_cases = [
        ("Basic (stride=1, pad=0)", (2, 3, 8, 8), (4, 3, 3, 3), 1, 0),
        ("Same conv (stride=1, pad=1)", (2, 8, 16, 16), (16, 8, 3, 3), 1, 1),
        ("Strided (stride=2, pad=1)", (4, 16, 32, 32), (32, 16, 3, 3), 2, 1),
        ("Large kernel (5x5)", (2, 3, 32, 32), (8, 3, 5, 5), 1, 2),
        ("1x1 conv", (2, 64, 16, 16), (128, 64, 1, 1), 1, 0),
    ]
    
    all_passed = True
    
    for test_name, input_shape, weight_shape, stride, padding in test_cases:
        # Generate random data
        input_np = np.random.randn(*input_shape).astype(np.float32)
        weight_np = np.random.randn(*weight_shape).astype(np.float32)
        bias_np = np.random.randn(weight_shape[0]).astype(np.float32)
        
        # Compute with PyTorch
        expected = conv2d_pytorch(input_np, weight_np, bias_np, stride, padding)
        
        # Compute with implementation
        try:
            result = conv_fn(input_np, weight_np, bias_np, stride, padding)
        except Exception as e:
            print(f"\n❌ {test_name}: ERROR - {e}")
            all_passed = False
            continue
        
        # Check shape
        if expected.shape != result.shape:
            print(f"\n❌ {test_name}: Shape mismatch!")
            print(f"   Expected: {expected.shape}")
            print(f"   Got:      {result.shape}")
            all_passed = False
            continue
        
        # Check values
        max_diff = np.max(np.abs(expected - result))
        mean_diff = np.mean(np.abs(expected - result))
        
        if max_diff < 1e-4:
            print(f"✅ {test_name}: PASSED (max_diff={max_diff:.2e})")
        else:
            print(f"❌ {test_name}: FAILED (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
            all_passed = False
    
    if all_passed:
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("Your implementation matches PyTorch.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ SOME TESTS FAILED")
        print("Review the differences above.")
        print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    from pathlib import Path
    
    print("=" * 60)
    print("Conv2D Sliding Window - Level 04 (PyTorch Reference)")
    print("=" * 60)
    
    if not check_pytorch():
        print("\n❌ PyTorch is not installed!")
        print("\nTo install:")
        print("  pip install torch")
        print("\nFor CPU-only version:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("\nThis reference implementation requires PyTorch for verification.")
        exit(1)
    
    print("\n✅ PyTorch is available!")
    
    # Generate golden outputs
    print("\n" + "=" * 60)
    print("Generating Golden Outputs")
    print("=" * 60)
    golden = generate_golden_outputs()
    
    # Verify Level 1 implementation
    print("\n" + "=" * 60)
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        from level01_naive import conv2d_naive
        verify_implementation(conv2d_naive, "Level 01 (Naive)")
    except Exception as e:
        print(f"\n❌ Could not verify Level 01: {e}")
    
    # Verify Level 2
    try:
        from level02_vectorized import conv2d_vectorized
        verify_implementation(conv2d_vectorized, "Level 02 (Vectorized)")
    except Exception as e:
        print(f"\n❌ Could not verify Level 02: {e}")
    
    # Verify Level 3
    try:
        from level03_memory_efficient import conv2d_strided
        verify_implementation(conv2d_strided, "Level 03 (Memory-Efficient)")
    except Exception as e:
        print(f"\n❌ Could not verify Level 03: {e}")
    
    print("\n" + "=" * 60)
    print("Level 04 Complete!")
    print("\nGolden outputs saved. Use these for testing.")
    print("Run: python -m pytest tests/ -v")
    print("=" * 60)

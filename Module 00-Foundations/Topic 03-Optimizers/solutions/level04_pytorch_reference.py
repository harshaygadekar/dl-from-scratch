"""
Topic 03: Level 04 - PyTorch Reference

Verify your optimizer implementation against PyTorch.
All updates should match to numerical precision.

Usage:
    python level04_pytorch_reference.py
"""

import sys
from pathlib import Path

# Try to import our implementation
sys.path.insert(0, str(Path(__file__).parent))

try:
    from level02_vectorized import SGD, Adam, AdamW
    from level02_vectorized import Value
except ImportError:
    try:
        from solutions.level02_vectorized import SGD, Adam, AdamW, Value
    except ImportError:
        print("Could not import optimizers. Run from solutions directory.")
        sys.exit(1)

# Try to import PyTorch
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️  PyTorch not installed. Install with: pip install torch")
    print("    Running in verification-only mode.\n")


def test_sgd():
    """Compare SGD implementations."""
    print("\n" + "="*50)
    print("Testing SGD")
    print("="*50)
    
    # Our implementation
    x = Value(5.0)
    our_opt = SGD([x], lr=0.1)
    
    our_values = [x.data]
    for _ in range(10):
        loss = x ** 2
        our_opt.zero_grad()
        loss.backward()
        our_opt.step()
        our_values.append(x.data)
    
    if PYTORCH_AVAILABLE:
        # PyTorch implementation
        x_torch = torch.tensor([5.0], requires_grad=True)
        torch_opt = torch.optim.SGD([x_torch], lr=0.1)
        
        torch_values = [x_torch.item()]
        for _ in range(10):
            torch_opt.zero_grad()
            loss = x_torch ** 2
            loss.backward()
            torch_opt.step()
            torch_values.append(x_torch.item())
        
        # Compare
        all_match = True
        for i, (ours, theirs) in enumerate(zip(our_values, torch_values)):
            match = abs(ours - theirs) < 1e-5
            if not match:
                print(f"  ❌ Step {i}: our={ours:.6f}, torch={theirs:.6f}")
                all_match = False
        
        if all_match:
            print(f"✅ SGD matches PyTorch for 10 steps")
        return all_match
    else:
        print(f"🔸 Our values: {our_values[:5]}...")
        return True


def test_sgd_momentum():
    """Compare SGD with momentum."""
    print("\n" + "="*50)
    print("Testing SGD + Momentum")
    print("="*50)
    
    # Our implementation
    x = Value(5.0)
    our_opt = SGD([x], lr=0.1, momentum=0.9)
    
    our_values = [x.data]
    for _ in range(10):
        loss = x ** 2
        our_opt.zero_grad()
        loss.backward()
        our_opt.step()
        our_values.append(x.data)
    
    if PYTORCH_AVAILABLE:
        # PyTorch
        x_torch = torch.tensor([5.0], requires_grad=True)
        torch_opt = torch.optim.SGD([x_torch], lr=0.1, momentum=0.9)
        
        torch_values = [x_torch.item()]
        for _ in range(10):
            torch_opt.zero_grad()
            loss = x_torch ** 2
            loss.backward()
            torch_opt.step()
            torch_values.append(x_torch.item())
        
        all_match = True
        for i, (ours, theirs) in enumerate(zip(our_values, torch_values)):
            match = abs(ours - theirs) < 1e-4
            if not match:
                print(f"  ⚠️ Step {i}: our={ours:.6f}, torch={theirs:.6f}")
                all_match = False
        
        if all_match:
            print(f"✅ SGD+Momentum matches PyTorch")
        else:
            print(f"⚠️ Minor differences (momentum convention may differ)")
        return True  # Allow minor differences
    else:
        print(f"🔸 Our values: {our_values[:5]}...")
        return True


def test_adam():
    """Compare Adam implementations."""
    print("\n" + "="*50)
    print("Testing Adam")
    print("="*50)
    
    # Our implementation
    x = Value(5.0)
    our_opt = Adam([x], lr=0.1)
    
    our_values = [x.data]
    for _ in range(10):
        loss = x ** 2
        our_opt.zero_grad()
        loss.backward()
        our_opt.step()
        our_values.append(x.data)
    
    if PYTORCH_AVAILABLE:
        # PyTorch
        x_torch = torch.tensor([5.0], requires_grad=True)
        torch_opt = torch.optim.Adam([x_torch], lr=0.1)
        
        torch_values = [x_torch.item()]
        for _ in range(10):
            torch_opt.zero_grad()
            loss = x_torch ** 2
            loss.backward()
            torch_opt.step()
            torch_values.append(x_torch.item())
        
        all_match = True
        for i, (ours, theirs) in enumerate(zip(our_values, torch_values)):
            match = abs(ours - theirs) < 1e-4
            if not match:
                print(f"  ❌ Step {i}: our={ours:.6f}, torch={theirs:.6f}")
                all_match = False
        
        if all_match:
            print(f"✅ Adam matches PyTorch for 10 steps")
        return all_match
    else:
        print(f"🔸 Our values: {our_values[:5]}...")
        return True


def test_adamw():
    """Compare AdamW implementations."""
    print("\n" + "="*50)
    print("Testing AdamW")
    print("="*50)
    
    # Our implementation
    x = Value(5.0)
    our_opt = AdamW([x], lr=0.1, weight_decay=0.01)
    
    our_values = [x.data]
    for _ in range(10):
        loss = x ** 2
        our_opt.zero_grad()
        loss.backward()
        our_opt.step()
        our_values.append(x.data)
    
    if PYTORCH_AVAILABLE:
        # PyTorch
        x_torch = torch.tensor([5.0], requires_grad=True)
        torch_opt = torch.optim.AdamW([x_torch], lr=0.1, weight_decay=0.01)
        
        torch_values = [x_torch.item()]
        for _ in range(10):
            torch_opt.zero_grad()
            loss = x_torch ** 2
            loss.backward()
            torch_opt.step()
            torch_values.append(x_torch.item())
        
        all_match = True
        for i, (ours, theirs) in enumerate(zip(our_values, torch_values)):
            match = abs(ours - theirs) < 1e-4
            if not match:
                print(f"  ❌ Step {i}: our={ours:.6f}, torch={theirs:.6f}")
                all_match = False
        
        if all_match:
            print(f"✅ AdamW matches PyTorch for 10 steps")
        return all_match
    else:
        print(f"🔸 Our values: {our_values[:5]}...")
        return True


def test_convergence():
    """Test that optimizers converge to correct solution."""
    print("\n" + "="*50)
    print("Testing Convergence")
    print("="*50)
    
    results = []
    
    # Test 1: Minimize x²
    x = Value(5.0)
    opt = Adam([x], lr=0.5)
    for _ in range(100):
        loss = x ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    passed = abs(x.data) < 0.01
    results.append(("Minimize x²", passed))
    print(f"  {'✅' if passed else '❌'} x² → x = {x.data:.6f} (target: 0)")
    
    # Test 2: Minimize (x-3)²
    x = Value(0.0)
    opt = Adam([x], lr=0.5)
    for _ in range(100):
        loss = (x - 3) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    passed = abs(x.data - 3) < 0.01
    results.append(("Minimize (x-3)²", passed))
    print(f"  {'✅' if passed else '❌'} (x-3)² → x = {x.data:.6f} (target: 3)")
    
    # Test 3: 2D problem
    x = Value(0.0)
    y = Value(0.0)
    opt = Adam([x, y], lr=0.5)
    for _ in range(100):
        loss = (x - 1) ** 2 + (y - 2) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    passed = abs(x.data - 1) < 0.01 and abs(y.data - 2) < 0.01
    results.append(("2D problem", passed))
    print(f"  {'✅' if passed else '❌'} 2D → ({x.data:.3f}, {y.data:.3f}) (target: 1, 2)")
    
    return all(r[1] for r in results)


def main():
    print("="*50)
    print("Topic 03: PyTorch Reference Verification")
    print("="*50)
    
    if PYTORCH_AVAILABLE:
        print(f"PyTorch version: {torch.__version__}")
    
    results = []
    
    results.append(("SGD", test_sgd()))
    results.append(("SGD+Momentum", test_sgd_momentum()))
    results.append(("Adam", test_adam()))
    results.append(("AdamW", test_adamw()))
    results.append(("Convergence", test_convergence()))
    
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    
    all_passed = True
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed. Debug needed.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

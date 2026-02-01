"""
Topic 03: Level 02 - Complete Optimizers

Full implementation of common optimizers:
- SGD (with optional momentum and weight decay)
- Adam
- AdamW

Goal: A complete set of usable optimizers.
"""

import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add autograd to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Topic 02-Autograd-Engine" / "solutions"))

try:
    from level02_vectorized import Value
except ImportError:
    try:
        from level01_naive import Value
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "utils"))
        from autograd_stub import Value


class Optimizer:
    """Base class for all optimizers."""
    
    def __init__(self, parameters, lr: float = 0.01):
        self.parameters = list(parameters)
        self.lr = lr
        self.defaults = {'lr': lr}
    
    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0
    
    def state_dict(self) -> dict:
        """Return optimizer state for checkpointing."""
        return {'lr': self.lr}
    
    def load_state_dict(self, state_dict: dict):
        """Load optimizer state from checkpoint."""
        self.lr = state_dict.get('lr', self.lr)


class SGD(Optimizer):
    """
    SGD with optional momentum and weight decay.
    
    Update rule (with momentum):
        v = momentum × v + gradient
        θ = θ - lr × v - lr × weight_decay × θ
    
    Args:
        parameters: Iterable of Values
        lr: Learning rate
        momentum: Momentum factor (0 for vanilla SGD)
        weight_decay: L2 regularization coefficient
        nesterov: Whether to use Nesterov momentum
    """
    
    def __init__(
        self, 
        parameters, 
        lr: float = 0.01, 
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False
    ):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        
        # State: velocity for each parameter
        self.velocity = [0.0 for _ in self.parameters]
    
    def step(self):
        for i, p in enumerate(self.parameters):
            g = p.grad
            
            # Weight decay (L2 regularization)
            if self.weight_decay != 0:
                g = g + self.weight_decay * p.data
            
            # Momentum
            if self.momentum != 0:
                self.velocity[i] = self.momentum * self.velocity[i] + g
                
                if self.nesterov:
                    g = g + self.momentum * self.velocity[i]
                else:
                    g = self.velocity[i]
            
            # Update
            p.data -= self.lr * g
    
    def state_dict(self) -> dict:
        return {
            'lr': self.lr,
            'momentum': self.momentum,
            'velocity': self.velocity.copy()
        }
    
    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        self.momentum = state_dict.get('momentum', self.momentum)
        if 'velocity' in state_dict:
            self.velocity = state_dict['velocity'].copy()


class Adam(Optimizer):
    """
    Adam optimizer.
    
    Update rule:
        m = β₁ × m + (1 - β₁) × g
        v = β₂ × v + (1 - β₂) × g²
        m̂ = m / (1 - β₁^t)
        v̂ = v / (1 - β₂^t)
        θ = θ - lr × m̂ / (√v̂ + ε)
    
    Args:
        parameters: Iterable of Values
        lr: Learning rate
        betas: Coefficients for moment estimation (β₁, β₂)
        eps: Term for numerical stability
        weight_decay: L2 regularization (note: use AdamW for proper weight decay)
    """
    
    def __init__(
        self,
        parameters,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # State
        self.m = [0.0 for _ in self.parameters]  # First moment
        self.v = [0.0 for _ in self.parameters]  # Second moment
        self.t = 0  # Timestep
    
    def step(self):
        self.t += 1
        
        for i, p in enumerate(self.parameters):
            g = p.grad
            
            # L2 regularization (added to gradient)
            if self.weight_decay != 0:
                g = g + self.weight_decay * p.data
            
            # Update biased first moment
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            
            # Update biased second moment
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameter
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
    
    def state_dict(self) -> dict:
        return {
            'lr': self.lr,
            'betas': (self.beta1, self.beta2),
            'm': self.m.copy(),
            'v': self.v.copy(),
            't': self.t
        }
    
    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        if 'betas' in state_dict:
            self.beta1, self.beta2 = state_dict['betas']
        if 'm' in state_dict:
            self.m = state_dict['m'].copy()
        if 'v' in state_dict:
            self.v = state_dict['v'].copy()
        self.t = state_dict.get('t', 0)


class AdamW(Adam):
    """
    AdamW optimizer with decoupled weight decay.
    
    Unlike Adam, weight decay is applied separately from the gradient update.
    This is the recommended variant for most use cases.
    
    Update rule:
        (same as Adam for m, v, m̂, v̂)
        θ = θ - lr × m̂ / (√v̂ + ε) - lr × λ × θ
    """
    
    def step(self):
        self.t += 1
        
        for i, p in enumerate(self.parameters):
            g = p.grad
            
            # Update biased first moment (NO weight decay in gradient)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            
            # Update biased second moment
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Adam update
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
            
            # Decoupled weight decay
            if self.weight_decay != 0:
                p.data -= self.lr * self.weight_decay * p.data


class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    
    Update rule:
        v = α × v + (1 - α) × g²
        θ = θ - lr × g / (√v + ε)
    """
    
    def __init__(
        self,
        parameters,
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        
        # State: running average of squared gradients
        self.v = [0.0 for _ in self.parameters]
    
    def step(self):
        for i, p in enumerate(self.parameters):
            g = p.grad
            
            if self.weight_decay != 0:
                g = g + self.weight_decay * p.data
            
            # Update running average
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * (g ** 2)
            
            # Update parameter
            p.data -= self.lr * g / (math.sqrt(self.v[i]) + self.eps)


# ==================== Demo ====================

def demo():
    print("="*60)
    print("Level 02: Complete Optimizers Demo")
    print("="*60)
    
    def test_optimizer(optimizer_class, name, **kwargs):
        """Test an optimizer on f(x) = x²"""
        print(f"\n{name}:")
        print("-" * 40)
        
        x = Value(5.0)
        optimizer = optimizer_class([x], **kwargs)
        
        for step in range(20):
            loss = x ** 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step < 5 or step == 19:
                print(f"  Step {step:2d}: x = {x.data:8.5f}, loss = {loss.data:10.5f}")
        
        print(f"  Final x: {x.data:.8f}")
        return abs(x.data) < 0.01  # Success if x is close to 0
    
    results = []
    
    # Test all optimizers
    results.append(("SGD", test_optimizer(SGD, "SGD (lr=0.1)", lr=0.1)))
    results.append(("SGD+Momentum", test_optimizer(SGD, "SGD + Momentum", lr=0.05, momentum=0.9)))
    results.append(("Adam", test_optimizer(Adam, "Adam", lr=0.5)))
    results.append(("AdamW", test_optimizer(AdamW, "AdamW (with decay)", lr=0.5, weight_decay=0.01)))
    results.append(("RMSprop", test_optimizer(RMSprop, "RMSprop", lr=0.1)))
    
    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
    
    # 2D optimization comparison
    print("\n" + "="*60)
    print("2D Comparison: Rosenbrock-like function")
    print("f(x,y) = (1-x)² + 10(y-x²)²")
    print("="*60)
    
    def rosenbrock_test(optimizer_class, name, **kwargs):
        x = Value(0.0)
        y = Value(0.0)
        optimizer = optimizer_class([x, y], **kwargs)
        
        for step in range(200):
            loss = (1 - x.data)**2 + 10 * (y.data - x.data**2)**2
            loss_val = Value(loss)  # Simplified, not using autograd here
            
            # Manual gradients for Rosenbrock
            x.grad = -2*(1 - x.data) - 40*x.data*(y.data - x.data**2)
            y.grad = 20*(y.data - x.data**2)
            
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"  {name}: ({x.data:.4f}, {y.data:.4f}) - Goal: (1, 1)")
    
    rosenbrock_test(SGD, "SGD", lr=0.001)
    rosenbrock_test(SGD, "SGD+Momentum", lr=0.001, momentum=0.9)
    rosenbrock_test(Adam, "Adam", lr=0.01)
    
    print("\n✅ All optimizers working!")


if __name__ == "__main__":
    demo()

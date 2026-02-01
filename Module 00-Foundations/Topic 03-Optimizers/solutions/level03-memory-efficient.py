"""
Topic 03: Level 03 - Memory-Efficient Optimizers

Optimized implementations with:
- Lazy state initialization
- Fused operations
- Gradient clipping
- Learning rate scheduling
- State management

Goal: Production-quality optimizer implementations.
"""

import math
from typing import Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class ParamState:
    """State container for a single parameter."""
    __slots__ = ['momentum_buffer', 'exp_avg', 'exp_avg_sq', 'step']
    
    def __init__(self):
        self.momentum_buffer: Optional[float] = None
        self.exp_avg: float = 0.0
        self.exp_avg_sq: float = 0.0
        self.step: int = 0


class Optimizer:
    """Base optimizer with state management."""
    
    def __init__(self, parameters, defaults: Dict):
        self.parameters = list(parameters)
        self.defaults = defaults
        self.state: Dict[int, ParamState] = {}  # param id -> state
    
    def _get_state(self, param_id: int) -> ParamState:
        """Lazy state initialization."""
        if param_id not in self.state:
            self.state[param_id] = ParamState()
        return self.state[param_id]
    
    def step(self, closure: Optional[Callable] = None):
        raise NotImplementedError
    
    def zero_grad(self, set_to_none: bool = False):
        """
        Reset gradients.
        
        Args:
            set_to_none: If True, set grad to None instead of zero.
                         More memory efficient but requires checking for None.
        """
        for p in self.parameters:
            if set_to_none:
                p.grad = None
            else:
                p.grad = 0.0
    
    def state_dict(self) -> Dict:
        """Return serializable state."""
        return {
            'defaults': self.defaults,
            'state': {k: vars(v) for k, v in self.state.items()}
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load state from dict."""
        self.defaults = state_dict['defaults']
        for k, v in state_dict['state'].items():
            state = ParamState()
            for attr, val in v.items():
                setattr(state, attr, val)
            self.state[k] = state


class SGD(Optimizer):
    """
    Memory-efficient SGD with fused weight decay.
    """
    
    def __init__(
        self,
        parameters,
        lr: float = 0.01,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False
    ):
        defaults = {
            'lr': lr,
            'momentum': momentum,
            'dampening': dampening,
            'weight_decay': weight_decay,
            'nesterov': nesterov
        }
        super().__init__(parameters, defaults)
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov requires momentum > 0 and dampening = 0")
    
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for p in self.parameters:
            if p.grad is None:
                continue
            
            g = p.grad
            
            # Weight decay (fused)
            if self.defaults['weight_decay'] != 0:
                g = g + self.defaults['weight_decay'] * p.data
            
            # Momentum
            if self.defaults['momentum'] != 0:
                state = self._get_state(id(p))
                
                if state.momentum_buffer is None:
                    state.momentum_buffer = g
                else:
                    state.momentum_buffer = (
                        self.defaults['momentum'] * state.momentum_buffer 
                        + (1 - self.defaults['dampening']) * g
                    )
                
                if self.defaults['nesterov']:
                    g = g + self.defaults['momentum'] * state.momentum_buffer
                else:
                    g = state.momentum_buffer
            
            # Update
            p.data -= self.defaults['lr'] * g
        
        return loss


class Adam(Optimizer):
    """
    Memory-efficient Adam with optional AMSGrad.
    """
    
    def __init__(
        self,
        parameters,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False
    ):
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'amsgrad': amsgrad
        }
        super().__init__(parameters, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()
        
        beta1, beta2 = self.defaults['betas']
        
        for p in self.parameters:
            if p.grad is None:
                continue
            
            state = self._get_state(id(p))
            state.step += 1
            
            g = p.grad
            
            # L2 regularization (added to gradient for standard Adam)
            if self.defaults['weight_decay'] != 0:
                g = g + self.defaults['weight_decay'] * p.data
            
            # Update moments
            state.exp_avg = beta1 * state.exp_avg + (1 - beta1) * g
            state.exp_avg_sq = beta2 * state.exp_avg_sq + (1 - beta2) * (g ** 2)
            
            # Bias correction
            bias_correction1 = 1 - beta1 ** state.step
            bias_correction2 = 1 - beta2 ** state.step
            
            # Compute step
            step_size = self.defaults['lr'] / bias_correction1
            
            if self.defaults['amsgrad']:
                # Use max of all v's for stability
                if not hasattr(state, 'max_exp_avg_sq'):
                    state.max_exp_avg_sq = 0.0
                state.max_exp_avg_sq = max(state.max_exp_avg_sq, state.exp_avg_sq)
                denom = math.sqrt(state.max_exp_avg_sq / bias_correction2) + self.defaults['eps']
            else:
                denom = math.sqrt(state.exp_avg_sq / bias_correction2) + self.defaults['eps']
            
            p.data -= step_size * state.exp_avg / denom
        
        return loss


class AdamW(Adam):
    """
    AdamW with decoupled weight decay.
    """
    
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()
        
        beta1, beta2 = self.defaults['betas']
        
        for p in self.parameters:
            if p.grad is None:
                continue
            
            state = self._get_state(id(p))
            state.step += 1
            
            g = p.grad  # No weight decay in gradient!
            
            # Update moments
            state.exp_avg = beta1 * state.exp_avg + (1 - beta1) * g
            state.exp_avg_sq = beta2 * state.exp_avg_sq + (1 - beta2) * (g ** 2)
            
            # Bias correction
            bias_correction1 = 1 - beta1 ** state.step
            bias_correction2 = 1 - beta2 ** state.step
            
            # Adam update
            step_size = self.defaults['lr'] / bias_correction1
            denom = math.sqrt(state.exp_avg_sq / bias_correction2) + self.defaults['eps']
            p.data -= step_size * state.exp_avg / denom
            
            # Decoupled weight decay (applied to parameter, not gradient)
            if self.defaults['weight_decay'] != 0:
                p.data -= self.defaults['lr'] * self.defaults['weight_decay'] * p.data
        
        return loss


# ==================== Learning Rate Schedulers ====================

class LRScheduler:
    """Base class for learning rate schedulers."""
    
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.base_lr = optimizer.defaults['lr']
        self.last_epoch = last_epoch
    
    def get_lr(self) -> float:
        raise NotImplementedError
    
    def step(self):
        self.last_epoch += 1
        new_lr = self.get_lr()
        self.optimizer.defaults['lr'] = new_lr


class StepLR(LRScheduler):
    """Decay LR by gamma every step_size epochs."""
    
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> float:
        return self.base_lr * (self.gamma ** (self.last_epoch // self.step_size))


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing schedule."""
    
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> float:
        return self.eta_min + (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * self.last_epoch / self.T_max)
        ) / 2


class WarmupLR(LRScheduler):
    """Linear warmup followed by constant LR."""
    
    def __init__(self, optimizer: Optimizer, warmup_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> float:
        if self.last_epoch < self.warmup_steps:
            return self.base_lr * (self.last_epoch + 1) / self.warmup_steps
        return self.base_lr


# ==================== Gradient Clipping ====================

def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0) -> float:
    """
    Clip gradient norm in-place.
    
    Args:
        parameters: Iterable of parameters
        max_norm: Maximum norm
        norm_type: Type of norm (default: L2)
    
    Returns:
        Total gradient norm before clipping
    """
    parameters = list(parameters)
    
    if norm_type == float('inf'):
        total_norm = max(abs(p.grad) for p in parameters if p.grad is not None)
    else:
        total_norm = sum(
            abs(p.grad) ** norm_type 
            for p in parameters 
            if p.grad is not None
        ) ** (1.0 / norm_type)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad *= clip_coef
    
    return total_norm


def clip_grad_value_(parameters, clip_value: float):
    """Clip gradient values in-place."""
    for p in parameters:
        if p.grad is not None:
            p.grad = max(-clip_value, min(clip_value, p.grad))


# ==================== Demo ====================

def demo():
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Topic 02-Autograd-Engine" / "solutions"))
    
    try:
        from level02_vectorized import Value
    except ImportError:
        from level01_naive import Value
    
    print("="*60)
    print("Level 03: Memory-Efficient Optimizers Demo")
    print("="*60)
    
    # Test with learning rate scheduler
    print("\n1. AdamW with Cosine Annealing:")
    print("-" * 40)
    
    x = Value(5.0)
    optimizer = AdamW([x], lr=0.5, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)
    
    for step in range(20):
        loss = x ** 2
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        clip_grad_norm_([x], max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        if step < 5 or step >= 15:
            print(f"  Step {step:2d}: x={x.data:7.4f}, loss={loss.data:9.4f}, lr={optimizer.defaults['lr']:.4f}")
    
    print(f"  Final x: {x.data:.6f}")
    
    # Test warmup
    print("\n2. Adam with Linear Warmup:")
    print("-" * 40)
    
    x = Value(5.0)
    optimizer = Adam([x], lr=0.5)
    scheduler = WarmupLR(optimizer, warmup_steps=5)
    
    for step in range(10):
        loss = x ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        print(f"  Step {step}: lr={optimizer.defaults['lr']:.4f}")
    
    print("\n✅ Memory-efficient optimizers working!")


if __name__ == "__main__":
    demo()

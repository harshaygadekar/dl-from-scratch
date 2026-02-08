"""
Topic 02: Level 02 - Complete Autograd (All Common Operations)

Extends level01 with all operations needed for neural networks:
- Power, division, exp, log
- Activation functions: relu, tanh, sigmoid
- Proper numeric handling

Goal: A fully usable autograd engine.
"""

import math


class Value:
    """
    Full-featured autograd Value class.

    Supports all operations needed for building neural networks.

    Example:
        x = Value(-2.0)
        z = x.relu() + x.tanh()
        z.backward()
    """

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ==================== Arithmetic Operations ====================

    def __add__(self, other):
        """z = self + other"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        """z = self * other"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __truediv__(self, other):
        """z = self / other"""
        return self * other**-1

    def __rtruediv__(self, other):
        """z = other / self"""
        return other * self**-1

    def __pow__(self, other):
        """z = self ** other (other must be a number, not Value)"""
        assert isinstance(other, (int, float)), "Only supports numeric powers"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            # d/dx (x^n) = n * x^(n-1)
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    # ==================== Transcendental Functions ====================

    def exp(self):
        """z = e^self"""
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            # d/dx (e^x) = e^x
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def log(self):
        """z = ln(self)"""
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            # d/dx (ln(x)) = 1/x
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward

        return out

    # ==================== Activation Functions ====================

    def relu(self):
        """z = max(0, self)"""
        out = Value(max(0, self.data), (self,), "ReLU")

        def _backward():
            # d/dx ReLU(x) = 1 if x > 0 else 0
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        """z = tanh(self)"""
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # d/dx tanh(x) = 1 - tanh(x)^2
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def sigmoid(self):
        """z = 1 / (1 + e^(-self))"""
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward

        return out

    # ==================== Backward Pass ====================

    def backward(self):
        """Compute gradients via reverse-mode autodiff."""
        topo = []
        visited = set()
        stack = [(self, False)]

        # Iterative DFS avoids recursion-depth failures on long computation chains.
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for child in node._prev:
                if child not in visited:
                    stack.append((child, False))

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        """Reset gradient to zero."""
        self.grad = 0.0


# ==================== Neural Network Building Blocks ====================


class Neuron:
    """A single neuron with weights and bias."""

    def __init__(self, nin):
        """
        Args:
            nin: Number of inputs
        """
        import random

        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x):
        """Forward pass: activation(sum(wi*xi) + b)"""
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    """A layer of neurons."""

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Multi-Layer Perceptron."""

    def __init__(self, nin, nouts):
        """
        Args:
            nin: Number of inputs
            nouts: List of layer sizes
        """
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ==================== Demo ====================


def demo():
    print("=" * 50)
    print("Level 02: Complete Autograd Demo")
    print("=" * 50)

    # Test all operations
    print("\n1. Testing division and power:")
    x = Value(4.0)
    z = x**0.5  # sqrt(4) = 2
    z.backward()
    print(f"sqrt(4) = {z.data:.4f}, d/dx = {x.grad:.4f}")
    print(f"Expected: 2.0, d/dx = 0.25 (= 1/(2*sqrt(4)))")

    print("\n2. Testing exp and log:")
    x = Value(2.0)
    z = x.exp().log()  # log(exp(x)) = x
    z.backward()
    print(f"log(exp(2)) = {z.data:.4f}, d/dx = {x.grad:.4f}")
    print(f"Expected: 2.0, d/dx = 1.0")

    print("\n3. Testing ReLU:")
    x = Value(-3.0)
    z = x.relu()
    z.backward()
    print(f"ReLU(-3) = {z.data}, d/dx = {x.grad}")
    print(f"Expected: 0.0, d/dx = 0.0")

    x = Value(3.0)
    z = x.relu()
    z.backward()
    print(f"ReLU(3) = {z.data}, d/dx = {x.grad}")
    print(f"Expected: 3.0, d/dx = 1.0")

    print("\n4. Testing tanh:")
    x = Value(0.0)
    z = x.tanh()
    z.backward()
    print(f"tanh(0) = {z.data:.4f}, d/dx = {x.grad:.4f}")
    print(f"Expected: 0.0, d/dx = 1.0")

    print("\n5. Testing sigmoid:")
    x = Value(0.0)
    z = x.sigmoid()
    z.backward()
    print(f"sigmoid(0) = {z.data:.4f}, d/dx = {x.grad:.4f}")
    print(f"Expected: 0.5, d/dx = 0.25")

    print("\n6. Testing simple neural network:")
    # Create a simple MLP
    mlp = MLP(3, [4, 4, 1])  # 3 inputs, 2 hidden layers of 4, 1 output

    # Forward pass
    x = [Value(1.0), Value(2.0), Value(3.0)]
    y = mlp(x)
    print(f"MLP output: {y.data:.4f}")

    # Backward pass
    y.backward()
    print(f"Number of parameters: {len(mlp.parameters())}")
    print(f"Sample gradient: {mlp.parameters()[0].grad:.4f}")

    print("\n✅ All operations working!")


if __name__ == "__main__":
    demo()

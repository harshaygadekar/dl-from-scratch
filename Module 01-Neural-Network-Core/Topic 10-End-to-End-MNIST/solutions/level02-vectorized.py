"""
Topic 10: End-to-End MNIST - Level 02 Vectorized
With Adam optimizer and better architecture.
"""

import numpy as np


class Linear:
    def __init__(self, in_f, out_f):
        self.W = np.random.randn(in_f, out_f) * np.sqrt(2.0 / in_f)
        self.b = np.zeros(out_f)
    
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    
    def backward(self, grad):
        self.grad_W = self.x.T @ grad
        self.grad_b = grad.sum(axis=0)
        return grad @ self.W.T


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return np.maximum(0, x)
    
    def backward(self, grad):
        return grad * self.mask


class Dropout:
    def __init__(self, p=0.2):
        self.p = p
        self.training = True
    
    def forward(self, x):
        if not self.training:
            return x
        self.mask = np.random.rand(*x.shape) > self.p
        return x * self.mask / (1 - self.p)
    
    def backward(self, grad):
        return grad * self.mask / (1 - self.p) if self.training else grad


class BatchNorm1d:
    def __init__(self, n, momentum=0.1, eps=1e-5):
        self.gamma = np.ones(n)
        self.beta = np.zeros(n)
        self.running_mean = np.zeros(n)
        self.running_var = np.ones(n)
        self.momentum = momentum
        self.eps = eps
        self.training = True
    
    def forward(self, x):
        if self.training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean, var = self.running_mean, self.running_var
        
        self.x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * self.x_norm + self.beta
    
    def backward(self, grad):
        # Simplified
        return grad * self.gamma


class Adam:
    def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.layers = [l for l in layers if hasattr(l, 'W')]
        self.t = 0
        self.m = {id(l): {'W': 0, 'b': 0} for l in self.layers}
        self.v = {id(l): {'W': 0, 'b': 0} for l in self.layers}
    
    def step(self):
        self.t += 1
        for l in self.layers:
            lid = id(l)
            for name in ['W', 'b']:
                g = getattr(l, f'grad_{name}')
                self.m[lid][name] = self.beta1 * self.m[lid][name] + (1 - self.beta1) * g
                self.v[lid][name] = self.beta2 * self.v[lid][name] + (1 - self.beta2) * (g ** 2)
                m_hat = self.m[lid][name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[lid][name] / (1 - self.beta2 ** self.t)
                update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                setattr(l, name, getattr(l, name) - update)


class SoftmaxCE:
    def forward(self, logits, y):
        x_max = logits.max(axis=-1, keepdims=True)
        exp_x = np.exp(logits - x_max)
        self.probs = exp_x / exp_x.sum(axis=-1, keepdims=True)
        self.y = y
        return -np.mean(np.sum(y * np.log(self.probs + 1e-15), axis=-1))
    
    def backward(self):
        return (self.probs - self.y) / len(self.y)


class MLP:
    def __init__(self, sizes, dropout=0.2, use_bn=True):
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                if use_bn:
                    self.layers.append(BatchNorm1d(sizes[i+1]))
                self.layers.append(ReLU())
                if dropout > 0:
                    self.layers.append(Dropout(dropout))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def train(self):
        for l in self.layers:
            if hasattr(l, 'training'):
                l.training = True
    
    def eval(self):
        for l in self.layers:
            if hasattr(l, 'training'):
                l.training = False


def demo():
    print("=" * 50)
    print("MNIST Training - Level 02 (Vectorized)")
    print("=" * 50)
    
    np.random.seed(42)
    x_train = np.random.randn(5000, 784).astype(np.float32) * 0.3
    y_train = np.random.randint(0, 10, 5000)
    y_train_oh = np.eye(10)[y_train]
    
    model = MLP([784, 256, 128, 10], dropout=0.2, use_bn=True)
    loss_fn = SoftmaxCE()
    optimizer = Adam(model.layers, lr=0.001)
    
    for epoch in range(3):
        model.train()
        idx = np.random.permutation(len(x_train))
        total_loss = 0
        
        for start in range(0, len(x_train), 64):
            batch_idx = idx[start:start+64]
            x, y = x_train[batch_idx], y_train_oh[batch_idx]
            
            logits = model.forward(x)
            loss = loss_fn.forward(logits, y)
            total_loss += loss
            
            grad = loss_fn.backward()
            model.backward(grad)
            optimizer.step()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")


if __name__ == "__main__":
    demo()

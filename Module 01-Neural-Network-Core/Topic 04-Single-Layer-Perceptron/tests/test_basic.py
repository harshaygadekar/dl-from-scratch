"""
Topic 04: Single Layer Perceptron - Basic Tests

Tests for core perceptron functionality.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add solutions to path
sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import Perceptron as NaivePerceptron
from level02_vectorized import Perceptron as VectorizedPerceptron


class TestSigmoid:
    """Test sigmoid activation."""
    
    def test_sigmoid_zero(self):
        """σ(0) = 0.5"""
        model = NaivePerceptron(1)
        assert np.isclose(model.sigmoid(0), 0.5)
    
    def test_sigmoid_large_positive(self):
        """σ(large) ≈ 1"""
        model = NaivePerceptron(1)
        assert model.sigmoid(100) > 0.99
    
    def test_sigmoid_large_negative(self):
        """σ(-large) ≈ 0"""
        model = NaivePerceptron(1)
        assert model.sigmoid(-100) < 0.01
    
    def test_sigmoid_symmetry(self):
        """σ(-z) = 1 - σ(z)"""
        model = NaivePerceptron(1)
        z = 2.5
        assert np.isclose(model.sigmoid(-z), 1 - model.sigmoid(z))


class TestForwardPass:
    """Test forward pass computation."""
    
    def test_forward_output_range(self):
        """Output should be in (0, 1)."""
        model = NaivePerceptron(2)
        x = np.array([1.0, 2.0])
        output = model.forward(x)
        assert 0 < output < 1
    
    def test_forward_with_zero_weights(self):
        """Output is σ(b) when weights are zero."""
        model = NaivePerceptron(2)
        model.w = np.zeros(2)
        model.b = 0.0
        x = np.array([1.0, 2.0])
        assert np.isclose(model.forward(x), 0.5)
    
    def test_forward_vectorized_matches_naive(self):
        """Vectorized forward should match naive."""
        np.random.seed(42)
        naive = NaivePerceptron(3)
        vectorized = VectorizedPerceptron(3)
        vectorized.w = naive.w.copy()
        vectorized.b = naive.b
        
        X = np.random.randn(10, 3)
        
        naive_outputs = [naive.forward(X[i]) for i in range(10)]
        vectorized_outputs = vectorized.forward(X)
        
        np.testing.assert_allclose(naive_outputs, vectorized_outputs, rtol=1e-5)


class TestLoss:
    """Test loss computation."""
    
    def test_loss_perfect_prediction(self):
        """Loss should be low for correct predictions."""
        model = NaivePerceptron(1)
        loss_correct = model.compute_loss(0.99, 1.0)
        loss_wrong = model.compute_loss(0.01, 1.0)
        assert loss_correct < loss_wrong
    
    def test_loss_symmetric(self):
        """Loss symmetric for y=0 and y=1 cases."""
        model = NaivePerceptron(1)
        loss_1 = model.compute_loss(0.9, 1.0)
        loss_0 = model.compute_loss(0.1, 0.0)
        assert np.isclose(loss_1, loss_0, rtol=1e-3)
    
    def test_loss_numerical_stability(self):
        """Loss should not be inf for extreme predictions."""
        model = NaivePerceptron(1)
        loss = model.compute_loss(1e-20, 1.0)
        assert np.isfinite(loss)
        loss = model.compute_loss(1 - 1e-20, 0.0)
        assert np.isfinite(loss)


class TestBackward:
    """Test gradient computation."""
    
    def test_gradient_direction(self):
        """Gradient should point toward reducing loss."""
        model = NaivePerceptron(2)
        x = np.array([1.0, 1.0])
        y = 1.0
        y_pred = 0.3  # Wrong prediction (should be closer to 1)
        
        model.backward(x, y, y_pred)
        
        # Error is negative (y_pred - y = -0.7)
        # So dw should be negative, meaning we should increase weights
        assert model.dw[0] < 0
        assert model.db < 0
    
    def test_gradient_magnitude(self):
        """Larger errors should give larger gradients."""
        model = NaivePerceptron(2)
        x = np.array([1.0, 1.0])
        y = 1.0
        
        model.backward(x, y, 0.3)  # Large error
        dw_large = model.dw.copy()
        
        model.backward(x, y, 0.9)  # Small error
        dw_small = model.dw.copy()
        
        assert np.linalg.norm(dw_large) > np.linalg.norm(dw_small)


class TestTraining:
    """Test training convergence."""
    
    def test_convergence_on_linearly_separable(self):
        """Should converge on linearly separable data."""
        np.random.seed(42)
        
        # Simple linearly separable data
        X_pos = np.random.randn(50, 2) + np.array([3, 3])
        X_neg = np.random.randn(50, 2) + np.array([-3, -3])
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * 50 + [0] * 50)
        
        model = VectorizedPerceptron(2)
        
        # Train
        for _ in range(100):
            y_pred = model.forward(X)
            model.backward(X, y, y_pred)
            model.update(lr=0.5)
        
        # Should achieve high accuracy
        accuracy = model.accuracy(X, y)
        assert accuracy > 0.95


class TestPrediction:
    """Test prediction functionality."""
    
    def test_predict_binary_output(self):
        """Predictions should be 0 or 1."""
        model = VectorizedPerceptron(2)
        X = np.random.randn(10, 2)
        predictions = model.predict(X)
        
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_threshold(self):
        """Threshold at 0.5."""
        model = NaivePerceptron(1)
        model.w = np.array([0.0])
        model.b = 0.0  # Output will be exactly 0.5
        
        # At exactly 0.5, should predict 1 (>= 0.5)
        prediction = model.predict(np.array([0.0]))
        assert prediction == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

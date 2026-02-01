"""
Topic 04: Single Layer Perceptron - Edge Case Tests

Tests for edge cases and numerical stability.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import Perceptron


class TestNumericalStability:
    """Test numerical stability edge cases."""
    
    def test_extreme_inputs(self):
        """Handle very large inputs."""
        model = Perceptron(2)
        model.w = np.array([100.0, 100.0])
        model.b = 0.0
        
        X = np.array([[100.0, 100.0]])  # Should give very large z
        output = model.forward(X)
        
        # Should not be nan or inf
        assert np.isfinite(output[0])
        assert 0 <= output[0] <= 1
    
    def test_extreme_negative_inputs(self):
        """Handle very negative z values."""
        model = Perceptron(2)
        model.w = np.array([100.0, 100.0])
        
        X = np.array([[-100.0, -100.0]])
        output = model.forward(X)
        
        assert np.isfinite(output[0])
        assert output[0] < 0.01
    
    def test_loss_with_predictions_near_zero(self):
        """Loss computation with y_pred ≈ 0."""
        model = Perceptron(1)
        
        # Should not produce inf
        loss = model.compute_loss(np.array([1e-10]), np.array([1.0]))
        assert np.isfinite(loss)
    
    def test_loss_with_predictions_near_one(self):
        """Loss computation with y_pred ≈ 1."""
        model = Perceptron(1)
        
        loss = model.compute_loss(np.array([1 - 1e-10]), np.array([0.0]))
        assert np.isfinite(loss)


class TestEdgeCases:
    """Test boundary conditions."""
    
    def test_single_sample(self):
        """Handle single sample input."""
        model = Perceptron(2)
        X = np.array([[1.0, 2.0]])
        y = np.array([1.0])
        
        output = model.forward(X)
        assert output.shape == (1,)
        
        model.backward(X, y, output)
        model.update(0.1)
        # Should not raise
    
    def test_high_dimensional_input(self):
        """Handle high-dimensional inputs."""
        model = Perceptron(1000)
        X = np.random.randn(10, 1000)
        y = np.random.randint(0, 2, 10).astype(float)
        
        output = model.forward(X)
        assert output.shape == (10,)
        assert np.all(np.isfinite(output))
    
    def test_zero_input(self):
        """Handle zero inputs."""
        model = Perceptron(2)
        X = np.zeros((5, 2))
        
        output = model.forward(X)
        
        # All outputs should be sigmoid(b)
        expected = 1 / (1 + np.exp(-model.b))
        np.testing.assert_allclose(output, expected, rtol=1e-5)
    
    def test_identical_samples(self):
        """Handle identical samples in batch."""
        model = Perceptron(2)
        X = np.ones((10, 2))  # All same
        y = np.ones(10)
        
        output = model.forward(X)
        
        # All outputs should be identical
        assert np.allclose(output, output[0])


class TestGradientEdgeCases:
    """Test gradient computation edge cases."""
    
    def test_gradient_with_perfect_prediction(self):
        """Gradient should be small for perfect predictions."""
        model = Perceptron(2)
        X = np.array([[1.0, 2.0]])
        y = np.array([1.0])
        y_pred = np.array([0.999])  # Almost perfect
        
        model.backward(X, y, y_pred)
        
        # Gradient should be very small
        assert np.abs(model.dw).max() < 0.01
        assert np.abs(model.db) < 0.01
    
    def test_gradient_with_wrong_prediction(self):
        """Gradient should be large for wrong predictions."""
        model = Perceptron(2)
        X = np.array([[1.0, 2.0]])
        y = np.array([1.0])
        y_pred = np.array([0.1])  # Very wrong
        
        model.backward(X, y, y_pred)
        
        # Gradient should be significant
        assert np.abs(model.dw).max() > 0.1
    
    def test_gradient_accumulation(self):
        """Test gradient averaging over batch."""
        model = Perceptron(2)
        
        # Same sample repeated
        X = np.array([[1.0, 1.0], [1.0, 1.0]])
        y = np.array([1.0, 1.0])
        y_pred = np.array([0.5, 0.5])
        
        model.backward(X, y, y_pred)
        
        # Single sample gradient
        model2 = Perceptron(2)
        X_single = np.array([[1.0, 1.0]])
        y_single = np.array([1.0])
        y_pred_single = np.array([0.5])
        model2.backward(X_single, y_single, y_pred_single)
        
        # Should be the same (averaged)
        np.testing.assert_allclose(model.dw, model2.dw, rtol=1e-5)


class TestLearningRateEdgeCases:
    """Test learning rate edge cases."""
    
    def test_zero_learning_rate(self):
        """No update with zero learning rate."""
        model = Perceptron(2)
        w_before = model.w.copy()
        b_before = model.b
        
        X = np.random.randn(10, 2)
        y = np.random.randint(0, 2, 10).astype(float)
        
        y_pred = model.forward(X)
        model.backward(X, y, y_pred)
        model.update(lr=0.0)
        
        np.testing.assert_array_equal(model.w, w_before)
        assert model.b == b_before
    
    def test_large_learning_rate(self):
        """Should not produce nan with large learning rate."""
        model = Perceptron(2)
        X = np.random.randn(10, 2)
        y = np.random.randint(0, 2, 10).astype(float)
        
        for _ in range(10):
            y_pred = model.forward(X)
            model.backward(X, y, y_pred)
            model.update(lr=10.0)  # Very large
        
        # Weights may be large but should not be nan
        assert np.all(np.isfinite(model.w))
        assert np.isfinite(model.b)


class TestBatchSizeEdgeCases:
    """Test various batch sizes."""
    
    def test_batch_size_one(self):
        """Single sample batch."""
        model = Perceptron(2)
        X = np.array([[1.0, 2.0]])
        y = np.array([1.0])
        
        y_pred = model.forward(X)
        model.backward(X, y, y_pred)
        model.update(0.1)
        
        assert np.all(np.isfinite(model.w))
    
    def test_large_batch(self):
        """Large batch training."""
        model = Perceptron(2)
        X = np.random.randn(10000, 2)
        y = np.random.randint(0, 2, 10000).astype(float)
        
        y_pred = model.forward(X)
        model.backward(X, y, y_pred)
        model.update(0.1)
        
        assert np.all(np.isfinite(model.w))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

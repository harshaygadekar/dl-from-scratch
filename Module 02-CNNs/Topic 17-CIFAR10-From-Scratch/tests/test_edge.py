import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import SoftmaxClassifier


def test_single_sample_batch():
    model = SoftmaxClassifier(in_features=12, num_classes=3)
    x = np.random.randn(1, 12).astype(np.float32)
    y = np.array([1])
    logits = model.forward(x)
    loss, grad_w, grad_b = model.loss_and_grad(logits, y)
    assert np.isfinite(loss)
    assert grad_w.shape == model.W.shape
    assert grad_b.shape == model.b.shape

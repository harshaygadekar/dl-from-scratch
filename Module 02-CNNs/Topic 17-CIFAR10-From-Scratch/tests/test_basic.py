import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import SoftmaxClassifier, train_one_epoch, evaluate


def test_forward_shape():
    model = SoftmaxClassifier(in_features=32 * 32 * 3, num_classes=10)
    x = np.random.randn(16, 32 * 32 * 3).astype(np.float32)
    logits = model.forward(x)
    assert logits.shape == (16, 10)


def test_one_training_epoch_runs():
    model = SoftmaxClassifier(in_features=32 * 32 * 3, num_classes=10)
    x = np.random.randn(64, 32 * 32 * 3).astype(np.float32)
    y = np.random.randint(0, 10, size=(64,))
    loss = train_one_epoch(model, x, y, batch_size=16, lr=0.1)
    assert np.isfinite(loss)


def test_evaluate_range():
    model = SoftmaxClassifier(in_features=32 * 32 * 3, num_classes=10)
    x = np.random.randn(32, 32 * 32 * 3).astype(np.float32)
    y = np.random.randint(0, 10, size=(32,))
    acc = evaluate(model, x, y)
    assert 0.0 <= acc <= 1.0

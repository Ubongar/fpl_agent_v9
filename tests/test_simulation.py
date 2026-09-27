from fplagent.models.simulation import _allocate_events
import numpy as np


def test_event_allocator_conserves_allocated_events():
    counts = np.array([2, 0, 1])
    probs = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]])
    out = _allocate_events(counts, probs, np.random.default_rng(1))
    assert out.shape == probs.shape
    assert out[:, 0].sum() == 2
    assert out[:, 1].sum() == 0
    assert out[:, 2].sum() <= 1

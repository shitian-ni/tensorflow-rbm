import pytest
import numpy as np
from rbmae import Batcher


def test_basic():
    n = np.random.randint(5, 100)
    m = np.random.randint(5, 100)
    a = np.random.rand(n, m)
    b = np.random.random_integers(0, n, n)
    batcher = Batcher(a, b, batch_size=1)
    for i in range(n):
        assert batcher.has_next_batch == True
        batch_a, batch_b =  batcher.next_batch()
        assert np.all(np.equal(batch_a, a[i, :]))
        assert np.all(np.equal(batch_b, b[i]))
    assert batcher.has_next_batch == False


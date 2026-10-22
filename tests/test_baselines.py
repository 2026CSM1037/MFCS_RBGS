"""Sanity tests for BCGS, BMGS, BCGS2: W ~ QR, Q^T Q ~ I, R upper triangular."""
import numpy as np
import pytest

from src.baselines import bcgs, bcgs2, bmgs
from src.metrics import factorization_error, orthogonality_loss


@pytest.mark.parametrize("method", [bcgs, bmgs, bcgs2])
def test_random_matrix(method):
    rng = np.random.default_rng(0)
    W = rng.standard_normal((300, 40))
    Q, R, info = method(W, 10)
    assert Q.shape == (300, 40) and R.shape == (40, 40)
    assert factorization_error(W, Q, R) < 1e-12
    assert orthogonality_loss(Q) < 1e-10
    assert np.allclose(R, np.triu(R))
    assert np.all(np.diag(R) > 0)
    assert "name" in info


def test_bad_block_size():
    with pytest.raises(ValueError):
        bcgs(np.ones((50, 10)), 3)
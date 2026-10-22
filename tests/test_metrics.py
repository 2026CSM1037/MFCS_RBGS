"""Sanity tests for metrics.py."""
import numpy as np

from src.metrics import (cond_Q, cond_per_iteration, certificates,
                         empirical_epsilon, factorization_error,
                         orthogonality_loss, relative_residual, timed,
                         theorem_3_2_bounds)


def _orthonormal(n=200, m=20, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, m)))
    return Q


def test_orthonormal_Q_is_perfect():
    Q = _orthonormal()
    assert abs(cond_Q(Q) - 1) < 1e-10
    assert orthogonality_loss(Q) < 1e-12


def test_scaled_Q_has_known_condition_number():
    Q = _orthonormal()
    Q[:, 0] *= 10.0
    assert abs(cond_Q(Q) - 10.0) < 1e-8


def test_factorization_error_zero_for_exact():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((100, 10))
    Q, R = np.linalg.qr(W)
    assert factorization_error(W, Q, R) < 1e-13


def test_cond_per_iteration_length():
    Q = _orthonormal(m=20)
    assert cond_per_iteration(Q, 5).shape == (4,)


def test_certificates_exact_sketch():
    rng = np.random.default_rng(2)
    W = rng.standard_normal((3000, 12))
    Q, R = np.linalg.qr(W)
    Theta = rng.standard_normal((1500, 3000)) / np.sqrt(1500)
    S, P = Theta @ Q, Theta @ W
    c = certificates(S, P, R)
    assert c["Delta_tilde"] < 1e-12      # P = S R exactly
    assert c["Delta"] < 1.0              # sketch of orthonormal Q is near-orthonormal


def test_empirical_epsilon_shrinks_with_k():
    Q = _orthonormal(n=2000, m=10)
    rng = np.random.default_rng(3)
    eps = []
    for k in (40, 800):
        Theta = rng.standard_normal((k, 2000)) / np.sqrt(k)
        eps.append(empirical_epsilon(Q, Theta))
    assert eps[1] < eps[0]


def test_bounds_and_residual_and_timer():
    lo, hi = theorem_3_2_bounds(0.1, 0.05, 1e-7)
    assert lo < 1 < hi
    A = np.eye(5) * 2
    X = np.ones((5, 2))
    B = 2 * np.ones((5, 2))
    assert relative_residual(A, X, B) < 1e-15
    out, t = timed(sum, [1, 2, 3])
    assert out == 6 and t >= 0
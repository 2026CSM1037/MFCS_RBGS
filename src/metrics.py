"""Measurements shared by all experiments.

Every function returns a float, an array, or a plain dict, so results can be
written straight to CSV. All norms are computed in float64 even when Q was
produced in float32.
"""
import time

import numpy as np


def _f64(X):
    return np.asarray(X, dtype=np.float64)


# ---------------------------------------------------------------- Q quality
def singular_values(Q):
    """Singular values of Q, largest first."""
    return np.linalg.svd(_f64(Q), compute_uv=False)


def cond_Q(Q):
    """Condition number sigma_max / sigma_min (about 1 for orthonormal Q)."""
    s = singular_values(Q)
    return float(s[0] / s[-1])


def orthogonality_loss(Q):
    """||I - Q^T Q||_2 : distance of Q from having orthonormal columns."""
    Q = _f64(Q)
    return float(np.linalg.norm(np.eye(Q.shape[1]) - Q.T @ Q, 2))


def factorization_error(W, Q, R, norm="fro"):
    """||W - Q R|| / ||W||  (Frobenius by default, as in Theorem 3.2)."""
    W, Q, R = _f64(W), _f64(Q), _f64(R)
    return float(np.linalg.norm(W - Q @ R, norm) / np.linalg.norm(W, norm))


def cond_per_iteration(Q, mp):
    """cond of the first i*mp columns of Q for i = 1..p (Figure 1a)."""
    p = Q.shape[1] // mp
    return np.array([cond_Q(Q[:, : i * mp]) for i in range(1, p + 1)])


def orthogonality_per_iteration(Q, mp):
    """||I - Q^T Q|| of the first i*mp columns for i = 1..p."""
    p = Q.shape[1] // mp
    return np.array([orthogonality_loss(Q[:, : i * mp]) for i in range(1, p + 1)])


def factorization_error_per_iteration(W, Q, R, mp):
    """||W_{1:i} - Q_{1:i} R_{1:i,1:i}|| / ||W_{1:i}|| for i = 1..p (Figure 1b)."""
    p = Q.shape[1] // mp
    out = []
    for i in range(1, p + 1):
        c = i * mp
        out.append(factorization_error(W[:, :c], Q[:, :c], R[:c, :c]))
    return np.array(out)


# --------------------------------------------- sketch-based (Theorem 3.2)
def certificates(S, P, R):
    """Delta = ||I - S^T S||_F and Delta_tilde = ||P - S R||_F / ||P||_F.

    S = sketch of Q (k x m), P = sketch of W (k x m), R = m x m factor.
    """
    S, P, R = _f64(S), _f64(P), _f64(R)
    delta = np.linalg.norm(np.eye(S.shape[1]) - S.T @ S, "fro")
    delta_tilde = np.linalg.norm(P - S @ R, "fro") / np.linalg.norm(P, "fro")
    return {"Delta": float(delta), "Delta_tilde": float(delta_tilde)}


def _sketch(Theta, X):
    """Apply a sketch: dense ndarray directly, otherwise via src.sketch."""
    if isinstance(Theta, np.ndarray):
        return Theta @ _f64(X)
    from src.sketch import apply_sketch   # teammate's module
    return apply_sketch(Theta, _f64(X))


def empirical_epsilon(Q, Theta):
    """Estimate how faithful Theta is on range(Q).

    Orthonormalize Q (so that exact Gram matrix = I), sketch it, and measure
    ||I - (Theta Q)^T (Theta Q)||_2. This is the epsilon in the embedding
    definition, restricted to range(Q).
    """
    Qo, _ = np.linalg.qr(_f64(Q))
    S = _sketch(Theta, Qo)
    return float(np.linalg.norm(np.eye(S.shape[1]) - S.T @ S, 2))


def theorem_3_2_bounds(eps, delta, u_crs):
    """Predicted (lower, upper) bounds on the singular values of Q (Thm 3.2)."""
    lower = (1 + eps) ** -0.5 * (1 - delta - 0.1 * u_crs)
    upper = (1 - eps) ** -0.5 * (1 + delta + 0.1 * u_crs)
    return float(lower), float(upper)


def theorem_3_2_factorization_bound(u_crs, m, normW_F=1.0):
    """Predicted bound 4 u_crs m^{3/2} ||W||_F on ||W - Q R||_F."""
    return float(4 * u_crs * m ** 1.5 * normW_F)


# ----------------------------------------------- GMRES / Rayleigh-Ritz
def relative_residual(A, X, B):
    """max_j ||A x_j - b_j|| / ||b_j|| (Section 5.2)."""
    X, B = _f64(X), _f64(B)
    if X.ndim == 1:
        X, B = X[:, None], B[:, None]
    res = np.linalg.norm(A @ X - B, axis=0) / np.linalg.norm(B, axis=0)
    return float(res.max())


def eig_residuals(A, U, mu):
    """||A u_j - mu_j u_j|| / ||mu_j u_j|| for every approximate eigenpair (5.3)."""
    U, mu = _f64(U), np.asarray(mu)
    R = A @ U - U * mu
    return np.linalg.norm(R, axis=0) / np.linalg.norm(U * mu, axis=0)


# --------------------------------------------------------------- utilities
def timed(fn, *args, **kwargs):
    """Return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def summary(W, Q, R):
    """One dict with the three headline numbers, ready for a CSV row."""
    return {
        "cond_Q": cond_Q(Q),
        "orth_loss": orthogonality_loss(Q),
        "fact_error": factorization_error(W, Q, R),
    }
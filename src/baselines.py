"""Standard block Gram-Schmidt baselines, written from scratch.

All three methods share one outer loop and differ only in how the earlier
blocks are removed from the new block (the 'projection' step):

    BCGS   one pass,  all earlier blocks at once (classical)
    BMGS   one pass,  earlier blocks one at a time (modified)
    BCGS2  two passes of BCGS (re-orthogonalization)

Each function returns (Q, R, info):
    Q    n x m, dtype as requested
    R    m x m, float64, upper triangular with positive diagonal
    info dict with the method name and working precision

Inter-block QR (orthonormalizing the cleaned block itself) uses Householder QR
from NumPy, always in float64, as in the paper's experiments. Only the
projection step runs in the working precision `dtype`.
"""
import numpy as np


def _householder_qr(X):
    """Householder QR (float64) with the diagonal of R made positive."""
    Q, R = np.linalg.qr(np.asarray(X, dtype=np.float64))
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    return Q * s, s[:, None] * R


def _bcgs_project(Qold, Wi, mp):
    """One classical pass: R = Qold^T Wi ; Q' = Wi - Qold R."""
    Rold = Qold.T @ Wi
    return Wi - Qold @ Rold, Rold


def _bcgs2_project(Qold, Wi, mp):
    """Two classical passes; the coefficients of both passes are added."""
    R1 = Qold.T @ Wi
    Qp = Wi - Qold @ R1
    R2 = Qold.T @ Qp
    Qp = Qp - Qold @ R2
    return Qp, R1 + R2


def _bmgs_project(Qold, Wi, mp):
    """One modified pass: earlier blocks are removed one at a time."""
    nb = Qold.shape[1] // mp
    Qp = Wi.copy()
    Rold = np.zeros((Qold.shape[1], Wi.shape[1]), dtype=Wi.dtype)
    for j in range(nb):
        s = slice(j * mp, (j + 1) * mp)
        Rj = Qold[:, s].T @ Qp
        Qp = Qp - Qold[:, s] @ Rj
        Rold[s] = Rj
    return Qp, Rold


def _block_gram_schmidt(W, mp, project, dtype, name):
    n, m = W.shape
    if m % mp != 0:
        raise ValueError("number of columns must be a multiple of the block size")
    if m > n:
        raise ValueError("W must be tall (m <= n)")
    p = m // mp

    Q = np.zeros((n, m), dtype=dtype)
    R = np.zeros((m, m), dtype=np.float64)
    for i in range(p):
        cur = slice(i * mp, (i + 1) * mp)
        old = slice(0, i * mp)
        Wi = np.asarray(W[:, cur], dtype=dtype)
        if i == 0:
            Qp = Wi
        else:
            Qp, Rold = project(Q[:, old], Wi, mp)
            R[old, cur] = Rold
        Qi, Rii = _householder_qr(Qp)      # inter-block QR in float64
        Q[:, cur] = Qi.astype(dtype)
        R[cur, cur] = Rii
    return Q, R, {"name": name, "dtype": str(np.dtype(dtype))}


def bcgs(W, mp, dtype=np.float64):
    """Block classical Gram-Schmidt."""
    return _block_gram_schmidt(W, mp, _bcgs_project, dtype, "BCGS")


def bmgs(W, mp, dtype=np.float64):
    """Block modified Gram-Schmidt."""
    return _block_gram_schmidt(W, mp, _bmgs_project, dtype, "BMGS")


def bcgs2(W, mp, dtype=np.float64):
    """Block classical Gram-Schmidt with one re-orthogonalization."""
    return _block_gram_schmidt(W, mp, _bcgs2_project, dtype, "BCGS2")
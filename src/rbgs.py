"""Randomized Block Gram-Schmidt (RBGS), Algorithm 2.1 of

    O. Balabanov and L. Grigori, "Randomized block Gram-Schmidt process for
    the solution of linear systems and eigenvalue problems", arXiv:2111.14641.

Usage (same style as src/baselines.py):

    from src.sketch import Sketch
    from src.rbgs import rbgs

    theta = Sketch("srht", k=400, n=W.shape[0], seed=0)
    Q, R, info = rbgs(W, mp=10, sketch=theta, dtype=np.float32)
    info["S"], info["P"]              # sketches Theta Q and Theta W
    info["Delta"], info["Delta_tilde"]  # a posteriori certificates (step 6)

The algorithm, for each block i = 1..p (W = [W_1, ..., W_p], m = p * mp):

    1. P_i        = Theta W_i                              fine precision
    2. R_{1:i-1,i} = argmin_Y || S_{1:i-1} Y - P_i ||_F     fine precision
    3. Q'_i       = W_i - Q_{1:i-1} R_{1:i-1,i}            COARSE precision
    4. Q'_i       = Q_i R_{i,i}, Q_i orthonormal w.r.t. <Theta., Theta.>
    5. S_i        = Theta Q_i                              fine precision

Step 3 is the only expensive n-dimensional operation, so it is the one done in
the cheap (coarse) working precision `dtype`. Steps 1, 2, 5 work on small
k x m sketches and use `fine_dtype`. As in the paper's experiments (Section 5)
and in src/baselines.py, the inter-block QR of step 4 always runs in float64.

The result: S = Theta Q has orthonormal columns, so Q is orthonormal in the
SKETCHED inner product and (if Theta is an eps-embedding) well-conditioned in
the ordinary sense, cond(Q) <= sqrt((1+eps)/(1-eps)), but Q^T Q is not I.

Options
-------
qr_method (step 4-5, paper Section 2.2):
    "l2_rcholesky"        Algorithm 2.5, Householder QR then RCholeskyQR
                          (used in all of the paper's experiments; default)
    "rcholesky"           Algorithm 2.3, RCholeskyQR
    "implicit_rcholesky"  Algorithm 2.4, RCholeskyQR with S'_i = P_i - S R
                          instead of a new sketch (one synchronization)
    "rgs"                 Section 2.2.1, single-vector RGS (column by column)

ls_solver (step 2, paper Section 2.1 and supplement):
    "householder"  direct solve via Householder QR of S_{1:i-1} (default)
    "richardson"   X <- X + S^T (P - S X), ls_iters steps (default 5, Sec 5.2)
    "cg"           CG on S^T S X = S^T P, ls_iters steps (default 20, Sec 5.1)

    Richardson converges only while the earlier sketch is nearly orthonormal
    (||I - S^T S||_2 < 1). On a numerically rank-deficient W (e.g. the hard
    matrix of Section 5.1 at large m in float32) it diverges, which is why the
    paper uses Householder or CG there and Richardson only for GMRES (5.2).
    ls_iters=1 is BCGS in the sketched inner product; each extra iteration
    acts like one re-orthogonalization.

Precision note: src.sketch.Sketch always multiplies in float64, so with
fine_dtype=float32 the sketches are computed in float64 and then rounded to
float32. This is slightly more accurate than true float32 sketching.
"""
import numpy as np
from scipy.linalg import solve_triangular

from src.sketch import PAPER_SKETCHED_LS_RICHARDSON_STEPS, Sketch, apply_sketch

QR_METHODS = ("l2_rcholesky", "rcholesky", "implicit_rcholesky", "rgs")
LS_SOLVERS = ("householder", "richardson", "cg")
DEFAULT_LS_ITERS = {"richardson": PAPER_SKETCHED_LS_RICHARDSON_STEPS,  # 5
                    "cg": 20}                                          # Sec 5.1


# ------------------------------------------------------------------ helpers
def _sketch_rows(sketch):
    """Return (k, n) of a Sketch object or an explicit Theta matrix."""
    if isinstance(sketch, Sketch):
        return sketch.k, sketch.n
    theta = np.asarray(sketch)
    if theta.ndim != 2:
        raise ValueError("sketch must be a Sketch or a 2-D Theta matrix")
    return theta.shape


def _apply(sketch, X, fine_dtype):
    """Theta @ X, returned in the fine precision."""
    return np.asarray(apply_sketch(sketch, X), dtype=fine_dtype)


def _positive_qr_r(X):
    """R factor of a Householder QR of X (float64) with positive diagonal."""
    R = np.linalg.qr(np.asarray(X, dtype=np.float64), mode="r")
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    return s[:, None] * R


def _positive_qr(X):
    """Householder QR (float64) with positive diagonal of R."""
    Qx, R = np.linalg.qr(np.asarray(X, dtype=np.float64))
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    return Qx * s, s[:, None] * R


def _right_solve_upper(X, R):
    """X R^{-1} for upper triangular R (forward substitution), in float64."""
    return solve_triangular(R, np.asarray(X, dtype=np.float64).T,
                            trans="T", lower=False).T


# ------------------------------------------------- step 2: sketched LS
def _solve_ls(S_old, P_i, solver, iters):
    """Approximate argmin_Y ||S_old Y - P_i||_F (all in S_old's dtype)."""
    if solver == "householder":
        Qs, Rs = np.linalg.qr(S_old)
        return solve_triangular(Rs, Qs.T @ P_i, lower=False)

    X = np.zeros((S_old.shape[1], P_i.shape[1]), dtype=S_old.dtype)
    if solver == "richardson":
        # Exact for orthonormal S_old; 1 step = BCGS in the sketched product.
        for _ in range(iters):
            X = X + S_old.T @ (P_i - S_old @ X)
        return X

    # CG on the normal equations, one independent CG per column of P_i.
    Rk = S_old.T @ P_i                 # residual of S^T S X = S^T P at X = 0
    D = Rk.copy()
    rr = np.sum(Rk * Rk, axis=0)
    tiny = np.finfo(S_old.dtype).tiny
    for _ in range(iters):
        if not np.any(rr > tiny):
            break
        SD = S_old @ D
        AD = S_old.T @ SD
        dAd = np.sum(D * AD, axis=0)
        alpha = np.where(dAd > tiny, rr / np.where(dAd > tiny, dAd, 1), 0)
        X = X + D * alpha
        Rk = Rk - AD * alpha
        rr_new = np.sum(Rk * Rk, axis=0)
        beta = np.where(rr > tiny, rr_new / np.where(rr > tiny, rr, 1), 0)
        D = Rk + D * beta
        rr = rr_new
    return X


# -------------------------------------------- steps 4-5: inter-block QR
def _rcholesky(Qp, Sp):
    """Algorithm 2.3 given Q'_i and its sketch S'_i: R = qr_R(S'), Q = Q' R^-1."""
    Rii = _positive_qr_r(Sp)
    return _right_solve_upper(Qp, Rii), Rii


def _rgs_block(Qp, sketch):
    """Section 2.2.1: single-vector RGS on one block (float64).

    Column by column: project out earlier columns using sketched least
    squares, then normalize with the sketched norm.
    """
    n, b = Qp.shape
    V = np.asarray(Qp, dtype=np.float64)
    Qb = np.zeros((n, b))
    Sb = None
    Rb = np.zeros((b, b))
    for j in range(b):
        v = V[:, j]
        p = np.asarray(apply_sketch(sketch, v), dtype=np.float64)
        if j > 0:
            r = np.linalg.lstsq(Sb[:, :j], p, rcond=None)[0]
            Rb[:j, j] = r
            v = v - Qb[:, :j] @ r
        s = np.asarray(apply_sketch(sketch, v), dtype=np.float64)
        Rb[j, j] = np.linalg.norm(s)
        Qb[:, j] = v / Rb[j, j]
        if Sb is None:
            Sb = np.zeros((s.shape[0], b))
        Sb[:, j] = s / Rb[j, j]
    return Qb, Rb


# ---------------------------------------------------------------- RBGS
class RBGSProcess:
    """RBGS run one block at a time (needed by block Arnoldi in krylov.py).

    proc = RBGSProcess(n, m, mp, sketch, ...)
    proc.step(W_i)  -> Q_i ; after i steps, proc.Q[:, :i*mp], proc.R[:i*mp, :i*mp]
    """

    def __init__(self, n, m, mp, sketch, dtype=np.float64, fine_dtype=np.float64,
                 qr_method="l2_rcholesky", ls_solver="householder", ls_iters=None):
        if mp <= 0 or m % mp != 0:
            raise ValueError("number of columns must be a multiple of the block size")
        if m > n:
            raise ValueError("W must be tall (m <= n)")
        k, n_theta = _sketch_rows(sketch)
        if n_theta != n:
            raise ValueError(f"sketch expects {n_theta} rows but W has {n}")
        if k < m:
            raise ValueError(f"sketch size k={k} must be at least m={m}")
        if qr_method not in QR_METHODS:
            raise ValueError(f"qr_method must be one of {QR_METHODS}")
        if ls_solver not in LS_SOLVERS:
            raise ValueError(f"ls_solver must be one of {LS_SOLVERS}")
        if ls_iters is None:
            ls_iters = DEFAULT_LS_ITERS.get(ls_solver)
        elif ls_solver != "householder" and ls_iters < 1:
            raise ValueError("ls_iters must be at least 1")

        self.n, self.m, self.mp, self.p = n, m, mp, m // mp
        self.sketch = sketch
        self.dtype, self.fine_dtype = dtype, fine_dtype
        self.qr_method, self.ls_solver, self.ls_iters = qr_method, ls_solver, ls_iters
        self.Q = np.zeros((n, m), dtype=dtype)
        self.R = np.zeros((m, m), dtype=np.float64)
        self.S = np.zeros((k, m), dtype=fine_dtype)
        self.P = np.zeros((k, m), dtype=fine_dtype)
        self.i = 0                                   # blocks processed so far

    def step(self, Wi):
        """Process the next block W_i (n x mp) and return Q_i."""
        if self.i >= self.p:
            raise RuntimeError("all blocks have already been processed")
        Wi = np.asarray(Wi)
        if Wi.shape != (self.n, self.mp):
            raise ValueError(f"block must be {self.n} x {self.mp}, got {Wi.shape}")
        mp, i, sketch, fine = self.mp, self.i, self.sketch, self.fine_dtype
        Q, R, S, P = self.Q, self.R, self.S, self.P
        cur, old = slice(i * mp, (i + 1) * mp), slice(0, i * mp)

        # 1. sketch the new block (fine)
        P[:, cur] = _apply(sketch, Wi, fine)

        # 2-3. sketched least squares (fine), projection (coarse)
        Wc = np.asarray(Wi, dtype=self.dtype)
        if i == 0:
            Qp, R_old = Wc, None
        else:
            R_old = _solve_ls(S[:, old], P[:, cur], self.ls_solver, self.ls_iters)
            R[old, cur] = R_old
            Qp = Wc - Q[:, old] @ np.asarray(R_old, dtype=self.dtype)

        # 4. QR of Q'_i w.r.t. <Theta., Theta.>  (float64)
        if self.qr_method == "rcholesky":
            Qi, Rii = _rcholesky(Qp, _apply(sketch, Qp, np.float64))
        elif self.qr_method == "implicit_rcholesky":
            Sp = np.asarray(P[:, cur], dtype=np.float64)
            if i > 0:
                Sp = Sp - (np.asarray(S[:, old], dtype=np.float64)
                           @ np.asarray(R_old, dtype=np.float64))
            Qi, Rii = _rcholesky(Qp, Sp)
        elif self.qr_method == "l2_rcholesky":
            Qstar, Rprime = _positive_qr(Qp)                     # l2-QR
            Qi, Rpp = _rcholesky(Qstar, _apply(sketch, Qstar, np.float64))
            Rii = Rpp @ Rprime
        else:  # "rgs"
            Qi, Rii = _rgs_block(Qp, sketch)

        Q[:, cur] = Qi.astype(self.dtype)
        R[cur, cur] = Rii

        # 5. sketch of the stored Q_i (fine)
        S[:, cur] = _apply(sketch, Q[:, cur], fine)
        self.i += 1
        return Q[:, cur]

    def info(self):
        """Settings, sketches and the step-6 certificates for the blocks done so far."""
        c = self.i * self.mp
        S64 = self.S[:, :c].astype(np.float64)
        P64 = self.P[:, :c].astype(np.float64)
        delta = np.linalg.norm(np.eye(c) - S64.T @ S64, "fro")
        normP = np.linalg.norm(P64, "fro")
        delta_tilde = np.linalg.norm(P64 - S64 @ self.R[:c, :c], "fro") / normP if c else 0.0
        return {
            "name": "RBGS",
            "dtype": str(np.dtype(self.dtype)),
            "fine_dtype": str(np.dtype(self.fine_dtype)),
            "qr_method": self.qr_method,
            "ls_solver": self.ls_solver,
            "ls_iters": self.ls_iters,
            "S": self.S[:, :c],
            "P": self.P[:, :c],
            "Delta": float(delta),
            "Delta_tilde": float(delta_tilde),
        }


def rbgs(W, mp, sketch, dtype=np.float64, fine_dtype=np.float64,
         qr_method="l2_rcholesky", ls_solver="householder", ls_iters=None):
    """Randomized block Gram-Schmidt QR factorization W = Q R.

    Args:
        W:          n x m matrix, m divisible by mp, m <= n.
        mp:         block size (number of columns per block).
        sketch:     src.sketch.Sketch (k x n) or an explicit k x n Theta,
                    with m <= k.
        dtype:      coarse working precision of step 3 (np.float32 or
                    np.float64). Q is returned in this dtype.
        fine_dtype: precision of the sketch-sized steps 1, 2, 5.
                    dtype=fine_dtype=np.float32 gives the paper's "unique
                    precision" RBGS; dtype=float32, fine_dtype=float64 gives
                    its "multi-precision" RBGS.
        qr_method:  step 4-5 routine, one of QR_METHODS.
        ls_solver:  step 2 solver, one of LS_SOLVERS.
        ls_iters:   iterations for "richardson"/"cg" (default 5 / 20).

    Returns:
        Q:    n x m, dtype `dtype`.
        R:    m x m float64, upper triangular with positive diagonal.
        info: dict with "name", settings, "S" (= Theta Q), "P" (= Theta W),
              and the certificates "Delta" and "Delta_tilde" (step 6).
    """
    W = np.asarray(W)
    if W.ndim != 2:
        raise ValueError("W must be a 2-D matrix")
    n, m = W.shape
    proc = RBGSProcess(n, m, mp, sketch, dtype=dtype, fine_dtype=fine_dtype,
                       qr_method=qr_method, ls_solver=ls_solver, ls_iters=ls_iters)
    for i in range(proc.p):
        proc.step(W[:, i * mp:(i + 1) * mp])
    return proc.Q, proc.R, proc.info()
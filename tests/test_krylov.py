"""Randomized block Krylov methods (Section 4 of Balabanov & Grigori).

block_arnoldi       Algorithm 4.1: Arnoldi with RBGS, or with BCGS / BMGS / BCGS2
gmres               restarted block GMRES, eq. (4.2)              (Section 5.2)
rr                  restarted Rayleigh-Ritz, Algorithm 4.2        (Section 5.3)
subspace_iteration  power-iteration baseline of Section 5.3
ilu_preconditioner  ILU + reverse Cuthill-McKee preconditioner of Section 5.2

Products with A and the small reduced problems are done in float64; the
Gram-Schmidt process runs in `dtype`, as in the paper's experiments.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import LinearOperator, aslinearoperator, spilu

from src.baselines import (_bcgs2_project, _bcgs_project, _bmgs_project,
                           _householder_qr)
from src.metrics import cond_Q, eig_residuals, relative_residual
from src.rbgs import RBGSProcess

METHODS = ("rbgs", "bcgs", "bmgs", "bcgs2")
_PROJECT = {"bcgs": _bcgs_project, "bmgs": _bmgs_project, "bcgs2": _bcgs2_project}


class _BGSProcess:
    """BCGS / BMGS / BCGS2 one block at a time, same steps as baselines.py."""

    def __init__(self, n, m, mp, method, dtype):
        if mp <= 0 or m % mp != 0:
            raise ValueError("number of columns must be a multiple of the block size")
        if m > n:
            raise ValueError("Krylov basis must be tall (m <= n)")
        self.mp, self.p, self.i = mp, m // mp, 0
        self.project, self.dtype, self.name = _PROJECT[method], dtype, method.upper()
        self.Q = np.zeros((n, m), dtype=dtype)
        self.R = np.zeros((m, m))

    def step(self, Wi):
        mp, i = self.mp, self.i
        cur, old = slice(i * mp, (i + 1) * mp), slice(0, i * mp)
        Wi = np.asarray(Wi, dtype=self.dtype)
        if i == 0:
            Qp = Wi
        else:
            Qp, Rold = self.project(self.Q[:, old], Wi, mp)
            self.R[old, cur] = Rold
        Qi, Rii = _householder_qr(Qp)
        self.Q[:, cur] = Qi.astype(self.dtype)
        self.R[cur, cur] = Rii
        self.i += 1
        return self.Q[:, cur]

    def info(self):
        return {"name": self.name, "dtype": str(np.dtype(self.dtype))}


def _new_process(method, n, m, mp, sketch, dtype, rbgs_kwargs):
    if method == "rbgs":
        if sketch is None:
            raise ValueError("method='rbgs' needs a sketch")
        # Sections 5.2-5.3 solve step 2 with 5 Richardson iterations; a direct
        # solver would refactorize the growing sketch S at every iteration.
        rbgs_kwargs = {"ls_solver": "richardson", **rbgs_kwargs}
        return RBGSProcess(n, m, mp, sketch, dtype=dtype, **rbgs_kwargs)
    if method in _PROJECT:
        return _BGSProcess(n, m, mp, method, dtype)
    raise ValueError(f"method must be one of {METHODS}")


def _matmat(Aop, X):
    return Aop.matmat(np.asarray(X, dtype=np.float64))


def _next_block(Aop, proc):
    """W_i = A Q_{i-1} (Algorithm 4.1, step 2)."""
    mp = proc.mp
    return _matmat(Aop, proc.Q[:, (proc.i - 1) * mp:proc.i * mp])


def block_arnoldi(A, B, p, method="rbgs", sketch=None, dtype=np.float64,
                  **rbgs_kwargs):
    """Block Arnoldi with p blocks: A Q[:, :-mp] = Q H.

    Returns Q (n x p*mp), the block Hessenberg H (p*mp x (p-1)*mp) and info.
    With method="rbgs", Q is orthonormal w.r.t. <Theta., Theta.>.
    """
    Aop = aslinearoperator(A)
    B = np.asarray(B, dtype=np.float64)
    n, mp = B.shape
    proc = _new_process(method, n, p * mp, mp, sketch, dtype, rbgs_kwargs)
    proc.step(B)
    for _ in range(p - 1):
        proc.step(_next_block(Aop, proc))
    return proc.Q, proc.R[:, mp:], proc.info()


# -------------------------------------------------------------------- GMRES
def gmres(A, B, method="rbgs", sketch=None, restart=30, max_iter=300, tol=1e-10,
          M=None, dtype=np.float64, track_cond=False, **rbgs_kwargs):
    """Restarted block GMRES for A X = B.

    Each cycle builds restart+1 Krylov blocks from the current residual, so
    the sketch needs k >= (restart+1)*mp rows; in practice take k a few times
    larger (the paper uses k = 5000 or 7500 for 3100 columns). M is an optional right
    preconditioner ~ A^{-1} (e.g. from ilu_preconditioner).

    info["residuals"][j] = max_i ||A x_i - b_i|| / ||b_i|| after iteration j+1
    (the error of Figure 2a); info["cond_Q"] is cond of the Krylov basis per
    iteration (Figure 2b), if track_cond=True.
    """
    Aop = aslinearoperator(A)
    if M is not None:
        Minv = aslinearoperator(M)
        Aop = Aop.dot(Minv)
    B = np.asarray(B, dtype=np.float64)
    n, mp = B.shape
    p = restart + 1

    Y = np.zeros((n, mp))
    history, conds = [], []
    it, cycles, res = 0, 0, np.inf
    while it < max_iter and res > tol:
        proc = _new_process(method, n, p * mp, mp, sketch, dtype, rbgs_kwargs)
        proc.step(B - _matmat(Aop, Y))
        cycles += 1
        U = Y
        for j in range(1, p):
            proc.step(_next_block(Aop, proc))
            it += 1
            c = (j + 1) * mp
            Z = np.linalg.lstsq(proc.R[:c, mp:c], proc.R[:c, :mp], rcond=None)[0]
            U = Y + np.asarray(proc.Q[:, :j * mp], dtype=np.float64) @ Z
            res = relative_residual(Aop, U, B)
            history.append(res)
            if track_cond:
                conds.append(cond_Q(proc.Q[:, :c]))
            if res <= tol or it >= max_iter:
                break
        Y = U

    X = _matmat(Minv, Y) if M is not None else Y
    info = {
        "method": method,
        "residuals": np.array(history),
        "iterations": it,
        "cycles": cycles,
        "converged": bool(res <= tol),
        "cond_Q": np.array(conds) if track_cond else None,
    }
    return X, info


# ------------------------------------------------------------ eigenvalues
_ORDER = {
    "LM": lambda mu: -np.abs(mu),
    "SM": lambda mu: np.abs(mu),
    "LR": lambda mu: -mu.real,
    "SR": lambda mu: mu.real,
}


def _ritz(Hsq, mp, which):
    """mp extreme eigenpairs of the square Hessenberg block, most extreme first."""
    if which not in _ORDER:
        raise ValueError(f"which must be one of {tuple(_ORDER)}")
    mu, Yv = np.linalg.eig(Hsq)
    idx = np.argsort(_ORDER[which](mu), kind="stable")[:mp]
    return mu[idx], Yv[:, idx]


def _real_basis(mu, Yv, mp):
    """Real mp-column basis of span(Yv); a complex pair contributes Re and Im."""
    if not np.iscomplexobj(Yv):
        return Yv
    cols, used = [], set()
    tol = 1e-12 * max(1.0, np.abs(mu).max())
    for j, lam in enumerate(mu):
        if j in used:
            continue
        if abs(lam.imag) <= tol:
            cols.append(Yv[:, j].real)
            continue
        cols += [Yv[:, j].real, Yv[:, j].imag]
        partner = np.where(np.abs(mu - np.conj(lam)) <= tol)[0]
        used.update(int(q) for q in partner if q != j)
    return np.column_stack(cols[:mp])


def _eig_error(Aop, U, mu, frac):
    """Max relative residual of the first `frac` of the Ritz pairs (Section 5.3)."""
    nf = max(1, int(frac * len(mu)))
    U, mu = U[:, :nf], mu[:nf]
    if not np.iscomplexobj(U):
        return float(eig_residuals(Aop, U, mu).max())
    AU = _matmat(Aop, U.real) + 1j * _matmat(Aop, U.imag)   # metrics casts to real
    return float((np.linalg.norm(AU - U * mu, axis=0)
                  / np.linalg.norm(U * mu, axis=0)).max())


def _normalize(U):
    return U / np.linalg.norm(U, axis=0)


def rr(A, B, p, n_cycles, method="rbgs", sketch=None, which="LM", frac=0.8,
       dtype=np.float64, track=True, track_cond=False, **rbgs_kwargs):
    """Restarted randomized Rayleigh-Ritz (Algorithm 4.2).

    Each cycle runs block Arnoldi with p blocks (p-1 iterations) from B and
    restarts with the mp extreme Ritz vectors. With track=True the error
    max_{j <= frac*mp} ||A u_j - mu_j u_j|| / ||mu_j u_j|| is recorded after
    every iteration (Figure 3a), otherwise once per cycle. With track_cond=True,
    info["cond_Q"] holds cond of the Krylov basis at the same points (Figure 3b).

    Returns the Ritz values mu (most extreme first), unit-norm Ritz vectors U
    and info (with "residual_estimate" = ||H_{p,1:p-1} y|| of step 3).
    """
    Aop = aslinearoperator(A)
    B = np.asarray(B, dtype=np.float64)
    n, mp = B.shape
    history, conds = [], []
    for _ in range(n_cycles):
        proc = _new_process(method, n, p * mp, mp, sketch, dtype, rbgs_kwargs)
        proc.step(B)
        for j in range(1, p):
            proc.step(_next_block(Aop, proc))
            if not track and j < p - 1:
                continue
            c = (j + 1) * mp
            H = proc.R[:c, mp:c]
            mu, Yv = _ritz(H[:j * mp], mp, which)
            Qj = np.asarray(proc.Q[:, :j * mp], dtype=np.float64)
            U = Qj @ Yv
            history.append(_eig_error(Aop, U, mu, frac))
            if track_cond:
                conds.append(cond_Q(proc.Q[:, :c]))
        estimate = np.linalg.norm(H[j * mp:] @ Yv, axis=0)
        B = Qj @ _real_basis(mu, Yv, mp)

    info = {
        "method": method,
        "errors": np.array(history),
        "iterations": n_cycles * (p - 1),
        "residual_estimate": estimate,
        "cond_Q": np.array(conds) if track_cond else None,
    }
    return np.real_if_close(mu), np.real_if_close(_normalize(U)), info


def subspace_iteration(A, B, n_iter, which="LM", frac=0.8):
    """Subspace (block power) iteration with a Rayleigh-Ritz step each iteration."""
    Aop = aslinearoperator(A)
    X, _ = np.linalg.qr(np.asarray(B, dtype=np.float64))
    mp = X.shape[1]
    history = []
    for _ in range(n_iter):
        AX = _matmat(Aop, X)
        mu, Yv = _ritz(X.T @ AX, mp, which)
        U = X @ Yv
        history.append(_eig_error(Aop, U, mu, frac))
        X, _ = np.linalg.qr(AX)
    info = {"method": "subspace_iteration", "errors": np.array(history),
            "iterations": n_iter}
    return np.real_if_close(mu), np.real_if_close(_normalize(U)), info


# ---------------------------------------------------------- preconditioner
def ilu_preconditioner(A, drop_tol=1e-4, fill_factor=10.0, rcm=True):
    """M ~ A^{-1} from an incomplete LU of A after reverse Cuthill-McKee ordering.

    The paper uses ILU(0), which SciPy does not provide. This is threshold ILU
    (SciPy spilu / SuperLU ILUTP): entries below drop_tol are dropped and
    fill is capped at fill_factor * nnz(A). Note: a small fill_factor such as 1
    drops entries even inside the pattern of A, so it is weaker than ILU(0).
    Use it as gmres(A, B, M=ilu_preconditioner(A)).
    """
    A = sparse.csr_matrix(A)
    n = A.shape[0]
    perm = reverse_cuthill_mckee(A, symmetric_mode=False) if rcm else np.arange(n)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(n)
    ilu = spilu(A[perm][:, perm].tocsc(), drop_tol=drop_tol, fill_factor=fill_factor,
                permc_spec="NATURAL", diag_pivot_thresh=0.0)

    def solve(X):
        return ilu.solve(np.asarray(X, dtype=np.float64)[perm])[inv]

    return LinearOperator((n, n), matvec=solve, matmat=solve, dtype=np.float64)
"""Tests for src/krylov.py: block Arnoldi, GMRES, Rayleigh-Ritz, preconditioner.

Each test compares against an answer we know independently: the Arnoldi
identity, a direct sparse solve, or matrices built with known eigenvalues.
"""
import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import block_diag
from scipy.sparse.linalg import spsolve

from src.krylov import (METHODS, block_arnoldi, gmres, ilu_preconditioner, rr,
                        subspace_iteration)
from src.matrices import prepare_sparse_test_problem
from src.metrics import cond_Q, relative_residual
from src.sketch import Sketch


# ------------------------------------------------------------------ helpers
def _sketch(n, k, kind="rademacher"):
    return Sketch(kind, k=k, n=n, seed=0)


def _known_spectrum(top, n=400, seed=1):
    """Matrix Q D Q^T with eigenvalues `top` (2x2 blocks give complex pairs)
    followed by small eigenvalues in [0, 1]."""
    rng = np.random.default_rng(seed)
    blocks = [np.atleast_2d(b) for b in top]
    size = sum(b.shape[0] for b in blocks)
    D = block_diag(*blocks, np.diag(rng.uniform(0, 1, n - size)))
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return Q @ D @ Q.T


def _convection_diffusion(g=30):
    """2-D convection-diffusion matrix (n = g*g): GMRES is slow without a
    preconditioner and ILU helps a lot."""
    Id = sparse.identity(g)
    T = sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(g, g))
    C = sparse.diags([-1.0, 1.0], [-1, 1], shape=(g, g))
    return (sparse.kron(Id, T) + sparse.kron(T, Id) + 0.3 * sparse.kron(Id, C)).tocsr()


# ------------------------------------------------------------ block Arnoldi
@pytest.mark.parametrize("method", METHODS)
def test_arnoldi_identity(method):
    """A Q[:, :-mp] = Q H, with H block upper Hessenberg (Algorithm 4.1)."""
    n, mp, p = 600, 5, 6
    A, B = prepare_sparse_test_problem(n=n, mp=mp)
    Q, H, info = block_arnoldi(A, B, p, method=method, sketch=_sketch(n, 150))

    assert Q.shape == (n, p * mp) and H.shape == (p * mp, (p - 1) * mp)
    AQ = A @ Q[:, :-mp]
    assert np.linalg.norm(AQ - Q @ H) / np.linalg.norm(AQ) < 1e-12
    for j in range(p - 1):                     # zero below the first subdiagonal block
        assert np.all(H[(j + 2) * mp:, j * mp:(j + 1) * mp] == 0)


def test_rbgs_arnoldi_basis_is_sketch_orthonormal():
    """RBGS-Arnoldi: Theta Q is orthonormal, so Q is well-conditioned."""
    n, mp = 600, 5
    A, B = prepare_sparse_test_problem(n=n, mp=mp)
    Q, H, info = block_arnoldi(A, B, 6, sketch=_sketch(n, 150))
    assert info["Delta"] < 1e-10
    assert cond_Q(Q) < 10


# -------------------------------------------------------------------- GMRES
@pytest.mark.parametrize("method", ["rbgs", "bcgs2"])
def test_gmres_matches_direct_solve(method):
    n, mp = 600, 5
    A, B = prepare_sparse_test_problem(n=n, mp=mp)
    X, info = gmres(A, B, method=method, sketch=_sketch(n, 500), tol=1e-10)
    X_direct = np.column_stack([spsolve(A.tocsc(), B[:, j]) for j in range(mp)])

    assert info["converged"]
    assert relative_residual(A, X, B) < 1e-9
    assert np.linalg.norm(X - X_direct) / np.linalg.norm(X_direct) < 1e-8


def test_gmres_history_and_restarts():
    """With a short restart GMRES needs several cycles but still converges,
    and one residual is recorded per iteration."""
    n, mp = 600, 5
    A, B = prepare_sparse_test_problem(n=n, mp=mp)
    X, info = gmres(A, B, sketch=_sketch(n, 200), restart=5, tol=1e-8,
                    track_cond=True)
    res = info["residuals"]

    assert info["converged"] and info["cycles"] > 1
    assert len(res) == info["iterations"] == len(info["cond_Q"])
    assert res[-1] <= 1e-8 < res[0]
    assert relative_residual(A, X, B) <= 1e-8


def test_gmres_max_iter_is_respected():
    n, mp = 600, 5
    A, B = prepare_sparse_test_problem(n=n, mp=mp)
    X, info = gmres(A, B, sketch=_sketch(n, 500), tol=0.0, max_iter=4)
    assert info["iterations"] == 4 and not info["converged"]


def test_gmres_with_ilu_preconditioner():
    """ILU cuts the iteration count, and the returned X solves A X = B
    (not the preconditioned system)."""
    A = _convection_diffusion()
    n, mp = A.shape[0], 4
    B = np.random.default_rng(0).standard_normal((n, mp))
    sk = _sketch(n, 300)
    _, plain = gmres(A, B, sketch=sk, tol=1e-8, max_iter=400)
    X, prec = gmres(A, B, sketch=sk, tol=1e-8, max_iter=400, M=ilu_preconditioner(A))

    assert prec["converged"]
    assert prec["iterations"] < plain["iterations"]
    assert relative_residual(A, X, B) <= 1e-8


def test_ilu_is_exact_for_tridiagonal():
    """A tridiagonal matrix has no fill-in, so ILU(0) is its exact LU and
    the preconditioner is A^{-1} up to roundoff (with or without RCM)."""
    n = 300
    A = sparse.diags([-1.0, 4.0, -1.5], [-1, 0, 1], shape=(n, n)).tocsr()
    x = np.random.default_rng(0).standard_normal((n, 3))
    for rcm in (True, False):
        M = ilu_preconditioner(A, rcm=rcm)
        assert np.allclose(M @ (A @ x), x, atol=1e-10)


# ----------------------------------------------------------- eigenvalues
def test_rr_finds_known_real_eigenvalues():
    A = _known_spectrum([10.0, 9.0, 8.0, 7.0, 6.0])
    B = np.random.default_rng(0).standard_normal((400, 5))
    mu, U, info = rr(A, B, p=11, n_cycles=3, sketch=_sketch(400, 200), frac=1.0)

    assert np.allclose(mu, [10, 9, 8, 7, 6], atol=1e-10)
    assert info["errors"][-1] < 1e-10
    assert np.allclose(np.linalg.norm(U, axis=0), 1.0)
    assert np.allclose(A @ U, U * mu, atol=1e-8)


def test_rr_finds_known_complex_eigenvalues():
    """A non-symmetric matrix with eigenvalues 5 +- 1i, 4 +- 2i, 3.5."""
    A = _known_spectrum([[[5, 1], [-1, 5]], [[4, 2], [-2, 4]], 3.5])
    B = np.random.default_rng(0).standard_normal((400, 5))
    mu, U, info = rr(A, B, p=11, n_cycles=4, sketch=_sketch(400, 200), frac=1.0)

    expected = [5 + 1j, 5 - 1j, 4 + 2j, 4 - 2j, 3.5]
    assert all(np.abs(np.asarray(mu) - e).min() < 1e-8 for e in expected)
    assert info["errors"][-1] < 1e-8


def test_rr_errors_recorded_per_iteration():
    A = _known_spectrum([10.0, 9.0, 8.0, 7.0, 6.0])
    B = np.random.default_rng(0).standard_normal((400, 5))
    _, _, every = rr(A, B, p=6, n_cycles=2, sketch=_sketch(400, 200), track_cond=True)
    _, _, ends = rr(A, B, p=6, n_cycles=2, sketch=_sketch(400, 200), track=False)
    assert len(every["errors"]) == every["iterations"] == len(every["cond_Q"]) == 10
    assert np.all(every["cond_Q"] < 10)
    assert len(ends["errors"]) == 2 and ends["cond_Q"] is None


def test_subspace_iteration_finds_known_eigenvalues():
    A = _known_spectrum([10.0, 9.0, 8.0, 7.0, 6.0])
    B = np.random.default_rng(0).standard_normal((400, 5))
    mu, U, info = subspace_iteration(A, B, n_iter=150)
    assert np.allclose(mu, [10, 9, 8, 7, 6], atol=1e-6)
    assert info["errors"][-1] < info["errors"][0]


# ------------------------------------------------------------ bad input
def test_rbgs_requires_a_sketch():
    A, B = prepare_sparse_test_problem(n=200, mp=5)
    with pytest.raises(ValueError):
        block_arnoldi(A, B, 4, method="rbgs", sketch=None)


def test_unknown_method_rejected():
    A, B = prepare_sparse_test_problem(n=200, mp=5)
    with pytest.raises(ValueError):
        gmres(A, B, method="householder")


def test_sketch_too_small_for_krylov_basis():
    """The basis has (restart+1)*mp columns, so the sketch needs k >= that."""
    A, B = prepare_sparse_test_problem(n=600, mp=5)
    with pytest.raises(ValueError):
        gmres(A, B, sketch=_sketch(600, 100), restart=30)
"""Tests for src/rbgs.py (Randomized Block Gram-Schmidt, Algorithm 2.1).

Expected interface (same style as src/baselines.py):

    Q, R, info = rbgs(W, mp, sketch, dtype=np.float64)

    W       n x m input matrix, m divisible by the block size mp
    mp      block size (columns per block)
    sketch  a src.sketch.Sketch object (Theta, k x n, with m <= k <= n)
    dtype   working ("coarse") precision of step 3, e.g. np.float32
    Q       n x m basis, dtype = `dtype`
    R       m x m upper triangular, positive diagonal
    info    dict with at least
              "name": "RBGS"
              "S":    sketch of Q, Theta @ Q   (k x m)
              "P":    sketch of W, Theta @ W   (k x m)

What we check, from the paper (Balabanov & Grigori, arXiv:2111.14641):
  * W = QR and R is upper triangular                         (Algorithm 2.1)
  * S = Theta Q has orthonormal columns: Q is orthonormal
    in the SKETCHED inner product, not the ordinary one       (Section 2)
  * Q is still well-conditioned in the ordinary sense         (Corollary 1.4)
  * the singular-value and residual bounds of Theorem 3.2
  * on the paper's hard matrix in float32, RBGS stays stable
    while BCGS loses orthogonality                           (Section 5.1)
"""
import numpy as np
import pytest

from src.baselines import bcgs
from src.matrices import generate_hard_matrix
from src.metrics import (certificates, cond_Q, empirical_epsilon,
                         factorization_error, orthogonality_loss,
                         singular_values, theorem_3_2_bounds,
                         theorem_3_2_factorization_bound)
from src.sketch import Sketch

try:
    from src.rbgs import rbgs
except ImportError:
    pytest.skip("src/rbgs.py does not define rbgs() yet",
                allow_module_level=True)


SKETCH_TYPES = ["rademacher", "srht"]

# Unit roundoff u of each precision (the u_crs / u_fine of the paper).
UNIT_ROUNDOFF = {np.float64: 2.0 ** -53, np.float32: 2.0 ** -24}


# ------------------------------------------------------------------ helpers
def _random_W(n=2000, m=40, seed=0):
    """A well-conditioned random test matrix."""
    return np.random.default_rng(seed).standard_normal((n, m))


def _run(W, mp=10, kind="rademacher", k=None, dtype=np.float64, seed=1):
    """Run RBGS with a fresh sketch; return (Q, R, info, sketch)."""
    n, m = W.shape
    k = 10 * m if k is None else k
    sketch = Sketch(kind, k=k, n=n, seed=seed)
    Q, R, info = rbgs(W, mp, sketch, dtype=dtype)
    return Q, R, info, sketch


# ------------------------------------------------------- basic structure
@pytest.mark.parametrize("kind", SKETCH_TYPES)
def test_output_shapes_and_info(kind):
    W = _random_W()
    n, m = W.shape
    Q, R, info, sketch = _run(W, kind=kind)
    k = sketch.k
    assert Q.shape == (n, m)
    assert R.shape == (m, m)
    assert info["S"].shape == (k, m)
    assert info["P"].shape == (k, m)
    assert info["name"] == "RBGS"


def test_R_is_upper_triangular_with_positive_diagonal():
    Q, R, info, _ = _run(_random_W())
    assert np.allclose(R, np.triu(R))
    assert np.all(np.diag(R) > 0)


# ------------------------------------------------------ core properties
@pytest.mark.parametrize("kind", SKETCH_TYPES)
def test_W_equals_QR(kind):
    """The factorization must reproduce W (in float64, to roundoff level)."""
    W = _random_W()
    Q, R, info, _ = _run(W, kind=kind)
    assert factorization_error(W, Q, R) < 1e-12


@pytest.mark.parametrize("kind", SKETCH_TYPES)
def test_sketch_of_Q_is_orthonormal(kind):
    """Main RBGS property: S = Theta Q satisfies S^T S = I.

    Delta = ||I - S^T S||_F is the paper's first certificate (Theorem 3.2).
    """
    Q, R, info, _ = _run(_random_W(), kind=kind)
    c = certificates(info["S"], info["P"], R)
    assert c["Delta"] < 1e-10
    assert c["Delta_tilde"] < 1e-12


def test_Q_is_orthonormal_in_sketch_space_not_in_l2():
    """RBGS orthonormalizes the SKETCH, so Q itself is only close to
    orthonormal: well-conditioned, but Q^T Q is visibly different from I.

    If this fails because Q^T Q = I to machine precision, the code is
    probably using ordinary QR / BCGS and ignoring the sketch.
    """
    Q, R, info, _ = _run(_random_W())          # k = 10m: a rough sketch
    assert cond_Q(Q) < 10                      # well-conditioned
    assert orthogonality_loss(Q) > 1e-3        # but not l2-orthonormal


def test_stored_sketches_match_theta():
    """info["S"] and info["P"] must really be Theta Q and Theta W."""
    W = _random_W()
    Q, R, info, sketch = _run(W)
    assert np.allclose(info["S"], sketch.apply(Q), rtol=1e-10, atol=1e-12)
    assert np.allclose(info["P"], sketch.apply(W), rtol=1e-10, atol=1e-12)


def test_blocks_are_built_left_to_right():
    """Gram-Schmidt property: the first blocks of Q and R depend only on
    the first blocks of W. Catches indexing bugs in the block loop."""
    W = _random_W()
    Q, R, _, _ = _run(W, k=400)
    Q2, R2, _, _ = _run(W[:, :20], k=400)      # same sketch (same seed)
    assert np.allclose(Q[:, :20], Q2, rtol=1e-10, atol=1e-12)
    assert np.allclose(R[:20, :20], R2, rtol=1e-10, atol=1e-12)


def test_single_block():
    """p = 1 (mp = m): there are no earlier blocks to project out."""
    W = _random_W(m=10)
    Q, R, info, _ = _run(W, mp=10)
    assert factorization_error(W, Q, R) < 1e-12
    assert certificates(info["S"], info["P"], R)["Delta"] < 1e-10


def test_reproducible_with_same_seed():
    W = _random_W()
    Q1, R1, _, _ = _run(W, seed=7)
    Q2, R2, _, _ = _run(W, seed=7)
    assert np.array_equal(Q1, Q2) and np.array_equal(R1, R2)


# ----------------------------------------------------- mixed precision
@pytest.mark.parametrize("kind", SKETCH_TYPES)
def test_float32_working_precision(kind):
    """Step 3 runs in the coarse precision: Q comes out in float32 and the
    error is at float32 level, while the sketch stays nearly orthonormal."""
    W = _random_W()
    Q, R, info, _ = _run(W, kind=kind, dtype=np.float32)
    err = factorization_error(W, Q, R)
    assert Q.dtype == np.float32
    assert 1e-10 < err < 1e-5     # float32 really used, and still accurate
    assert certificates(info["S"], info["P"], R)["Delta"] < 1e-3


def test_theorem_3_2_bounds_hold():
    """Theorem 3.2 (a posteriori): if Theta is an eps-embedding (eps <= 1/2)
    and Delta, Delta_tilde <= 0.1, then

      (1+eps)^(-1/2) (1 - Delta - 0.1u) <= sigma(Q) <= (1-eps)^(-1/2) (1 + Delta + 0.1u)
      ||W - QR||_F <= 4 u m^(3/2) ||W||_F

    with u = u_crs (float32 here; float64 is the fine precision)."""
    n, m = 2000, 40
    W = _random_W(n, m)
    Q, R, info, sketch = _run(W, k=1000, dtype=np.float32)
    u_crs, u_fine = UNIT_ROUNDOFF[np.float32], UNIT_ROUNDOFF[np.float64]

    # Hypotheses of the theorem
    assert 100 * m ** 0.5 * n ** 1.5 * u_fine <= u_crs <= 0.01
    eps = max(empirical_epsilon(Q, sketch), empirical_epsilon(W, sketch))
    assert eps <= 0.5
    c = certificates(info["S"], info["P"], R)
    assert c["Delta"] <= 0.1 and c["Delta_tilde"] <= 0.1

    # Conclusions of the theorem
    lower, upper = theorem_3_2_bounds(eps, c["Delta"], u_crs)
    s = singular_values(Q)
    assert lower <= s[-1] and s[0] <= upper
    assert factorization_error(W, Q, R) <= theorem_3_2_factorization_bound(u_crs, m)


# ------------------------------------------ paper experiment, small scale
def test_hard_matrix_rbgs_stable_where_bcgs_fails():
    """Section 5.1 in miniature: on the paper's hard matrix in float32,
    BCGS loses orthogonality (cond(Q) explodes) but RBGS does not."""
    n, m, mp = 2000, 100, 10
    W = generate_hard_matrix(n, m)                   # cond(W) ~ 1e5
    Q, R, info, _ = _run(W, mp=mp, k=4 * m, dtype=np.float32)
    Qb, _, _ = bcgs(W, mp, dtype=np.float32)

    assert cond_Q(Q) < 10
    assert certificates(info["S"], info["P"], R)["Delta"] < 0.1
    assert factorization_error(W, Q, R) < 1e-5
    assert cond_Q(Qb) > 100


# ------------------------------------------------------- invalid input
def test_block_size_must_divide_columns():
    with pytest.raises(ValueError):
        rbgs(_random_W(m=40), 3, Sketch("rademacher", k=400, n=2000, seed=0))


def test_W_must_be_tall():
    with pytest.raises(ValueError):
        rbgs(np.ones((20, 40)), 10, Sketch("rademacher", k=20, n=20, seed=0))


def test_sketch_must_have_at_least_m_rows():
    """S is k x m and must have orthonormal columns, so k >= m is required."""
    with pytest.raises(ValueError):
        rbgs(_random_W(m=40), 10, Sketch("rademacher", k=30, n=2000, seed=0))
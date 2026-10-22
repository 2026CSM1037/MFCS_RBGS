"""
Matrix utilities for reproducing experiments from:
"Randomized block Gram-Schmidt process for the solution of linear systems
and eigenvalue problems" — Balabanov & Grigori.

This module provides:
1. The paper's hard dense matrix (Section 5.1), with the original dimensions
   as defaults.
2. A placeholder loader for the original SuiteSparse matrix Ga41As41H72
   used in the Krylov experiment (Section 5.2).
3. A generated sparse test matrix for development when SuiteSparse downloads
   are unavailable.

IMPORTANT:
- Generating the full paper hard matrix allocates a large dense array. The
  output alone requires about 2.24 GiB in float64, and temporary arrays may
  require additional memory. Use smaller n/m while debugging.
- The generated sparse matrix is for implementation testing only; it is not
  the paper's Ga41As41H72 matrix and must not be reported as an exact
  replication result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.io import mmread


# ---------------------------------------------------------------------------
# Paper parameters
# ---------------------------------------------------------------------------

# Section 5.1: hard matrix W, block size mp, sketch dimension k.
PAPER_HARD_N = 1_000_000
PAPER_HARD_M = 300
PAPER_BLOCK_SIZE = 10
PAPER_SKETCH_K = 3_000

# Section 5.2: Krylov / block-GMRES experiment.
PAPER_KRYLOV_N = 268_096
PAPER_KRYLOV_RHS_BLOCK_SIZE = 100
PAPER_KRYLOV_SHIFT = 0.2
PAPER_GMRES_RESTART = 30
PAPER_KRYLOV_SPACE_DIM = 3_100
PAPER_KRYLOV_MATRIX_NAME = "Ga41As41H72"


def generate_hard_matrix(
    n: int = PAPER_HARD_N,
    m: int = PAPER_HARD_M,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    r"""Generate the hard dense matrix from Section 5.1.

    The paper defines

        W_ij = sin(10 * (mu_j + x_i)) / (cos(100 * (mu_j - x_i)) + 1.1),

    where x and mu are equally spaced in [0, 1]. The paper's defaults are
    n=1,000,000 and m=300.

    For a development smoke test, call generate_hard_matrix(n=2000, m=30).
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer.")
    if not isinstance(m, (int, np.integer)) or m <= 0:
        raise ValueError("m must be a positive integer.")

    estimated_gib = (n * m * np.dtype(dtype).itemsize) / (1024**3)
    if estimated_gib >= 1.0:
        import warnings
        warnings.warn(
            f"This dense matrix requires about {estimated_gib:.2f} GiB for "
            "the output alone; temporary arrays can increase peak memory. "
            "Pass smaller n and m for testing.",
            ResourceWarning,
            stacklevel=2,
        )

    x = np.linspace(0.0, 1.0, n, dtype=dtype)[:, None]
    mu = np.linspace(0.0, 1.0, m, dtype=dtype)[None, :]

    # Compute in row chunks to limit temporary memory. The returned matrix
    # itself still requires n*m*dtype.itemsize bytes.
    W = np.empty((n, m), dtype=dtype)
    rows_per_chunk = max(1, min(n, 100_000))
    for start in range(0, n, rows_per_chunk):
        stop = min(start + rows_per_chunk, n)
        x_chunk = x[start:stop]
        numerator = np.sin(10.0 * (mu + x_chunk))
        denominator = np.cos(100.0 * (mu - x_chunk)) + 1.1
        W[start:stop] = numerator / denominator

    return W


def load_krylov_matrix(
    matrix_path: str | Path,
    expected_n: int | None = PAPER_KRYLOV_N,
    *,
    format: Literal["csr", "csc"] = "csr",
) -> sparse.spmatrix:
    """Load the original SuiteSparse matrix from a Matrix Market .mtx file.

    Download/extract Ga41As41H72 separately, then pass the path to its .mtx
    file. This function intentionally does not silently substitute a
    generated matrix if the file is missing.
    """
    path = Path(matrix_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"SuiteSparse matrix file not found: {path}\n"
            "Download Ga41As41H72 in Matrix Market (.mtx) format and pass "
            "the extracted .mtx path, or use generate_sparse_test_matrix() "
            "for development only."
        )

    matrix = mmread(str(path))
    if not sparse.issparse(matrix):
        matrix = sparse.coo_matrix(np.asarray(matrix))
    matrix = matrix.asformat(format)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {matrix.shape}.")
    if expected_n is not None and matrix.shape != (expected_n, expected_n):
        raise ValueError(
            f"Expected a {expected_n}x{expected_n} matrix for "
            f"{PAPER_KRYLOV_MATRIX_NAME}, got {matrix.shape}. "
            "If intentionally loading a different matrix, pass "
            "expected_n=None."
        )

    return matrix


def generate_sparse_test_matrix(
    n: int = 2_000,
    density: float = 0.005,
    seed: int = 42,
    diagonal_shift: float = 10.0,
    *,
    format: Literal["csr", "csc"] = "csr",
) -> sparse.spmatrix:
    """Generate a reproducible sparse matrix for development/testing only.

    It creates a random sparse matrix, symmetrizes it, and adds a positive
    diagonal shift. This is not the paper's SuiteSparse matrix.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer.")
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in the interval (0, 1].")
    if diagonal_shift <= 0:
        raise ValueError("diagonal_shift must be positive.")

    rng = np.random.default_rng(seed)
    upper = sparse.random(
        n,
        n,
        density=density,
        format="coo",
        random_state=rng,
        data_rvs=lambda size: rng.standard_normal(size),
    )
    # Symmetrize without relying on a particular random sparsity pattern.
    A = (upper.tocsr() + upper.T.tocsr()) * 0.5
    A = A.tolil()
    A.setdiag(np.asarray(A.diagonal()).ravel() + diagonal_shift)
    return A.asformat(format)


def generate_krylov_rhs(
    n: int = PAPER_KRYLOV_N,
    mp: int = PAPER_KRYLOV_RHS_BLOCK_SIZE,
    seed: int = 42,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Generate an n x mp standard-normal block right-hand side.

    The paper specifies the distribution and dimensions, not this exact seed;
    seed=42 is selected here only for reproducibility.
    """
    if n <= 0 or mp <= 0:
        raise ValueError("n and mp must be positive integers.")
    return np.random.default_rng(seed).standard_normal((n, mp)).astype(dtype)


def prepare_krylov_problem(
    matrix_path: str | Path,
    seed: int = 42,
    alpha: float = PAPER_KRYLOV_SHIFT,
) -> tuple[sparse.spmatrix, np.ndarray]:
    """Load original Ga41As41H72, shift it, and generate the RHS.

    Returns (A_Ga + alpha*I, B). ILU(0), reverse Cuthill-McKee ordering,
    and the implicit right-preconditioned operator belong in solver code.
    """
    A = load_krylov_matrix(matrix_path, expected_n=PAPER_KRYLOV_N)
    A_shifted = (A + alpha * sparse.eye(PAPER_KRYLOV_N, format=A.format)).asformat(A.format)
    B = generate_krylov_rhs(
        n=PAPER_KRYLOV_N,
        mp=PAPER_KRYLOV_RHS_BLOCK_SIZE,
        seed=seed,
    )
    return A_shifted, B


def prepare_sparse_test_problem(
    n: int = 2_000,
    mp: int = 10,
    density: float = 0.005,
    seed: int = 42,
    alpha: float = PAPER_KRYLOV_SHIFT,
) -> tuple[sparse.spmatrix, np.ndarray]:
    """Generate a sparse test system (A_test + alpha*I, B_test).

    Use this function while developing/debugging. It does not reproduce the
    original paper's Krylov experiment.
    """
    A_test = generate_sparse_test_matrix(n=n, density=density, seed=seed)
    A_shifted = (A_test + alpha * sparse.eye(n, format=A_test.format)).asformat(A_test.format)
    B_test = generate_krylov_rhs(n=n, mp=mp, seed=seed)
    return A_shifted, B_test

# Backward-compatible alias 
make_rank_deficient_matrix = generate_hard_matrix
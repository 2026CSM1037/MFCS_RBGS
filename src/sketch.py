"""
Sketching operators for the RBGS paper replication.

Supports:
- "srht": implicit Subsampled Randomized Hadamard Transform (recommended for
  paper-scale runs; does not materialize the k-by-n sketch matrix).
- "rademacher": explicit rescaled random-sign sketch (useful as a baseline
  or for small tests).

This module ONLY creates/applies Theta. Mixed-precision GMRES arithmetic and
the five Richardson iterations for the sketched least-squares problems must
be implemented in the RBGS/Krylov solver, not here.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

SketchType = Literal["rademacher", "srht"]

# Paper experiment settings (Balabanov & Grigori, arXiv:2111.14641v2).
PAPER_DEFAULT_SKETCH_TYPE: SketchType = "srht"
PAPER_HARD_N = 1_000_000
PAPER_HARD_SKETCH_K = 3_000
PAPER_KRYLOV_N = 268_096
# The paper's GMRES figure compares these RBGS sketch dimensions.
PAPER_KRYLOV_SKETCH_SIZES = (5_000, 7_500)

# Section 5.2 mixed-precision GMRES settings. These are consumed by solver code.
PAPER_KRYLOV_RHS_BLOCK_SIZE = 100
PAPER_KRYLOV_SHIFT = 0.2
PAPER_GMRES_RESTART = 30
PAPER_KRYLOV_SPACE_DIM = 3_100
PAPER_BGS_DTYPE = np.float32
PAPER_OPERATOR_AND_REDUCED_LS_DTYPE = np.float64
PAPER_SKETCHED_LS_RICHARDSON_STEPS = 5


def _next_power_of_two(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    return 1 << (n - 1).bit_length()


def _fwht(X: NDArray) -> NDArray:
    """Normalized fast Walsh-Hadamard transform along axis 0."""
    Y = np.array(X, dtype=np.result_type(X, np.float64), copy=True)
    if Y.ndim not in (1, 2):
        raise ValueError("X must be a vector or a 2-D matrix")
    N = Y.shape[0]
    if N == 0 or N & (N - 1):
        raise ValueError("The number of rows must be a power of two")

    h = 1
    while h < N:
        for start in range(0, N, 2 * h):
            left = Y[start:start + h].copy()
            right = Y[start + h:start + 2 * h].copy()
            Y[start:start + h] = left + right
            Y[start + h:start + 2 * h] = left - right
        h *= 2
    return Y / np.sqrt(N)


class Sketch:
    """Reproducible sketch Theta: R^n -> R^k.

    Args:
        sketch_type: "srht" (recommended) or "rademacher".
        k: Number of sketch rows.
        n: Number of rows in the input matrix.
        seed: Optional RNG seed for reproducibility.

    Example:
        # Paper hard-matrix experiment
        theta = Sketch("srht", k=PAPER_HARD_SKETCH_K,
                       n=PAPER_HARD_N, seed=42)

        # Krylov experiment: run separately for k=5000 and k=7500
        theta = Sketch("srht", k=5000, n=PAPER_KRYLOV_N, seed=42)
        S = theta.apply(W)
    """

    def __init__(
        self,
        sketch_type: SketchType = PAPER_DEFAULT_SKETCH_TYPE,
        k: int = PAPER_HARD_SKETCH_K,
        n: int = PAPER_HARD_N,
        seed: int | None = None,
    ):
        sketch_type = sketch_type.lower()
        if sketch_type not in ("rademacher", "srht"):
            raise ValueError("sketch_type must be 'rademacher' or 'srht'")
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("n must be a positive integer")
        if not isinstance(k, (int, np.integer)) or not 0 < k <= n:
            raise ValueError("k must be a positive integer no greater than n")

        self.sketch_type = sketch_type
        self.k, self.n, self.seed = int(k), int(n), seed
        rng = np.random.default_rng(seed)
        self._theta = None
        self._N = None
        self._signs = None
        self._indices = None

        if sketch_type == "rademacher":
            signs = rng.choice([-1.0, 1.0], size=(self.k, self.n))
            self._theta = signs / np.sqrt(self.k)
        else:
            self._N = _next_power_of_two(self.n)
            self._signs = rng.choice([-1.0, 1.0], size=self._N)
            self._indices = rng.choice(self._N, size=self.k, replace=False)

    @property
    def Theta(self) -> NDArray:
        """Return explicit Theta for Rademacher; SRHT remains implicit."""
        if self._theta is None:
            raise AttributeError("SRHT is implicit; use sketch.apply(X)")
        return self._theta

    def apply(self, X: NDArray) -> NDArray:
        """Compute Theta @ X, accepting a vector or 2-D matrix."""
        X = np.asarray(X)
        if X.ndim not in (1, 2):
            raise ValueError("X must be a vector or a 2-D matrix")
        if X.shape[0] != self.n:
            raise ValueError(f"Expected {self.n} rows; received {X.shape[0]}")

        if self.sketch_type == "rademacher":
            return self._theta @ X

        N = self._N
        work_dtype = np.result_type(X.dtype, np.float64)
        if X.ndim == 1:
            padded = np.zeros(N, dtype=work_dtype)
            padded[:self.n] = X
            padded *= self._signs
        else:
            padded = np.zeros((N, X.shape[1]), dtype=work_dtype)
            padded[:self.n, :] = X
            padded *= self._signs[:, None]

        transformed = _fwht(padded)
        # _fwht is normalized by 1/sqrt(N); this produces the conventional
        # SRHT scaling 1/sqrt(k).
        return np.sqrt(N / self.k) * transformed[self._indices]

    def __repr__(self) -> str:
        return (
            f"Sketch(sketch_type={self.sketch_type!r}, k={self.k}, "
            f"n={self.n}, seed={self.seed})"
        )


def create_sketch(
    kind: SketchType = PAPER_DEFAULT_SKETCH_TYPE,
    n: int | None = None,
    k: int | None = None,
    seed: int | None = None,
) -> Sketch:
    """Factory interface; defaults to SRHT but requires n and k."""
    if n is None or k is None:
        raise ValueError("Both n and k must be supplied")
    return Sketch(kind, k=k, n=n, seed=seed)


def create_rademacher_sketch(
    n: int, k: int, seed: int | None = None
) -> Sketch:
    """Create an explicit Rademacher sketch (best for small baseline tests)."""
    return Sketch("rademacher", k=k, n=n, seed=seed)


def create_paper_hard_sketch(seed: int | None = None) -> Sketch:
    """Create the SRHT sketch configuration for Section 5.1."""
    return Sketch(
        "srht", k=PAPER_HARD_SKETCH_K, n=PAPER_HARD_N, seed=seed
    )


def create_paper_krylov_sketch(
    k: int = 5_000, seed: int | None = None
) -> Sketch:
    """Create a Krylov SRHT sketch; paper figure compares k=5000 and 7500."""
    if k not in PAPER_KRYLOV_SKETCH_SIZES:
        raise ValueError(
            f"For the paper Krylov comparison, use k in "
            f"{PAPER_KRYLOV_SKETCH_SIZES}; got {k}."
        )
    return Sketch("srht", k=k, n=PAPER_KRYLOV_N, seed=seed)


def apply_sketch(sketch: Sketch | NDArray, X: NDArray) -> NDArray:
    """Apply a Sketch object or an explicit Theta matrix."""
    if isinstance(sketch, Sketch):
        return sketch.apply(X)
    theta, X = np.asarray(sketch), np.asarray(X)
    if theta.ndim != 2 or X.ndim not in (1, 2):
        raise ValueError("Theta must be 2-D and X must be a vector or matrix")
    if theta.shape[1] != X.shape[0]:
        raise ValueError("Theta column count must match X row count")
    return theta @ X


if __name__ == "__main__":
    # Small safe smoke tests; do not allocate paper-scale arrays here.
    W = np.arange(64, dtype=np.float64).reshape(16, 4)
    srht = Sketch("srht", k=8, n=16, seed=42)
    rad = Sketch("rademacher", k=8, n=16, seed=42)
    assert srht.apply(W).shape == (8, 4)
    assert rad.apply(W).shape == (8, 4)
    print("Sketch smoke tests passed.")
    print("Paper hard sketch:", PAPER_DEFAULT_SKETCH_TYPE,
          f"n={PAPER_HARD_N}, k={PAPER_HARD_SKETCH_K}")
    print("Paper Krylov sketch sizes:", PAPER_KRYLOV_SKETCH_SIZES)
    print("Paper Krylov precision settings are constants for the solver; "
          "they are not applied by Sketch.apply().")

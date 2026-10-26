"""Experiment 3: eigenvalues with the restarted Rayleigh-Ritz method (paper Section 5.3, Figure 3).

PURPOSE
    Compute the dominant eigenpairs of a symmetric matrix with restarted
    Rayleigh-Ritz (Algorithm 4.2) built on different Gram-Schmidt processes,
    and compare with plain subspace (power) iteration.

SETUP
    Paper (pass --matrix path/to/Ga41As41H72.mtx): the smallest eigenpairs of
    A_Ga are the dominant ones of A = 1500 I - A_Ga (n = 268,096). B = n x 50
    Gaussian, Arnoldi restarted every 50 iterations (basis of 2550 columns),
    200 iterations, everything in float64, RBGS with SRHT k = 5000 and 7500
    and 5 Richardson iterations in step 2.
    Without --matrix a development problem with the same structure is used
    (qualitative behaviour only): A = 8 I - L with L the 2-D Laplacian on a
    70 x 70 grid (n = 4900), whose dominant eigenvalues are tightly clustered;
    B = n x 10, restart every 50 iterations (basis of 510 columns), and
    k = 1000 and 1500, about the paper's k / basis-size ratios.
    Methods: subspace iteration, BCGS, BCGS restarted every 5 iterations,
    BCGS2, BMGS, and RBGS with two sketch sizes.

MEASURED (per iteration)
    Error = max_j ||A u_j - mu_j u_j|| / ||mu_j u_j|| over the first 80% of the
    Ritz pairs (Figure 3a), and cond of the Krylov basis (Figure 3b).

PREDICTION (from the paper)
    Subspace iteration stagnates (the eigenvalues are clustered). BCGS-Arnoldi
    becomes unstable early (cond(Q) close to 1/u); restarting it every 5
    iterations keeps it stable but converges much more slowly. BMGS produces an
    ill-conditioned basis yet still converges, at a somewhat slower rate.
    BCGS2 and RBGS stay well-conditioned and converge to machine precision
    within 200 iterations.

NOTE
    Unlike GMRES, the accuracy of the sketched Rayleigh-Ritz projection is not
    guaranteed by the embedding property alone (paper, Section 4.4). With
    k = 1000 on the development problem, seed 0 loses accuracy after the third
    restart (cond(Q) stays small and Delta < 0.1), while seed 1 converges to
    1e-14; k = 1500 converges for both. Check several seeds before drawing
    conclusions about the smaller sketch.

RUN
    python -m experiments.exp3_rr
    python -m experiments.exp3_rr --matrix data/Ga41As41H72.mtx
    add --no-cond to skip cond(Q) (one SVD per iteration; slow on the paper matrix)

OUTPUT
    results/exp3_rr.csv and results/exp3_rr.png
"""
import argparse

import numpy as np
from scipy import sparse

from experiments.exp_common import print_table, save_and_plot
from src.krylov import rr, subspace_iteration
from src.matrices import generate_krylov_rhs, load_krylov_matrix
from src.sketch import PAPER_KRYLOV_SKETCH_SIZES, create_sketch

N_ITER = 200          # Arnoldi iterations in total
RESTART = 50          # iterations per Arnoldi cycle, so p = RESTART + 1 blocks
BCGS_RESTART = 5      # the restarted BCGS variant
FRAC = 0.8            # error over the first 80% of the Ritz pairs
SEED = 0

PAPER_ALPHA, PAPER_MP = 1500.0, 50
DEV_GRID, DEV_ALPHA, DEV_MP = 50, 8.0, 10
DEV_SKETCH_SIZES = (1000, 1500)


def laplacian_2d(g):
    Id = sparse.identity(g)
    T = sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(g, g))
    return (sparse.kron(Id, T) + sparse.kron(T, Id)).tocsr()


def load_problem(matrix_path):
    """Return (A, B, sketch sizes, description)."""
    if matrix_path:
        A_ga = load_krylov_matrix(matrix_path)
        n = A_ga.shape[0]
        A = (PAPER_ALPHA * sparse.identity(n, format="csr") - A_ga).tocsr()
        B = generate_krylov_rhs(n, PAPER_MP, seed=SEED)
        return A, B, PAPER_KRYLOV_SKETCH_SIZES, f"{PAPER_ALPHA:g} I - Ga41As41H72"
    L = laplacian_2d(DEV_GRID)
    n = L.shape[0]
    A = (DEV_ALPHA * sparse.identity(n, format="csr") - L).tocsr()
    B = np.random.default_rng(SEED).standard_normal((n, DEV_MP))
    desc = f"{DEV_ALPHA:g} I - 2-D Laplacian {DEV_GRID}x{DEV_GRID} (development)"
    return A, B, DEV_SKETCH_SIZES, desc


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", help="path to Ga41As41H72.mtx (paper setting)")
    parser.add_argument("--no-cond", action="store_true", help="do not track cond(Q)")
    args = parser.parse_args()

    A, B, sketch_sizes, desc = load_problem(args.matrix)
    n, mp = B.shape
    print(f"Experiment 3: {desc}, n={n}, mp={mp}, restart={RESTART}, {N_ITER} iterations")

    curves = {}
    print("  running subspace iteration ...")
    _, _, info = subspace_iteration(A, B, N_ITER, frac=FRAC)
    curves["power iteration"] = {"error": info["errors"]}

    runs = [("BCGS", "bcgs", None, RESTART),
            (f"BCGS-restarted ({BCGS_RESTART})", "bcgs", None, BCGS_RESTART),
            ("BCGS2", "bcgs2", None, RESTART),
            ("BMGS", "bmgs", None, RESTART)]
    runs += [(f"RBGS, k={k}", "rbgs", k, RESTART) for k in sketch_sizes]
    for label, method, k, restart in runs:
        print(f"  running {label} ...")
        sketch = create_sketch("srht", n=n, k=k, seed=SEED) if k else None
        _, _, info = rr(A, B, p=restart + 1, n_cycles=N_ITER // restart, method=method,
                        sketch=sketch, frac=FRAC, track_cond=not args.no_cond)
        curves[label] = {"error": info["errors"], "cond_Q": info["cond_Q"]}

    save_and_plot("exp3_rr", f"Experiment 3: restarted Rayleigh-Ritz, {desc}", curves,
                  [("error", "max_j ||A u_j - mu_j u_j|| / ||mu_j u_j||"), ("cond_Q", "cond(Q)")])
    print_table(curves, [("error", "final error")])
    for label, data in curves.items():
        if data.get("cond_Q") is not None:
            print(f"{label:<26} max cond(Q) = {np.max(data['cond_Q']):.2e}")


if __name__ == "__main__":
    main()
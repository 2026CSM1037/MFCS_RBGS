"""Experiment 2: block GMRES with different Gram-Schmidt processes (paper Section 5.2, Figure 2).

PURPOSE
    Solve A X = B with restarted block GMRES and compare how the Arnoldi
    basis built by BCGS, BMGS, BCGS2 and RBGS affects convergence and the
    conditioning of the Krylov basis.

SETUP
    Paper (pass --matrix path/to/Ga41As41H72.mtx, from the SuiteSparse collection):
        A = A_Ga + 0.2 I (n = 268,096), B = n x 100 Gaussian, right-preconditioned
        by ILU after reverse Cuthill-McKee ordering, restart every 30 iterations
        (Krylov basis of 3100 columns), 120 iterations, RBGS with SRHT k = 5000
        and 7500 and 5 Richardson iterations in step 2.
    Without --matrix a small development problem is used instead (it does NOT
    reproduce the paper's numbers, only the qualitative behaviour):
        2-D convection-diffusion on a 60 x 60 grid (n = 3600), B = n x 10,
        ILU with drop_tol = 0.2, restart 30 (basis of 310 columns), and
        k = 500 and 750, i.e. the same k / basis-size ratios as the paper.
    In both cases the Gram-Schmidt process runs in float32, while products
    with A and the small least-squares problems use float64, as in the paper.
    The paper uses ILU(0), which SciPy lacks; threshold ILU is used instead.

MEASURED (per iteration)
    max_j ||A x_j - b_j|| / ||b_j|| (Figure 2a) and cond of the Krylov basis (Figure 2b).

PREDICTION (from the paper)
    BCGS loses orthogonality right away and its residual stagnates; BMGS is
    partially unstable and converges more slowly; BCGS2 and both RBGS runs stay
    well-conditioned (RBGS: cond(Q) of order 1-10, growing within each restart
    cycle) and converge at the same, optimal rate.

RUN
    python -m experiments.exp2_gmres
    python -m experiments.exp2_gmres --matrix data/Ga41As41H72.mtx
    add --no-cond to skip cond(Q) (one SVD per iteration; slow on the paper matrix)

OUTPUT
    results/exp2_gmres.csv and results/exp2_gmres.png
"""
import argparse

import numpy as np
from scipy import sparse

from experiments.exp_common import print_table, save_and_plot
from src.krylov import gmres, ilu_preconditioner
from src.matrices import PAPER_GMRES_RESTART, prepare_krylov_problem
from src.sketch import PAPER_KRYLOV_SKETCH_SIZES, create_sketch

MAX_ITER = 120
TOL = 1e-12
SEED = 0

DEV_GRID, DEV_MP, DEV_DROP_TOL = 60, 10, 0.2
DEV_SKETCH_SIZES = (500, 750)
PAPER_DROP_TOL = 1e-3          # tune if the ILU needs too much memory or time


def convection_diffusion(g, convection=0.3):
    """Non-symmetric 2-D convection-diffusion matrix on a g x g grid (n = g*g)."""
    Id = sparse.identity(g)
    T = sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(g, g))
    C = sparse.diags([-1.0, 1.0], [-1, 1], shape=(g, g))
    return (sparse.kron(Id, T) + sparse.kron(T, Id) + convection * sparse.kron(Id, C)).tocsr()


def load_problem(matrix_path):
    """Return (A, B, preconditioner, sketch sizes, description)."""
    if matrix_path:
        A, B = prepare_krylov_problem(matrix_path)
        print("Building the ILU preconditioner (can take a while) ...")
        return (A, B, ilu_preconditioner(A, drop_tol=PAPER_DROP_TOL),
                PAPER_KRYLOV_SKETCH_SIZES, "Ga41As41H72 + 0.2 I")
    A = convection_diffusion(DEV_GRID)
    B = np.random.default_rng(SEED).standard_normal((A.shape[0], DEV_MP))
    return (A, B, ilu_preconditioner(A, drop_tol=DEV_DROP_TOL),
            DEV_SKETCH_SIZES, f"convection-diffusion {DEV_GRID}x{DEV_GRID} (development)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", help="path to Ga41As41H72.mtx (paper setting)")
    parser.add_argument("--no-cond", action="store_true", help="do not track cond(Q)")
    args = parser.parse_args()

    A, B, M, sketch_sizes, desc = load_problem(args.matrix)
    n, mp = B.shape
    print(f"Experiment 2: {desc}, n={n}, mp={mp}, restart={PAPER_GMRES_RESTART}, "
          f"basis={(PAPER_GMRES_RESTART + 1) * mp} columns")

    runs = [("BCGS", "bcgs", None), ("BCGS2", "bcgs2", None), ("BMGS", "bmgs", None)]
    runs += [(f"RBGS, k={k}", "rbgs", k) for k in sketch_sizes]

    curves = {}
    for label, method, k in runs:
        print(f"  running {label} ...")
        sketch = create_sketch("srht", n=n, k=k, seed=SEED) if k else None
        _, info = gmres(A, B, method=method, sketch=sketch, restart=PAPER_GMRES_RESTART,
                        max_iter=MAX_ITER, tol=TOL, M=M, dtype=np.float32,
                        track_cond=not args.no_cond)
        curves[label] = {"residual": info["residuals"], "cond_Q": info["cond_Q"]}

    save_and_plot("exp2_gmres", f"Experiment 2: block GMRES, {desc}", curves,
                  [("residual", "max_j ||A x_j - b_j|| / ||b_j||"), ("cond_Q", "cond(Q)")])
    print_table(curves, [("residual", "final residual")])
    for label, data in curves.items():
        if data["cond_Q"] is not None:
            print(f"{label:<26} max cond(Q) = {np.max(data['cond_Q']):.2e}, "
                  f"iterations = {len(data['residual'])}")


if __name__ == "__main__":
    main()
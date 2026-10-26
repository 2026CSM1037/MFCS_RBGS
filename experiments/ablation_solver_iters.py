"""Ablation: accuracy of the sketched least-squares solve (step 2 of RBGS).

PURPOSE
    Step 2 solves  min_Y || S_{1:i-1} Y - P_i ||_F  on the small sketches. A
    direct solver is exact but cubic; the paper replaces it with a few
    Richardson iterations (Section 2.1 and the supplement, Section 5.2 uses 5)
    or CG (Section 5.1 uses 20). How many iterations are really needed?

VARIED (one thing)
    The number of Richardson iterations ls_iters in ITERS_VALUES
    (ls_solver="richardson"). Two reference points are added for comparison:
        "householder (exact)"  direct solve of the least-squares problem.
        "cg l=20"              CG with the paper's iteration count (optional,
                               see INCLUDE_CG).

FIXED
    Matrix (paper's rank-deficient matrix), block size, sketch size and kind,
    qr_method, precisions, seeds: all from experiments/config.py.

MEASURED
    cond(Q), ||W - QR||/||W||, Delta (and Delta_tilde in the CSV), run time.

PREDICTION (written before running; edit if you change it)
    One Richardson iteration is BCGS in the sketched inner product, and each
    extra iteration acts like one re-orthogonalization; in exact arithmetic the
    error of the solve shrinks like Delta^l (supplement, eq. SM.2). So Delta
    and cond(Q) should drop quickly over the first few iterations and then
    level off near the exact Householder result, which is why a small fixed
    number (the paper's 5) is enough when S is nearly orthonormal. The
    iteration needs ||I - S^T S|| < 1: on this numerically rank-deficient
    matrix Richardson may stagnate or diverge for the later blocks, in which
    case the exact solver and CG should be clearly better. Run time should
    grow slowly with l, since each iteration costs only O(mp * m * k).

RUN
    python -m experiments.ablation_solver_iters
"""
from experiments.ablation_common import run_ablation

ITERS_VALUES = [1, 2, 3, 5, 10, 20]   # the paper uses 5
INCLUDE_CG = True                      # also run CG with 20 iterations


def main():
    settings = [{"label": "householder (exact)",
                 "rbgs_kwargs": {"ls_solver": "householder"}}]
    settings += [{"label": f"richardson l={l}",
                  "rbgs_kwargs": {"ls_solver": "richardson", "ls_iters": l}}
                 for l in ITERS_VALUES]
    if INCLUDE_CG:
        settings.append({"label": "cg l=20",
                         "rbgs_kwargs": {"ls_solver": "cg", "ls_iters": 20}})
    run_ablation("ablation_solver_iters",
                 "Ablation: sketched least-squares solver", settings,
                 connect=False)


if __name__ == "__main__":
    main()
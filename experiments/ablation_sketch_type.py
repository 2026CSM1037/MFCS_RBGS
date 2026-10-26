"""Ablation: type of random sketching matrix Theta.

PURPOSE
    Does the choice of sketching distribution change the stability or the cost
    of RBGS? The paper uses rescaled Rademacher matrices and the Subsampled
    Randomized Hadamard Transform (SRHT) and reports that they give very
    similar results (Section 1.3 and the start of Section 5).

VARIED (one thing)
    kind (names as defined in src/sketch.py):
        "srht"        implicit SRHT: random signs, fast Walsh-Hadamard
                      transform, uniform row subsampling (the default).
        "rademacher"  explicit k x n matrix of random +-1/sqrt(k) entries.

FIXED
    Matrix (paper's rank-deficient matrix), block size, sketch size k,
    qr_method, least-squares solver, precisions, seeds: all from
    experiments/config.py.

MEASURED
    cond(Q), ||W - QR||/||W||, Delta (and Delta_tilde in the CSV), run time.

PREDICTION (written before running; edit if you change it)
    Both distributions are (eps, delta, d) oblivious subspace embeddings with
    k = O(m) rows, so cond(Q), Delta and the factorization error should be
    almost identical for the same k. Differences should show up in cost, not
    in accuracy: the Rademacher sketch stores a dense k x n matrix and applies
    it with a matrix-matrix product, while the SRHT never forms Theta and costs
    about n log n flops per column. At the small n of config.py the dense
    product may still be as fast as (or faster than) the Python Walsh-Hadamard
    loop, so the time ranking can differ from the paper's large-n setting.

RUN
    python -m experiments.ablation_sketch_type
"""
from experiments.ablation_common import run_ablation

# Names src.sketch does not accept are skipped with a printed message.
SKETCH_KINDS = ["srht", "rademacher"]


def main():
    settings = [{"label": kind, "kind": kind} for kind in SKETCH_KINDS]
    run_ablation("ablation_sketch_type", "Ablation: sketch type",
                 settings, connect=False)


if __name__ == "__main__":
    main()
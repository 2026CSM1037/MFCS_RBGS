"""Experiment 1: block QR of a numerically rank-deficient matrix (paper Section 5.1, Figure 1).

PURPOSE
    Compare the stability of BCGS, BMGS, BCGS2 and RBGS when the leading
    columns W_(1:i) become numerically rank-deficient as i grows.

SETUP
    W = generate_hard_matrix(n, m): W_ij = sin(10(mu_j + x_i)) / (cos(100(mu_j - x_i)) + 1.1),
    m = 300 columns in blocks of mp = 10, SRHT sketch with k = 3000 rows.
    The paper uses n = 10^6; the default here is n = 20,000 so it runs in about a
    minute (pass --paper for n = 10^6: needs several GB of memory and is slow).
    Methods (as in the paper):
        BCGS, BMGS, BCGS2          float32, inter-block QR in float64
        u.p. CG-RBGS               unique float32, step 2 by 20 CG iterations
        u.p. HH-RBGS               unique float32, step 2 by Householder
        m.p. CG-RBGS               step 3 in float32, rest in float64, 20 CG iterations

MEASURED (per iteration i, i.e. after i blocks)
    cond(Q_(1:i)) (Figure 1a), with sqrt(cond(W_(1:i))) as reference,
    ||W_(1:i) - Q_(1:i) R_(1:i,1:i)|| / ||W_(1:i)|| (Figure 1b),
    and Delta = ||I - S^T S||_F for the RBGS variants (Theorem 3.2 certificate).

PREDICTION (from the paper)
    BCGS becomes unstable first (around i = 8), BCGS2 later (around i = 17),
    BMGS loses orthogonality gradually. Unique-precision RBGS is about as stable
    as BMGS with Householder and less stable with 20 CG iterations.
    Multi-precision RBGS stays stable with cond(Q) = O(1) at every iteration.
    The error stays near float32 roundoff for all methods except BCGS2.

RUN
    python -m experiments.exp1_qr_stability            (n = 20,000)
    python -m experiments.exp1_qr_stability --paper    (n = 10^6)
    python -m experiments.exp1_qr_stability --n 100000

OUTPUT
    results/exp1_qr_stability.csv and results/exp1_qr_stability.png
"""
import argparse

import numpy as np

from experiments.exp_common import print_table, save_and_plot
from src.baselines import bcgs, bcgs2, bmgs
from src.matrices import PAPER_HARD_N, generate_hard_matrix
from src.metrics import certificates
from src.rbgs import rbgs
from src.sketch import create_sketch

N_DEFAULT = 20_000
M, MP, K = 300, 10, 3000
SEED = 0

F32, F64 = np.float32, np.float64
RBGS_VARIANTS = {
    "u.p. CG-RBGS": dict(dtype=F32, fine_dtype=F32, ls_solver="cg", ls_iters=20),
    "u.p. HH-RBGS": dict(dtype=F32, fine_dtype=F32, ls_solver="householder"),
    "m.p. CG-RBGS": dict(dtype=F32, fine_dtype=F64, ls_solver="cg", ls_iters=20),
}


def prefix_conds(X, mp):
    """cond(X[:, :i*mp]) for every i, from one Householder QR (the R factor of a
    prefix is the leading block of R), much faster than one SVD per prefix."""
    R = np.linalg.qr(np.asarray(X, dtype=F64), mode="r")
    return np.array([np.linalg.cond(R[:c, :c]) for c in range(mp, X.shape[1] + 1, mp)])


def prefix_errors(W, Q, R, mp):
    """||W_(1:i) - Q_(1:i) R_(1:i,1:i)||_F / ||W_(1:i)||_F for every i.
    R is upper triangular, so these are the leading columns of W - QR."""
    E = np.asarray(W, dtype=F64) - np.asarray(Q, dtype=F64) @ R
    err2 = np.cumsum(np.sum(E * E, axis=0))
    w2 = np.cumsum(np.sum(np.asarray(W, dtype=F64) ** 2, axis=0))
    return np.sqrt(err2 / w2)[mp - 1::mp]


def prefix_deltas(S, P, R, mp):
    return np.array([certificates(S[:, :c], P[:, :c], R[:c, :c])["Delta"]
                     for c in range(mp, S.shape[1] + 1, mp)])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paper", action="store_true", help="use the paper's n = 10^6")
    parser.add_argument("--n", type=int, default=None, help="number of rows of W")
    args = parser.parse_args()
    n = args.n or (PAPER_HARD_N if args.paper else N_DEFAULT)

    print(f"Experiment 1: n={n}, m={M}, mp={MP}, k={K} (SRHT)")
    W = generate_hard_matrix(n, M)
    curves = {"sqrt(cond(W))": {"cond_Q": np.sqrt(prefix_conds(W, MP))}}

    for name, method in (("BCGS", bcgs), ("BMGS", bmgs), ("BCGS2", bcgs2)):
        print(f"  running {name} ...")
        Q, R, _ = method(W, MP, dtype=F32)
        curves[name] = {"cond_Q": prefix_conds(Q, MP), "error": prefix_errors(W, Q, R, MP)}

    sketch = create_sketch("srht", n=n, k=K, seed=SEED)
    for name, kwargs in RBGS_VARIANTS.items():
        print(f"  running {name} ...")
        Q, R, info = rbgs(W, MP, sketch, **kwargs)
        curves[name] = {"cond_Q": prefix_conds(Q, MP), "error": prefix_errors(W, Q, R, MP),
                        "Delta": prefix_deltas(info["S"], info["P"], R, MP)}

    save_and_plot("exp1_qr_stability",
                  f"Experiment 1: block QR of the hard matrix (n={n}, m={M}, mp={MP})",
                  curves, [("cond_Q", "cond(Q)"), ("error", "||W - QR|| / ||W||")])
    print_table(curves, [("cond_Q", "final cond(Q)"), ("error", "final error"),
                         ("Delta", "final Delta")])


if __name__ == "__main__":
    main()
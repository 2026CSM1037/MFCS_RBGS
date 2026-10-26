"""Ablation: inter-block QR method (steps 4-5 of RBGS).

PURPOSE
    Does the way the cleaned block Q'_i is orthonormalized change the stability
    or the cost of RBGS?  (Paper, Section 2.2.)

VARIED (one thing)
    qr_method (names as defined in src/rbgs.py):
        "l2_rcholesky"        Householder QR of the big block first, then
                              RCholeskyQR (Algorithm 2.5; the default).
        "rcholesky"           RCholeskyQR (Algorithm 2.3): QR of the small
                              sketch only, then a triangular solve.
        "implicit_rcholesky"  Algorithm 2.4: the sketch of the block is
                              obtained as P_i - S R instead of a new product.
        "rgs"                 single-vector RGS, column by column (Sec 2.2.1).

FIXED
    Matrix (paper's rank-deficient matrix), block size, sketch size and kind,
    least-squares solver, precisions, seeds: all from experiments/config.py.

MEASURED
    cond(Q), ||W - QR||/||W||, Delta (and Delta_tilde in the CSV), run time.

PREDICTION (written before running; edit if you change it)
    The l2_rcholesky variant (Algorithm 2.5) adds an ordinary QR of the
    n x mp block, so it should cost more time but be at least as stable as
    plain RCholeskyQR, especially on the later, nearly dependent blocks. All
    variants should keep cond(Q) small; single-vector rgs needs the most
    passes over the data and should be the slowest.

RUN
    python -m experiments.ablation_qr_method
"""
from experiments.ablation_common import run_ablation

# Names rbgs does not accept are skipped with a printed message.
QR_METHODS = ["l2_rcholesky", "rcholesky", "implicit_rcholesky", "rgs"]


def main():
    settings = [{"label": name, "rbgs_kwargs": {"qr_method": name}}
                for name in QR_METHODS]
    run_ablation("ablation_qr_method",
                 "Ablation: inter-block QR method", settings, connect=False)


if __name__ == "__main__":
    main()
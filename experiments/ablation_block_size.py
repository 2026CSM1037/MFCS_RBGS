"""Ablation: block size m_p.

PURPOSE
    How does the block size (columns processed together) affect stability and
    speed? Smaller blocks mean more steps; larger blocks mean bigger
    matrix-matrix operations and a bigger small QR in each step.

VARIED (one thing)
    mp in MP_VALUES (only values that divide the number of columns m are used).

FIXED
    Matrix, sketch size k and kind, qr_method, least-squares solver,
    precisions, seeds: all from experiments/config.py.

MEASURED
    cond(Q), ||W - QR||/||W||, Delta, run time.

PREDICTION (written before running; edit if you change it)
    The bounds in Theorem 3.2 depend on the total number of columns m, not on
    the block size, so stability should be roughly insensitive to m_p. Run
    time should drop as m_p grows (fewer loop iterations, larger BLAS3
    operations) until the bigger per-block work starts to dominate.

RUN
    python -m experiments.ablation_block_size
"""
from experiments.ablation_common import M, run_ablation

MP_VALUES = [2, 5, 10, 20, 50]


def main():
    values = [mp for mp in MP_VALUES if mp <= M and M % mp == 0]
    if not values:
        raise SystemExit(f"None of {MP_VALUES} divides M={M}; edit MP_VALUES.")
    settings = [{"label": f"mp={mp}", "mp": mp} for mp in values]
    run_ablation("ablation_block_size", "Ablation: block size", settings)


if __name__ == "__main__":
    main()
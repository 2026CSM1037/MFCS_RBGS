"""Ablation: working precision of the expensive operation (step 3).

PURPOSE
    RBGS uses two precisions: a coarse one for the big high-dimensional
    subtraction and a fine one for everything involving the sketch. How much
    does the coarse precision matter? (Paper, Sections 2 and 3, Theorem 3.2.)

VARIED (one thing)
    The working precisions (rbgs arguments dtype and fine_dtype):
        all float64                    reference
        multi-precision                dtype=float32 (step 3), fine_dtype=float64
        unique float32                 dtype=float32, fine_dtype=float32
    The last two are the paper's "multi-precision" and "unique precision" RBGS.

FIXED
    Matrix, block size, sketch size and kind, qr_method, least-squares solver,
    seeds: all from experiments/config.py.

MEASURED
    cond(Q), ||W - QR||/||W||, Delta, run time.

PREDICTION (written before running; edit if you change it)
    Theorem 3.2 bounds the factorization error by about 4*u_crs*m^(3/2)*||W||,
    so the error should scale with the coarse unit roundoff (around 1e-7 for
    float32, around 1e-16 for float64). cond(Q) should stay near 1 in the
    mixed-precision case as long as the certificates Delta and Delta_tilde
    stay below 0.1; the unique-precision variant may be less stable.

RUN
    python -m experiments.ablation_precision
"""
import numpy as np

from experiments.ablation_common import run_ablation


def main():
    f32, f64 = np.float32, np.float64
    settings = [
        {"label": "all float64",
         "rbgs_kwargs": {"dtype": f64, "fine_dtype": f64}},
        {"label": "multi-precision (f32 / f64)",
         "rbgs_kwargs": {"dtype": f32, "fine_dtype": f64}},
        {"label": "unique float32",
         "rbgs_kwargs": {"dtype": f32, "fine_dtype": f32}},
    ]
    run_ablation("ablation_precision",
                 "Ablation: working precision", settings, connect=False)


if __name__ == "__main__":
    main()
"""Ablation: sketch size k (number of rows of Theta).

PURPOSE
    How many rows must the sketching matrix have? A larger k makes Theta a
    better epsilon-embedding of range(W) and range(Q) (smaller epsilon), but
    the sketched products and the small least-squares / QR problems cost more.
    (Paper, Section 1.3 and Section 3.3; Section 5 compares k=5000 vs k=7500.)

VARIED (one thing)
    k in K_VALUES (only values with m <= k <= n are used; RBGS needs k >= m).

FIXED
    Matrix (paper's rank-deficient matrix), block size, sketch kind,
    qr_method, least-squares solver, precisions, seeds: all from
    experiments/config.py.

MEASURED
    cond(Q), ||W - QR||/||W||, Delta (and Delta_tilde in the CSV), run time.

PREDICTION (written before running; edit if you change it)
    The embedding quality improves roughly like epsilon ~ sqrt(m / k), and
    cond(Q) <= sqrt((1 + eps) / (1 - eps)) (Corollary 1.4), so cond(Q) should
    shrink toward 1 as k grows. When k is close to m the embedding is poor:
    cond(Q) and Delta should be noticeably larger and vary more between random
    seeds. The factorization error ||W - QR|| is governed by the working
    precision (Theorem 3.2), so it should be nearly flat in k. Run time should
    grow slowly with k, since only the small sketch-sized operations scale
    with it.

RUN
    python -m experiments.ablation_sketch_size
"""
from experiments.ablation_common import M, N, run_ablation

# Multiples of m, from barely-enough rows up to a large oversampling factor.
K_VALUES = [60, 90, 120, 180, 300, 600, 1200, 2400]


def main():
    values = [k for k in K_VALUES if M <= k <= N]
    if not values:
        raise SystemExit(f"None of {K_VALUES} lies in [M={M}, N={N}]; "
                         "edit K_VALUES.")
    settings = [{"label": f"k={k}", "k": k} for k in values]
    run_ablation("ablation_sketch_size", "Ablation: sketch size k", settings)


if __name__ == "__main__":
    main()
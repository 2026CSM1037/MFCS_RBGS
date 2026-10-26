"""Shared helpers for the ablation scripts (qr_method, block_size, precision, ...).

An ablation changes ONE thing and keeps everything else at the defaults from
experiments/config.py. Each script only has to describe its list of `settings`:

    {"label": "text shown in plots",
     "mp": block size (optional), "k": sketch size (optional),
     "kind": sketch kind (optional),
     "rbgs_kwargs": {...extra keyword arguments for rbgs...}}

and call `run_ablation(...)`, which runs every setting for N_SEEDS random
sketches, writes results/<name>_raw.csv, results/<name>_summary.csv and
results/<name>.png, and prints a table.

Uses the project's real interfaces:
    src.matrices.generate_hard_matrix(n, m)
    src.sketch.create_sketch(kind, n=..., k=..., seed=...)
    src.rbgs.rbgs(W, mp, sketch, dtype=, fine_dtype=, qr_method=, ls_solver=)
        -> (Q, R, info) with info["Delta"] and info["Delta_tilde"].
Precision keywords: dtype = coarse precision of step 3, fine_dtype = precision
of the sketch-sized steps.
"""
import csv
import inspect
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")          # no window needed; figures are saved to files
import matplotlib.pyplot as plt

from experiments import config
from src.matrices import generate_hard_matrix
from src.metrics import cond_Q, factorization_error, orthogonality_loss, timed
from src.rbgs import rbgs
from src.sketch import create_sketch

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# Defaults come from config.py (fallbacks only matter if a name is missing there).
SEED = getattr(config, "SEED", 0)
N_SEEDS = getattr(config, "N_SEEDS", 5)
N = getattr(config, "N", 10000)
M = getattr(config, "M", 100)
MP = getattr(config, "MP", 10)
K = getattr(config, "K", 10 * M)
SKETCH_KIND = getattr(config, "SKETCH_KIND", "srht")
LS_SOLVER = getattr(config, "LS_SOLVER", "householder")
QR_METHOD = getattr(config, "QR_METHOD", "l2_rcholesky")

METRICS = ["cond_Q", "orth_loss", "fact_error", "Delta", "Delta_tilde", "time_s"]
# (column, y-axis label, log scale?)
PLOTS = [
    ("cond_Q", "cond(Q)", True),
    ("fact_error", "||W - QR|| / ||W||", True),
    ("Delta", "Delta = ||I - S^T S||_F", True),
    ("time_s", "time (s)", False),
]


def rbgs_accepts(argument):
    """True if rbgs has a keyword argument with this name."""
    return argument in inspect.signature(rbgs).parameters


def run_once(W, setting, seed):
    """Run RBGS once for one setting and one seed; return a dict of metrics."""
    mp = setting.get("mp", MP)
    k = setting.get("k", K)
    kind = setting.get("kind", SKETCH_KIND)
    kwargs = {"ls_solver": LS_SOLVER, "qr_method": QR_METHOD}
    kwargs.update(setting.get("rbgs_kwargs", {}))

    sketch = create_sketch(kind, n=W.shape[0], k=k, seed=seed)
    (Q, R, info), elapsed = timed(rbgs, W, mp, sketch, **kwargs)
    return {
        "cond_Q": cond_Q(Q),
        "orth_loss": orthogonality_loss(Q),
        "fact_error": factorization_error(W, Q, R),
        "Delta": float(info.get("Delta", np.nan)),
        "Delta_tilde": float(info.get("Delta_tilde", np.nan)),
        "time_s": elapsed,
    }


def run_sweep(settings, W):
    """Run all settings over all seeds. Returns a list of per-run rows."""
    rows = []
    for s in settings:
        for j in range(N_SEEDS):
            seed = SEED + j
            try:
                res = run_once(W, s, seed)
            except (ValueError, NotImplementedError, TypeError) as err:
                print(f"[skip] {s['label']}: {err}")
                break
            rows.append({"label": s["label"], "seed": seed, **res})
            print(f"  {s['label']:<28} seed {seed}: cond(Q)={res['cond_Q']:.3g} "
                  f"err={res['fact_error']:.2e} time={res['time_s']:.2f}s")
    return rows


def summarize(rows):
    """Mean / min / max of every metric for each label (order preserved)."""
    labels = list(dict.fromkeys(r["label"] for r in rows))
    out = []
    for lab in labels:
        group = [r for r in rows if r["label"] == lab]
        entry = {"label": lab, "n_seeds": len(group)}
        for m in METRICS:
            v = np.array([g[m] for g in group], dtype=float)
            entry[m + "_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
            entry[m + "_min"] = float(np.nanmin(v)) if np.isfinite(v).any() else np.nan
            entry[m + "_max"] = float(np.nanmax(v)) if np.isfinite(v).any() else np.nan
        out.append(entry)
    return out


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_summary(path, summary, title, connect=True):
    """One panel per metric: mean with min-max bars over the random seeds."""
    shown = [p for p in PLOTS
             if np.isfinite([s[p[0] + "_mean"] for s in summary]).any()]
    fig, axes = plt.subplots(1, len(shown), figsize=(4.2 * len(shown), 3.8))
    axes = np.atleast_1d(axes)
    x = np.arange(len(summary))
    labels = [s["label"] for s in summary]
    for ax, (col, ylabel, logy) in zip(axes, shown):
        mean = np.array([s[col + "_mean"] for s in summary])
        lo = np.array([s[col + "_min"] for s in summary])
        hi = np.array([s[col + "_max"] for s in summary])
        ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt="o-" if connect else "o", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_ablation(name, title, settings, W=None, connect=True):
    """Run, save and plot one ablation; returns the summary list.

    connect=False draws separate points (use it when the settings are
    categories, not an ordered numeric scale such as block size).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if W is None:
        W = generate_hard_matrix(N, M)
    print(f"{name}: n={W.shape[0]}, m={W.shape[1]}, seeds={N_SEEDS}")
    rows = run_sweep(settings, W)
    if not rows:
        print("No successful runs - nothing saved.")
        return []
    summary = summarize(rows)
    _write_csv(os.path.join(RESULTS_DIR, f"{name}_raw.csv"), rows)
    _write_csv(os.path.join(RESULTS_DIR, f"{name}_summary.csv"), summary)
    plot_summary(os.path.join(RESULTS_DIR, f"{name}.png"), summary, title, connect)

    print("\nSummary (mean over seeds)")
    print(f"{'setting':<28}{'cond(Q)':>12}{'||W-QR||/||W||':>16}{'Delta':>10}{'time (s)':>10}")
    for s in summary:
        print(f"{s['label']:<28}{s['cond_Q_mean']:>12.3g}{s['fact_error_mean']:>16.2e}"
              f"{s['Delta_mean']:>10.2e}{s['time_s_mean']:>10.2f}")
    print(f"\nSaved results/{name}_raw.csv, _summary.csv and results/{name}.png")
    return summary
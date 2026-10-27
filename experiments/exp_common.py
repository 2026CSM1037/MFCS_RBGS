"""Shared helpers for exp1_qr_stability, exp2_gmres and exp3_rr.

Each experiment produces, for every method, one value per iteration of a few
quantities (cond(Q), error, ...). `save_and_plot` writes them to
results/<name>.csv (one row per method and iteration) and draws one panel per
quantity, like the figures in Section 5 of the paper.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from experiments.ablation_common import RESULTS_DIR, _write_csv   # also sets the Agg backend


def save_and_plot(name, title, curves, panels):
    """curves: {method label: {quantity: 1-D array over iterations}}
    panels: [(quantity, y-axis label)], all drawn on a log scale.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for label, data in curves.items():
        length = max(len(v) for v in data.values())
        for i in range(length):
            row = {"method": label, "iteration": i + 1}
            for q, _ in panels:
                v = data.get(q)
                row[q] = float(v[i]) if v is not None and i < len(v) else np.nan
            rows.append(row)
    _write_csv(os.path.join(RESULTS_DIR, f"{name}.csv"), rows)

    # one fixed colour per method, so it matches across panels
    palette = plt.get_cmap("tab10").colors
    colors = {label: ("black" if label.startswith("sqrt") else palette[i % len(palette)])
              for i, label in enumerate(curves)}

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.0))
    for ax, (q, ylabel) in zip(np.atleast_1d(axes), panels):
        for label, data in curves.items():
            v = data.get(q)
            if v is None or not np.isfinite(v).any():
                continue
            marker = "" if len(v) > 40 or label.startswith("sqrt") else "o"
            ax.plot(np.arange(1, len(v) + 1), v, marker=marker, color=colors[label],
                    label=label, markersize=3, linewidth=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"{name}.png"), dpi=150)
    plt.close(fig)
    print(f"Saved results/{name}.csv and results/{name}.png")


def print_table(curves, columns):
    """Final value of each quantity per method. columns: [(quantity, header)]."""
    print(f"\n{'method':<26}" + "".join(f"{h:>16}" for _, h in columns))
    for label, data in curves.items():
        cells = []
        for q, _ in columns:
            v = data.get(q)
            cells.append(f"{v[-1]:>16.2e}" if v is not None and len(v) else f"{'-':>16}")
        print(f"{label:<26}" + "".join(cells))
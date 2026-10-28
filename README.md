# Randomized Block Gram-Schmidt (RBGS) - MFCS Project

From-scratch implementation and study of

> O. Balabanov and L. Grigori, *Randomized block Gram-Schmidt process for the
> solution of linear systems and eigenvalue problems*, arXiv:2111.14641.

RBGS computes a QR factorization W = QR of a tall, skinny matrix by
orthogonalizing a small random sketch of the vectors instead of the full ones.
We implement RBGS, the standard baselines (BCGS, BMGS, BCGS2), and the
randomized block Arnoldi / GMRES / Rayleigh-Ritz methods built on top of it,
then reproduce the paper's experiments and run ablation studies.

Report deadline: 28 Oct 2026.

**Quick start** (after the one-time setup in section 1), from the repo root:

    pytest                                     # 53 tests should pass
    python -m experiments.exp1_qr_stability    # paper Figure 1
    python -m experiments.exp2_gmres           # paper Figure 2
    python -m experiments.exp3_rr              # paper Figure 3
    python -m experiments.ablation_precision   # one of the six ablations

Results (CSV + PNG) are written to `results/`.

---

## 1. Getting started (do this once)

You need Python 3.9 or newer and Git.

**Step 1 - Clone the repository and go into it**

    git clone <repo-url>
    cd MFCS_RBGS

**Step 2 - Create a virtual environment** (an isolated Python for this project)

    python -m venv .venv

**Step 3 - Activate it.** You must do this every time you open a new terminal.

Windows (PowerShell):

    .venv\Scripts\Activate.ps1

Windows (Command Prompt):

    .venv\Scripts\activate.bat

macOS / Linux:

    source .venv/bin/activate

When it works, your prompt starts with `(.venv)`.

If PowerShell says "running scripts is disabled", run this once and retry:

    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

**Step 4 - Install the packages** (numpy, scipy, matplotlib, pytest)

    pip install -r requirements.txt

**Step 5 - Check that everything works**

    pytest

You should see **53 passed**. If something fails, see section 7.

---

## 2. Project layout

    MFCS_RBGS/
    ├── pytest.ini              lets pytest find src/ (must be in the repo root)
    ├── requirements.txt
    ├── src/                    the library
    │   ├── rbgs.py             RBGS, Algorithm 2.1 (rbgs, RBGSProcess)
    │   ├── krylov.py           block Arnoldi, GMRES, Rayleigh-Ritz, ILU preconditioner
    │   ├── baselines.py        BCGS, BMGS, BCGS2
    │   ├── sketch.py           sketching matrices Theta (SRHT, Rademacher)
    │   ├── matrices.py         test matrices and the paper's problems
    │   └── metrics.py          cond(Q), errors, Theorem 3.2 certificates
    ├── tests/                  one test file per src/ module
    ├── experiments/
    │   ├── exp1_qr_stability.py    paper Figure 1 (Section 5.1)
    │   ├── exp2_gmres.py           paper Figure 2 (Section 5.2)
    │   ├── exp3_rr.py              paper Figure 3 (Section 5.3)
    │   ├── exp_common.py           shared code for the three experiments
    │   ├── ablation_*.py           six ablation studies (section 5)
    │   ├── ablation_common.py      shared code for the ablations
    │   └── config.py               settings shared by the ablations
    ├── results/                CSV files and figures written by the scripts
    └── theory/                 notes on the paper (algorithm, assumptions, theorems)

`src/` is a library of functions; the scripts in `experiments/` are what you
run to produce results.

---

## 3. How to run any script

All experiments and ablations are run the same way: **from the repo root
folder, with the virtual environment active, using `python -m`**:

    python -m experiments.<script name without .py>

For example `python -m experiments.exp2_gmres`.

Do not run `python experiments/exp2_gmres.py` and do not run from inside the
`experiments/` folder: both fail with `No module named 'experiments'`.

Every script prints its progress and a summary table, and saves its results
to `results/`. Re-running a script overwrites its previous files.

---

## 4. Experiments (reproducing the paper's figures)

| Command | Paper | Time* | Output in `results/` |
|---|---|---|---|
| `python -m experiments.exp1_qr_stability` | Figure 1, Section 5.1 | ~1 min | `exp1_qr_stability.csv`, `.png` |
| `python -m experiments.exp2_gmres` | Figure 2, Section 5.2 | ~30 s | `exp2_gmres.csv`, `.png` |
| `python -m experiments.exp3_rr` | Figure 3, Section 5.3 | ~3 min | `exp3_rr.csv`, `.png` |

\* on a laptop, with the default (development) sizes.

Each CSV has one row per method and iteration; each PNG has the two panels of
the corresponding paper figure.

### exp1_qr_stability - block QR of a numerically rank-deficient matrix

Compares BCGS, BMGS, BCGS2 (float32) with three RBGS variants (unique-precision
with CG or Householder in step 2, and multi-precision with CG) on the paper's
hard matrix (m = 300, blocks of 10, SRHT with k = 3000). Plots cond(Q) and
||W - QR||/||W|| after each block.

    python -m experiments.exp1_qr_stability              # n = 20,000 (default)
    python -m experiments.exp1_qr_stability --n 100000   # larger matrix
    python -m experiments.exp1_qr_stability --paper      # paper size n = 10^6

`--paper` needs several GB of RAM (W alone is 2.4 GB) and takes a long time.

**Expected:** BCGS breaks first (around block 8), BCGS2 later (around block
17-20), multi-precision RBGS keeps cond(Q) close to 1 at every block.

### exp2_gmres - block GMRES

Restarted block GMRES (restart every 30 iterations) with a BCGS, BCGS2, BMGS
or RBGS Arnoldi process in float32, right-preconditioned with ILU. Plots the
residual and cond(Q) per iteration.

    python -m experiments.exp2_gmres                                  # development problem
    python -m experiments.exp2_gmres --matrix data/Ga41As41H72.mtx    # paper problem
    python -m experiments.exp2_gmres --matrix data/Ga41As41H72.mtx --no-cond

**Expected:** BCGS and BMGS lose orthogonality and converge more slowly;
BCGS2 and RBGS converge together, and RBGS's cond(Q) stays small (it grows
within each restart cycle and resets after it).

### exp3_rr - eigenvalues with restarted Rayleigh-Ritz

Dominant eigenpairs of a symmetric matrix with clustered eigenvalues, using
Rayleigh-Ritz restarted every 50 iterations (200 in total) with BCGS, BCGS
restarted every 5, BCGS2, BMGS, RBGS (two sketch sizes), and subspace
iteration as a baseline. Plots the eigenpair error and cond(Q) per iteration.

    python -m experiments.exp3_rr                                  # development problem
    python -m experiments.exp3_rr --matrix data/Ga41As41H72.mtx    # paper problem
    python -m experiments.exp3_rr --matrix data/Ga41As41H72.mtx --no-cond

**Expected:** subspace iteration and restarted BCGS stagnate, BCGS becomes
unstable (cond(Q) ~ 1e16), BCGS2 and RBGS (k = 1500) converge to ~1e-14.
With k = 1000, RBGS can lose accuracy after the third restart for some random
seeds (see the note in `exp3_rr.py` and paper Section 4.4).

### Using the paper's matrix (exp2 and exp3)

Without `--matrix`, exp2 and exp3 use small generated matrices that show the
same behaviour as the paper but **not** its numbers. To reproduce the paper:

1. Download **Ga41As41H72** in Matrix Market format (`.mtx`) from the
   SuiteSparse Matrix Collection (sparse.tamu.edu, group PARSEC) and extract it,
   e.g. to `data/Ga41As41H72.mtx` (`data/` is not committed).
2. Pass the path with `--matrix`, as above.

With the paper matrix (n = 268,096) the runs take much longer. `--no-cond`
skips cond(Q), which costs one large SVD per iteration. The ILU drop
tolerance used for it is `PAPER_DROP_TOL` in `exp2_gmres.py`; increase it if
the preconditioner needs too much memory.

---

## 5. Ablation studies

Each ablation changes **one** setting of RBGS and keeps everything else at
the defaults in `experiments/config.py`. It runs every setting for several
random sketches (seeds) on the paper's hard matrix.

| Command | Varies | Settings compared |
|---|---|---|
| `python -m experiments.ablation_block_size` | block size mp | 2, 5, 10, 20, 50 (those dividing m) |
| `python -m experiments.ablation_precision` | working precision | all float64; multi-precision (f32/f64); unique float32 |
| `python -m experiments.ablation_qr_method` | step 4-5 QR method | l2_rcholesky, rcholesky, implicit_rcholesky, rgs |
| `python -m experiments.ablation_sketch_size` | sketch size k | 60, 90, 120, 180, 300, 600, 1200, 2400 (those with m <= k <= n) |
| `python -m experiments.ablation_sketch_type` | sketch distribution | srht, rademacher |
| `python -m experiments.ablation_solver_iters` | step 2 solver | Householder; Richardson with 1, 2, 3, 5, 10, 20 iterations; CG 20 |

With the default settings each one takes only a few seconds (2-7 s on a
laptop) and writes three files to `results/`:

| File | Contents |
|---|---|
| `<name>_raw.csv` | one row per setting and seed: cond(Q), orthogonality loss, \|\|W-QR\|\|/\|\|W\|\|, Delta, Delta_tilde, time |
| `<name>_summary.csv` | mean, min and max of each metric per setting |
| `<name>.png` | one panel per metric: mean with min-max bars over the seeds |

### Changing the ablation settings

Shared settings are in `experiments/config.py`:

    N = 3000        # rows of the test matrix
    M = 60          # columns
    MP = 10         # block size (when not varied)
    K = 600         # sketch size (when not varied)
    N_SEEDS = 2     # random sketches per setting

Increase `N`, `M` or `N_SEEDS` for results closer to the paper (runs take
longer). The list of values each ablation tries is at the top of its file
(for example `K_VALUES` in `ablation_sketch_size.py`). Every ablation file
also starts with its purpose, what is varied and fixed, and the prediction
made before running.

### Running everything

macOS / Linux:

    for s in exp1_qr_stability exp2_gmres exp3_rr ablation_block_size ablation_precision ablation_qr_method ablation_sketch_size ablation_sketch_type ablation_solver_iters; do python -m experiments.$s; done

Windows (PowerShell):

    foreach ($s in "exp1_qr_stability","exp2_gmres","exp3_rr","ablation_block_size","ablation_precision","ablation_qr_method","ablation_sketch_size","ablation_sketch_type","ablation_solver_iters") { python -m experiments.$s }

---

## 6. Tests

    pytest                         all 53 tests
    pytest tests/test_rbgs.py      one file
    pytest -v                      show each test name
    pytest -x                      stop at the first failure

| Test file | Tests | Checks |
|---|---|---|
| `test_sketch.py` | 5 | sketch shapes, reproducibility, invalid input |
| `test_baselines.py` | 4 | BCGS / BMGS / BCGS2 give W = QR with orthonormal Q |
| `test_metrics.py` | 7 | metrics on matrices with known answers |
| `test_rbgs.py` | 19 | W = QR, sketch orthonormality, Theorem 3.2 bounds, float32, RBGS stable where BCGS fails |
| `test_krylov.py` | 18 | Arnoldi identity, GMRES vs direct solve, preconditioner, known eigenvalues |

---

## 7. Troubleshooting

| Error | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` (pytest) | `pytest.ini` is missing from the repo root, or you are not in the repo root. Fallback: `python -m pytest`. |
| `No module named 'experiments'` | Run from the repo root with `python -m experiments.<name>`, not `python experiments/<name>.py`. |
| `No module named 'experiments.exp_common'` | `experiments/exp_common.py` is missing; restore it from git. |
| `ImportError: cannot import name 'RBGSProcess'` | `src/rbgs.py` is an old version; pull the latest. |
| `KeyError: 'cond_Q'` in exp3 | `src/krylov.py` is an old version (no `track_cond` in `rr`); pull the latest. |
| `test_ilu_is_exact_for_tridiagonal` fails | `src/krylov.py` is an old version (old ILU defaults); pull the latest. |
| `No module named 'numpy'` / `'matplotlib'` | The virtual environment is not active, or step 4 was skipped. |

To check which version of `krylov.py` you have:

    python -c "import inspect; from src.krylov import rr, ilu_preconditioner; print(inspect.signature(rr)); print(inspect.signature(ilu_preconditioner))"

The output should include `track_cond=False` and `drop_tol=0.0001, fill_factor=10.0`.

---

## 8. Notes for the report

- **Development vs paper sizes:** default runs use smaller matrices than the
  paper; say so when presenting them. Use `--paper` / `--matrix` for the
  real replication.
- **Speed:** on laptop-sized problems RBGS is slower than BCGS2. The paper's
  advantage is fewer flops and less communication at large n on parallel
  machines; do not claim speed-ups from small runs.
- **Sketch size:** for Krylov methods use k a few times larger than the basis
  (about 2-3 x (restart + 1) x block size); a sketch barely larger than the
  basis makes the basis ill-conditioned.
- **Step-2 solver:** `gmres` and `rr` default to 5 Richardson iterations
  (paper Sections 5.2-5.3). Richardson diverges on numerically rank-deficient
  matrices, so `rbgs` itself defaults to Householder.
- **Preconditioner:** the paper uses ILU(0); SciPy has none, so
  `ilu_preconditioner` uses threshold ILU. State this in the report.
- **Unique precision:** `sketch.py` multiplies in float64, so with
  `fine_dtype=np.float32` sketches are rounded to float32 rather than
  computed in it (slightly more accurate than true float32).
- **Sketched Rayleigh-Ritz:** its accuracy is not guaranteed by the
  embedding property alone (paper Section 4.4); check several seeds before
  drawing conclusions about small sketches.
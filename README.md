# Randomized Block Gram-Schmidt (RBGS) - MFCS Project

From-scratch implementation and study of

> O. Balabanov and L. Grigori, *Randomized block Gram-Schmidt process for the
> solution of linear systems and eigenvalue problems*, arXiv:2111.14641.

RBGS computes a QR factorization W = QR of a tall, skinny matrix by
orthogonalizing a small random sketch of the vectors instead of the full ones.
We implement RBGS, the standard baselines (BCGS, BMGS, BCGS2), and the
randomized block Arnoldi / GMRES / Rayleigh-Ritz methods built on top of it,
then compare them and run ablation studies.

Report deadline: 28 Oct 2026.

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

**Step 4 - Install the packages**

    pip install -r requirements.txt

**Step 5 - Check that everything works**

    pytest

You should see all tests pass (green). If you see `ModuleNotFoundError: No
module named 'src'`, check that:

- you are in the repo root folder (the one containing `src/`, `tests/` and
  `pytest.ini`), and
- the file `pytest.ini` exists there. As a fallback, run `python -m pytest`.

---


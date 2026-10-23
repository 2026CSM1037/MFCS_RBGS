# Concept Map: Course Concepts in the RBGS Paper

**Paper:** O. Balabanov and L. Grigori, *Randomized Block Gram–Schmidt Process for the Solution of Linear Systems and Eigenvalue Problems*, arXiv:2111.14641v2 (2023 version).

> Page numbers below refer to the arXiv v2 PDF pagination. Confirm them against the exact PDF attached to the project before final submission, because journal pagination differs.

| Course concept | Where it appears in the paper | How it is used |
|---|---|---|
| QR factorization | Section 1.2, block Gram–Schmidt; Section 2, RBGS algorithm; Algorithm 2.1 | Factorizes the input as \(W=QR\), where \(Q\) spans the same column space as \(W\), and \(R\) is upper triangular. RBGS changes how projection coefficients are computed by using a sketch. |
| Orthogonal projection | Section 1.2 and Section 2 | Previously computed basis blocks are projected out of the current input block. The quality of this step affects loss of orthogonality and numerical stability. |
| Least squares | Section 2.1, solution of the block least-squares problem; equation (9) in Section 2's motivation | RBGS solves a smaller sketched least-squares problem, \(\min_Y\|\Theta Q_{\mathrm{old}}Y-\Theta W_{\mathrm{new}}\|_F\), instead of solving the corresponding tall problem directly. |
| Singular values and condition number | Section 1.1 preliminaries; Section 3 stability analysis, especially Theorem 3.2 | Singular values measure how much a matrix stretches vectors. The ratio \(\sigma_{\max}(Q)/\sigma_{\min}(Q)\) is its condition number; a well-conditioned \(Q\) indicates that the computed basis has not become close to rank deficient. |
| Cholesky factorization / Cholesky QR | Section 2.2, inter-block QR factorization; Algorithm 2.1 and the RCholeskyQR variant | Cholesky QR obtains a QR factorization through a Gram matrix and a Cholesky factorization. It is efficient, but its stability depends on conditioning and precision. |
| Random embedding / sketching | Section 1.3, Definitions 1.1–1.2 and Corollaries 1.3–1.4 | The random matrix \(\Theta\) maps vectors to a smaller space while approximately preserving inner products on the relevant subspace. This makes the smaller least-squares problem meaningful. |
| Floating-point error and mixed precision | Section 1.4 and Section 3 | The analysis distinguishes coarse- and fine-precision unit roundoffs and tracks rounding errors. Expensive large matrix operations can use coarse precision while selected smaller operations use fine precision. |
| Eigenvalues and Rayleigh–Ritz | Section 4, randomized block Krylov methods, particularly the Rayleigh–Ritz discussion | Rayleigh–Ritz extracts approximate eigenpairs from a computed subspace by solving a smaller projected eigenvalue problem. The sketched Galerkin condition is used to define the randomized variant. |
| Low-rank structure / shrinking idea | Section 4's Krylov and Rayleigh–Ritz discussion; connect to the experiment section used by the project | A large problem is reduced to a smaller representation or subspace. The sketch reduces the dimension of projection/least-squares calculations; the Krylov subspace similarly compresses the eigenvalue or linear-system problem. Avoid describing this as a low-rank approximation theorem unless the specific passage establishes one. |
| GMRES and residual minimization | Section 4, randomized GMRES discussion | GMRES searches a Krylov subspace for a solution with a small residual. The randomized version minimizes a sketched residual; under the embedding and stability assumptions, the paper establishes quasi-optimality up to a factor involving \(\operatorname{cond}(Q)\). |

## Conceptual flow

1. Start with a matrix \(W\) and seek a stable QR factorization \(W\approx QR\).
2. Use \(\Theta\) to form smaller sketches \(\Theta W\) and \(\Theta Q\).
3. Solve projection/least-squares problems in sketch space, reducing the cost of operations involving tall matrices.
4. Control the rounding errors and verify the computed basis using the stability analysis.
5. Use the resulting basis in block Krylov methods: GMRES for linear systems and Rayleigh–Ritz for eigenvalue problems.

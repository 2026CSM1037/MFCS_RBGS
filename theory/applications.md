# How RBGS Theory Connects to GMRES and Rayleigh–Ritz

**Source:** Balabanov and Grigori, *Randomized Block Gram–Schmidt Process for the Solution of Linear Systems and Eigenvalue Problems*, Section 4, especially the randomized GMRES discussion (around paper p. 14) and the Rayleigh–Ritz discussion (around pp. 15–16 in the project brief's pagination).

## 1. Randomized GMRES: residual minimization and quasi-optimality

GMRES solves a linear system by searching within a Krylov subspace. At each iteration, it chooses the candidate whose residual norm is smallest among the candidates in the current search space. Standard GMRES uses the Euclidean residual norm. The randomized version uses a sketching matrix \(\Theta\) and minimizes the sketched residual norm instead:

\[
\min_y \|\Theta(b-Ax_0-\mathcal{K}y)\|_2,
\]

where \(\mathcal{K}\) represents the relevant Krylov-space action/basis in the paper's notation.

The benefit is that the residual comparison can be carried out in a lower-dimensional space. If \(\Theta\) is an \(\varepsilon\)-embedding for the relevant residual/search subspace, the sketched norm approximates the original norm. Thus a minimizer of the sketched problem cannot be arbitrarily worse than the best candidate in the original norm.

The paper's argument gives a quasi-optimality result for stable randomized GMRES, with a multiplicative factor involving \(\operatorname{cond}(Q)\), where \(Q\) is the computed basis. Intuitively, two ingredients are needed:
1. the embedding makes sketched and original residual norms comparable on the relevant subspace;
2. the computed basis must be sufficiently well-conditioned for the representation in basis coordinates not to amplify errors excessively.

**What this does not mean:** randomized GMRES is not guaranteed to match standard GMRES exactly, nor is the result unconditional. The embedding property and stability/conditioning requirements matter.

## 2. Rayleigh–Ritz: a sketched Galerkin condition

Rayleigh–Ritz approximates eigenpairs of a large matrix by restricting the eigenvalue problem to a smaller trial subspace. If \(Q\) spans the trial subspace, the classical Galerkin condition requires the eigen-residual to be orthogonal to that subspace:

\[
Q^T(AQy-\lambda Qy)=0.
\]

The randomized method replaces this full-space orthogonality condition by its sketched counterpart:

\[
(\Theta Q)^T\Theta(AQy-\lambda Qy)=0.
\]

This is the **sketched Galerkin condition**. It enforces orthogonality after applying the sketch rather than directly in the original inner product. The reduced problem can be cheaper because the sketch has fewer rows than the original matrix.

The paper reformulates the analysis using projection operators to bound the residual error of the randomized Rayleigh–Ritz approximation. The main intuition is that the sketch should preserve the geometry of the relevant subspaces well enough for the sketched orthogonality condition to remain informative about the original residual.

## 3. How the ideas fit together

- **RBGS** constructs a basis efficiently while controlling its numerical quality.
- **Randomized Arnoldi** uses the basis construction to build block Krylov subspaces.
- **Randomized GMRES** uses the subspace to minimize a sketched residual.
- **Randomized Rayleigh–Ritz** uses the subspace to obtain approximate eigenpairs through sketched Galerkin orthogonality.

The common principle is to do the expensive search/projection work in a smaller sketch space, while relying on embedding and stability results to relate the result to the original problem.

## 4. Practical interpretation

The method is attractive when the original matrices are tall or expensive to access, and when operations on a smaller sketch can reduce computation or communication. However, sketch dimension, sketch quality, basis conditioning, and floating-point precision must be chosen carefully. Small-scale NumPy timings alone do not demonstrate the parallel-hardware advantage emphasized by the paper.

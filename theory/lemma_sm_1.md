# Supporting Result: Lemma SM.1 — Sketched and Ordinary Projectors

**Purpose:** This note records the role of the supplementary projector result requested in the project brief. It is intended to connect the sketch-space Galerkin condition to the ordinary Euclidean projector used in the Rayleigh–Ritz analysis.

## Projector intuition

Let \(Q\) have full column rank and let \(\Theta Q\) also have full column rank. The Euclidean orthogonal projector onto \(\operatorname{range}(Q)\) is

\[
P_Q=Q(Q^TQ)^{-1}Q^T.
\]

The corresponding projector associated with the sketched inner product is

\[
P_{Q,\Theta}
=Q\big((\Theta Q)^T(\Theta Q)\big)^{-1}(\Theta Q)^T\Theta.
\]

It projects a vector \(x\) onto \(\operatorname{range}(Q)\) by making the residual orthogonal to \(Q\) in the sketched inner product:
\[
(\Theta Q)^T\Theta(x-P_{Q,\Theta}x)=0.
\]

The ordinary projector instead imposes \(Q^T(x-P_Qx)=0\). If the sketch preserves inner products well on the subspaces involved, the two projectors are related and the sketched condition can be interpreted as a controlled approximation to ordinary Galerkin orthogonality.

## Why this result is useful

Rayleigh–Ritz analysis is naturally stated in terms of projection operators and residuals. The randomized method uses a sketched Galerkin condition, so a projector comparison is the natural bridge between the algorithm's condition and the standard eigen-residual interpretation.

## Important source-verification note

The project brief names **Lemma SM.1**, which appears to refer to the paper's supplementary material rather than the main paper body. The exact statement and constants of that supplementary lemma are not reproduced here because they must be copied from the team's specific supplementary PDF. Do not submit this explanatory note as a verbatim statement or proof of Lemma SM.1. To finish this optional item, insert the exact lemma statement from the supplement and then use the projector definitions above to explain its role.

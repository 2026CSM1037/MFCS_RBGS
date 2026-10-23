# Proposition 3.5 — When an Embedding for \(W\) Also Embeds the Computed \(Q\)

**Source:** Balabanov and Grigori, *Randomized Block Gram–Schmidt Process for the Solution of Linear Systems and Eigenvalue Problems*, Section 3.3, Proposition 3.5.

## Statement (formatted)

Consider RBGS with a backward-stable solver for Step 2 satisfying Assumptions 2.2 and the paper's Assumptions 3.1. If \(\Theta\) is an \(\varepsilon\)-embedding for the computed input \(\widehat W\), with \(\varepsilon\leq 1/4\), then \(\Theta\) is an \(\varepsilon'\)-embedding for the computed basis \(\widehat Q\), where the paper's bound is

\[
\varepsilon' = 2\varepsilon+
180u_{\mathrm{crs}}m_p^2p^2\operatorname{cond}(\widehat W).
\]

Use the exact notation and assumptions from the project PDF when transcribing this into the final report. In particular, the paper's statement depends on the backward-stability assumptions for the least-squares solver and on Assumptions 3.1; it is not an unconditional property of every sketch.

## Idea in one paragraph (no proof required)

The sketch \(\Theta\) is chosen independently of \(W\), so an oblivious subspace embedding can preserve the geometry of the input column space with high probability. RBGS computes \(Q\) from \(W\) using finite-precision operations, however, so \(Q\) is not simply identical to \(W\). Proposition 3.5 bounds how much the embedding parameter must worsen when moving from \(W\) to the computed basis \(Q\). The new distortion \(\varepsilon'\) includes the original embedding error and a floating-point term that grows with the block size/iteration factors and the condition number of \(W\).

## What the result says and does not say

- It gives a route to justify the embedding property for the computed basis from an embedding guarantee for the input matrix.
- The extra term shows why poor conditioning or larger accumulated rounding errors can weaken the guarantee.
- It relies on the stated stability and precision assumptions.
- It does not say that an arbitrary random sketch always embeds the computed basis, nor that \(\varepsilon'=\varepsilon\).

## Reference

Section 3.3, Proposition 3.5; the paper notes that the result restates the corresponding single-vector RGS result for the block algorithm.

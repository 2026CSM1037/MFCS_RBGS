# Theorem 3.4: A Priori Stability of RBGS

This note explains Theorem 3.4 from Balabanov and Grigori, *Randomized Block Gram–Schmidt Process for the Solution of Linear Systems and Eigenvalue Problems*.

## 1. What does “a priori” mean?

An **a priori** guarantee establishes bounds from conditions imposed before running the algorithm. In contrast with Theorem 3.2, it does not rely only on measuring the final error indicators: it assumes properties of the input matrix, the least-squares solver, the sketch, and the arithmetic.

Theorem 3.4 gives a route to predicting the accuracy of RBGS when these conditions are satisfied.

## 2. Why are more assumptions needed?

The paper explains that deriving bounds on the error indicators in advance requires more information than certifying them afterward. In particular:

- the least-squares solver in Step 2 must satisfy a stability condition;
- the input matrix must be numerically full rank at the working precision;
- the sketch must have the required embedding property;
- the rounding-error assumptions from Section 3.1 must hold.

A matrix that is numerically rank-deficient can make orthogonalization unreliable: directions that are mathematically distinct may become indistinguishable at the chosen precision.

## 3. Conditions in Theorem 3.4

The theorem considers Algorithm 2.1 with a backward-stable least-squares solver satisfying Assumptions 2.2. Under Assumptions 3.1, it assumes that $\Theta$ is an $\varepsilon$-embedding for $\widehat Q_{(1:p-1)}$ and $\widehat W$, with $\varepsilon\leq 1/2$. (Note: $\widehat Q_{(1:p-1)}$, not the full $\widehat Q$ as in Theorem 3.2.)

Assumptions 2.2 (backward stability of step 2) require each column $\hat x$ of $\widehat R_{(1:i-1,i)}$ to solve $\min_y\|(\widehat S_{(1:i-1)}+\Delta S)y-(\hat p+\Delta p)\|$ exactly for some perturbations with $\|\Delta S\|_F\le0.01u_{\mathrm{crs}}\|\widehat S_{(1:i-1)}\|$ and $\|\Delta p\|\le0.01u_{\mathrm{crs}}\|\hat p\|$.

The paper additionally requires the coarse precision to satisfy a condition of the form

$$
u_{\mathrm{crs}}
\leq
10^{-3}\operatorname{cond}(\widehat W)^{-1}m^{-2},
$$

and the fine precision to satisfy

$$
u_{\mathrm{fine}}
\leq
10^{-2}m^{-1/2}n^{-3/2}u_{\mathrm{crs}}.
$$

Here $\operatorname{cond}(\widehat W)$ is the condition number of the relevant computed input matrix in the paper's notation. A condition number describes sensitivity: a large value means the matrix is more ill-conditioned.

**Important:** These bounds should be quoted together with all the theorem's assumptions. They are not universal requirements for every QR algorithm or every numerical problem.

## 4. Conclusions of the theorem

The theorem bounds the two error indicators:

$$
\widetilde{\Delta}^{(p)}
\leq
4.2u_{\mathrm{crs}}m^{3/2}
\frac{\|\widehat W\|_F}{\|\widehat P\|_F}
\leq 6u_{\mathrm{crs}}m^{3/2},
\tag{3.7}
$$

and

$$
\Delta^{(p)}
\leq 20u_{\mathrm{crs}}m^2
\operatorname{cond}(\widehat W).
\tag{3.8}
$$

The $6u_{\mathrm{crs}}m^{3/2}$ bound is the second inequality in (3.7) — the same quantity $\widetilde\Delta^{(p)}=\|\widehat P-\widehat S\widehat R\|_F/\|\widehat P\|_F$, not a separate result. It follows from $\|\widehat P\|_F\ge0.7\|\widehat W\|_F$ (end of the proof in Section 6).

Checked against p. 12 of arXiv:2111.14641v2.

## 5. Interpreting the bounds

The bounds reveal the trade-offs:

- **Coarse precision:** smaller $u_{\mathrm{crs}}$ improves the error bounds, but may reduce the speed or energy benefit of using lower precision.
- **Block size / number of columns $m$:** the bounds contain powers of $m$, so the permitted rounding error becomes more restrictive as the matrix becomes wider.
- **Condition number:** the orthogonality bound depends on $\operatorname{cond}(\widehat W)$. Poorly conditioned input matrices require more care.
- **Ambient dimension $n$:** the theorem's stated condition on $u_{\mathrm{fine}}$ includes $n$, but the paper emphasizes that the stability guarantee for the coarse working precision can be independent of the large ambient dimension under the stated conditions.

## 6. Main intuition

Theorem 3.4 follows the pattern of many numerical-stability results:

1. assume the input is sufficiently well-conditioned at the working precision;
2. assume the sketch preserves the relevant subspace geometry;
3. assume the least-squares and QR steps behave stably;
4. control the rounding errors introduced by the mixed-precision update;
5. derive upper bounds on the final orthogonality and factorization indicators.

If the input is too close to rank-deficient for the chosen precision, the theorem's assumptions may fail, and its stability guarantee no longer applies.

## 7. Comparison with Theorem 3.2

| Feature | Theorem 3.2 | Theorem 3.4 |
|---|---|---|
| Type | A posteriori | A priori |
| Main approach | Measure indicators after execution | Derive indicator bounds from prior assumptions |
| Condition number requirement | No condition-number assumption of the same type is needed for its certification result | Requires a condition involving the input matrix's condition number |
| Least-squares solver | Does not require the same prior solver-stability condition for certification | Requires a backward-stable solver satisfying the stated assumptions |
| Practical interpretation | “The computed run passes the checks” | “Under these input and arithmetic conditions, the run is expected to meet the derived bounds” |

## Source

Balabanov and Grigori, *Randomized Block Gram–Schmidt Process for the Solution of Linear Systems and Eigenvalue Problems*, Section 3.2.2, Theorem 3.4 and equations (3.7)–(3.8).

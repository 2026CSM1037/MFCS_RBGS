# Assumptions for the Stability Analysis of RBGS

This note summarizes **Assumptions 3.1** in Balabanov and Grigori, *Randomized Block Gram–Schmidt Process for the Solution of Linear Systems and Eigenvalue Problems*. The assumptions are presented in an explanatory form; the paper's equations remain the authoritative definitions.

## 1. Why assumptions are needed

A stability theorem cannot guarantee that an algorithm behaves well for every possible input and every possible rounding error. The analysis therefore places conditions on:

1. the sketching matrix $\Theta$;
2. rounding errors in the coarse-precision update in Algorithm 2.1, Step 3;
3. the QR factorization and sketch computations in Steps 4–5.

Together, these conditions let the authors bound how far the computed factors are from the desired factorization and how well the sketch preserves the relevant geometry.

## 2. Assumption on the norm of the sketching matrix

The paper assumes, for some $\varepsilon \leq 1/2$,

$$
\|\Theta\|_F \leq \sqrt{1+\varepsilon}\sqrt{n}.
\tag{3.2}
$$

Here, $\|\Theta\|_F$ is the Frobenius norm and $n$ is the number of rows of the original matrix.

**Intuition:** the sketching matrix must have a controlled norm. If its norm were excessively large, rounding errors introduced before or during sketching could be amplified.

The paper states that this condition holds with probability at least $1-\delta$ if $\Theta$ is a $(1/2,\delta/n,1)$ oblivious $\ell_2$-subspace embedding (Corollary 1.3).

## 3. Assumptions on the Step 3 rounding error

Step 3 of Algorithm 2.1 computes the residual block

$$
Q'_{(i)} = W_{(i)} - Q_{(1:i-1)}R_{(1:i-1,i)}
$$

in coarse precision. The computed result therefore includes rounding error. The paper denotes that error by $\Delta Q'_{(i)}$.

Assumption (3.3a) bounds the entrywise magnitude of this error:

$$
|\Delta Q'_{(i)}|
\leq 1.02\,u_{\mathrm{crs}}
\left(
|\widehat W_{(i)}|+
i\,m_p\,|\widehat Q_{(1:i-1)}|\,|\widehat R_{(1:i-1,i)}|
\right).
\tag{3.3a}
$$

Absolute-value signs around matrices are entrywise, and $|\widehat Q_{(1:i-1)}||\widehat R_{(1:i-1,i)}|$ is an ordinary matrix product of absolute-value matrices. The bound holds for $1\le i\le p$. The factor $i\,m_p$ is the worst-case dot-product length factor (inner dimension $(i-1)m_p$, plus the subtraction); it is what produces the $m^{3/2}$ in Theorem 3.2's residual bound.

The paper also assumes a bound on the sketched rounding error:

$$
\|\Theta\Delta Q'_{(i)}\|_F
\leq
1.02\,u_{\mathrm{crs}}\sqrt{1+\varepsilon}
\left\|
|\widehat W_{(i)}|+
i\,m_p\,|\widehat Q_{(1:i-1)}|\,|\widehat R_{(1:i-1,i)}|
\right\|_F.
\tag{3.3b}
$$

(3.3a) is the standard worst-case rounding bound. (3.3b) holds under the probabilistic rounding model with probability at least $1-4\delta$ if $\Theta$ is a $(1/8,\,m^{-1}\binom{n}{d}^{-1}\delta,\,d)$ oblivious subspace embedding with $d=\mathcal O(\log(m/\delta))$; e.g. Rademacher with $k=\mathcal O(\log n\log(m/\delta))$ or SRHT with $k=\mathcal O(\log^2 n\log^2(m/\delta))$.

**Intuition:** the update is deliberately done in coarse precision to reduce cost. These conditions limit the rounding error both in the original update and after applying the sketch. The sketch must not make the update error unexpectedly large.

## 4. Assumptions on the QR step and the new sketch

Steps 4–5 orthonormalize the residual block with respect to $\langle\Theta\cdot,\Theta\cdot\rangle$ (by any of the routines of Section 2.2 — RGS, RCholeskyQR, implicit RCholeskyQR, or $\ell_2$-QR+RCholeskyQR) and form the sketch of the result. The paper assumes:

$$
|\widehat Q'_{(i)}-\widehat Q_{(i)}\widehat R_{(i,i)}|
\le u_{\mathrm{fine}}m\big(|\widehat Q'_{(i)}|+|\widehat Q_{(i)}||\widehat R_{(i,i)}|\big)
\le 0.1u_{\mathrm{crs}}|\widehat Q_{(i)}||\widehat R_{(i,i)}|,
\tag{3.4a}
$$

$$
\|\Theta(\widehat Q'_{(i)}-\widehat Q_{(i)}\widehat R_{(i,i)})\|_F
\le u_{\mathrm{fine}}m\|\Theta\|_F\,\big\||\widehat Q'_{(i)}|+|\widehat Q_{(i)}||\widehat R_{(i,i)}|\big\|
\le 0.1u_{\mathrm{crs}}\|\widehat Q_{(i)}\|\,\|\widehat R_{(i,i)}\|,
\tag{3.4b}
$$

and, if $\Theta$ is an $\varepsilon'$-embedding for $\widehat Q'_{(i)}$,

$$
1-0.1u_{\mathrm{crs}}\operatorname{cond}(\widehat Q'_{(i)})
\le\sigma_{\min}(\Theta\widehat Q_{(i)})
\le\sigma_{\max}(\Theta\widehat Q_{(i)})
\le1+0.1u_{\mathrm{crs}}\operatorname{cond}(\widehat Q'_{(i)}),
\qquad
\|\widehat Q_{(i)}\|\le(1-\varepsilon')^{-1/2}\|\Theta\widehat Q_{(i)}\|.
\tag{3.5}
$$

Norms without a subscript are spectral ($\|\cdot\|=\sigma_{\max}$). (3.4) says the block factorization is accurate at fine precision, small relative to the coarse error scale. (3.5) says the routine returns a block whose **sketch** is nearly orthonormal, losing only $\mathcal O(u_{\mathrm{crs}}\operatorname{cond}(\widehat Q'_{(i)}))$.

The implicit RCholeskyQR (Section 2.2.3) does not satisfy (3.5) directly; the paper covers it by a perturbation argument reducing it to regular RCholeskyQR. Theorem 3.2 is stated "possibly excluding (3.5)": the a posteriori result does not use (3.5) at all.

**Intuition:** the orthonormalization step and the sketch must be accurate enough that they do not undo the error control established for the residual update.

## 5. Role of the two precision levels

The paper uses:

- $u_{\mathrm{crs}}$: unit roundoff for coarse-precision operations;
- $u_{\mathrm{fine}}$: unit roundoff for fine-precision operations.

Smaller unit roundoff generally means higher precision. Algorithm 2.1 uses coarse precision for the expensive update in Step 3 and fine precision for the other listed operations.

The stability analysis imposes relationships between these precision levels. For example, Theorem 3.2 requires

$$
100m^{1/2}n^{3/2}u_{\mathrm{fine}}
\leq u_{\mathrm{crs}}\leq 0.01,
$$

in addition to the applicable assumptions. This is a theorem-specific condition, not a replacement for Assumptions 3.1.

## 6. How the assumptions are used

| Condition | What it controls | Why it matters |
|---|---|---|
| Bound on $\Vert \Theta\Vert _F$ | Size of the sketching operator | Helps control errors after sketching |
| Bound (3.3a) | Entrywise rounding error in Step 3 | Controls the coarse-precision residual update |
| Bound (3.3b) | Sketched Step 3 error | Controls how the sketch interacts with update error |
| Bounds (3.4a)–(3.4b) | Error in the block QR factorization | Keeps the computed factorization close to its intended relation |
| Condition (3.5) | Sketch-orthonormality of the output block $\Theta\widehat Q_{(i)}$ | Needed for the a priori analysis (Thm 3.4, Prop 3.5); not needed by Thm 3.2 |

## 7. Important distinction

These are **analysis assumptions**, not extra instructions that a user must manually add to Algorithm 2.1. Some are rounding-error models; others describe properties of the sketch or of the QR routine.

The paper explains probabilistic conditions under which certain sketch assumptions hold. Do not claim that every random matrix automatically satisfies them.

## 8. Connection to the next files

- `theorem_3_2.md` explains an **a posteriori** guarantee: after running RBGS, compute two error indicators and check whether they are sufficiently small.
- `theorem_3_4.md` explains an **a priori** guarantee: impose additional conditions, including a condition involving the input matrix's condition number, to derive bounds before running the algorithm.

## Source

Balabanov and Grigori, *Randomized Block Gram–Schmidt Process for the Solution of Linear Systems and Eigenvalue Problems*, Section 3.1, Assumptions 3.1, and Section 3.2.1–3.2.2.


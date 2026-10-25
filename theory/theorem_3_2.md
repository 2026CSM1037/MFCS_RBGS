# Theorem 3.2 — A Posteriori Stability of RBGS

**Source:** Balabanov and Grigori, *Randomized Block Gram–Schmidt Process for the Solution of Linear Systems and Eigenvalue Problems*, Section 3.2.1, Theorem 3.2; proof in Section 6.

This note follows the paper's theorem and proof. Hats indicate computed quantities, and the paper's notation is retained where possible. All displayed equations below were checked against the page images of arXiv:2111.14641v2 (pp. 11–12 and 21). Block indices are subscripts, as in the paper ($W_{(i)}$); superscripts are only iteration counts such as $\Delta^{(p)}$.

## 1. What does “a posteriori” mean?

An *a posteriori* result checks numerical quality after the algorithm has run. RBGS stores the sketches of the computed basis and input, together with the computed $R$ factor. It can use these comparatively small quantities to form error indicators.

Theorem 3.2 says that if the assumptions hold and these indicators are sufficiently small, then the computed basis has controlled singular values and the computed factorization has a small residual.

## 2. Symbols used in the theorem

- $W$: input matrix; $\widehat W$ denotes its computed/input representation in the paper's floating-point analysis.
- $\widehat Q$: computed basis factor.
- $\widehat R$: computed upper-triangular factor.
- $\Theta$: sketching matrix.
- $\widehat S$: computed sketch of the basis, approximately $\Theta\widehat Q$.
- $\widehat P$: computed sketch of the input, approximately $\Theta\widehat W$.
- $m$: number of columns of the input matrix.
- $n$: number of rows of the input matrix.
- $p$: number of column blocks; $m_p$: number of columns in each block, so $m=p\,m_p$ (paper, Section 1.2).
- $\|\cdot\|$ without subscript: spectral norm, $\|\cdot\|=\sigma_{\max}(\cdot)$ (paper, Section 1.1). Keep it distinct from $\|\cdot\|_F$: several bounds below use one and not the other.
- $u_{\mathrm{crs}}$: unit roundoff of the coarse-precision arithmetic.
- $u_{\mathrm{fine}}$: unit roundoff of the fine-precision arithmetic.
- $\varepsilon$: embedding distortion parameter.
- $\sigma_{\min}(A)$, $\sigma_{\max}(A)$: smallest and largest singular values of $A$.
- $\|\cdot\|_F$: Frobenius norm, the square root of the sum of squared entries.

The two indicators are defined below. They measure different properties and should not be confused.

### 2.1 Sketch orthogonality indicator

$$
\Delta^{(p)}=\|I-\widehat S^T\widehat S\|_F.
$$

If the columns of $\widehat S$ are exactly orthonormal, then $\widehat S^T\widehat S=I$ and this indicator is zero. A small value says the computed basis is close to orthonormal **in sketch space**.

### 2.2 Relative sketched factorization residual

$$
\widetilde{\Delta}^{(p)}
=\frac{\|\widehat P-\widehat S\widehat R\|_F}
{\|\widehat P\|_F}.
$$

A small value says that $\widehat S\widehat R$ reproduces the input sketch $\widehat P$ accurately. It is a factorization residual in sketch space, not the same as the original-space residual.

## 3. Complete theorem statement

**Theorem 3.2 (paper, notation formatted for readability).** Consider Algorithm 2.1. Assume that

$$
100m^{1/2}n^{3/2}u_{\mathrm{fine}}
\leq u_{\mathrm{crs}}\leq 0.01,
$$

together with Assumptions 3.1, possibly excluding condition (3.5).

If $\Theta$ is an $\varepsilon$-embedding for $\widehat Q$ and $\widehat W$, with

$$
\varepsilon\leq \frac12,
$$

and if

$$
\Delta^{(p)}\leq 0.1,
\qquad
\widetilde{\Delta}^{(p)}\leq 0.1,
$$

then

$$
(1+\varepsilon)^{-1/2}
\left(1-\Delta^{(p)}-0.1u_{\mathrm{crs}}\right)
\leq \sigma_{\min}(\widehat Q)
$$

and

$$
\sigma_{\max}(\widehat Q)
\leq
(1-\varepsilon)^{-1/2}
\left(1+\Delta^{(p)}+0.1u_{\mathrm{crs}}\right).
$$

The theorem also gives the factorization-residual bound

$$
\|\widehat W-\widehat Q\widehat R\|_F
\leq
4u_{\mathrm{crs}}m^{3/2}\|\widehat W\|_F.
$$

The paper's Remark 3.3 additionally states

$$
\|\Theta(\widehat W-\widehat Q\widehat R)\|_F
\leq
5u_{\mathrm{crs}}m^{3/2}\|\widehat W\|_F.
$$

**Notation caution:** The original paper uses hats on computed quantities; preserve them in the final report. The statement above matches the theorem as printed on p. 12, with line breaks added.

## 4. Assumptions 3.1 — What they mean

Assumptions 3.1 describe bounds on the sketch and the floating-point errors in Algorithm 2.1. They are hypotheses used by the proof, not conclusions that automatically hold for every run.

### (3.2) Bounded sketch norm

$$
\|\Theta\|_F\leq \sqrt{1+\varepsilon}\sqrt n,
\qquad \varepsilon\leq \frac12.
$$

**Meaning:** The sketching matrix must not have an excessively large Frobenius norm. The paper notes that this condition holds with probability at least $1-\delta$ when $\Theta$ is a $(1/2,\delta/n,1)$ oblivious subspace embedding.

### (3.1), (3.3a) and (3.3b) — Error in the projection/subtraction step

The paper defines the rounding error in step 3 as

$$
\Delta Q'_{(i)}
=
\widehat Q'_{(i)}
-
\left(
\widehat W_{(i)}
-
\widehat Q_{(1:i-1)}
\widehat R_{(1:i-1,i)}
\right).
\tag{3.1}
$$

The two assumed bounds are

$$
|\Delta Q'_{(i)}|
\leq
1.02u_{\mathrm{crs}}
\left(
|\widehat W_{(i)}|
+
i\,m_p\,|\widehat Q_{(1:i-1)}|
|\widehat R_{(1:i-1,i)}|
\right),
\tag{3.3a}
$$

and

$$
\|\Theta\Delta Q'_{(i)}\|_F
\leq
1.02u_{\mathrm{crs}}\sqrt{1+\varepsilon}
\left\|
|\widehat W_{(i)}|
+
i\,m_p\,|\widehat Q_{(1:i-1)}|
|\widehat R_{(1:i-1,i)}|
\right\|_F.
\tag{3.3b}
$$

Here $|A|$ is the entrywise absolute value and $|\widehat Q_{(1:i-1)}||\widehat R_{(1:i-1,i)}|$ is the ordinary matrix product of the two absolute-value matrices. The factor $i\,m_p$ is the standard worst-case dot-product factor (cf. (1.4), $U=1.02u(|H|+m|Y'||Z'|)$): the inner dimension of $\widehat Q_{(1:i-1)}\widehat R_{(1:i-1,i)}$ is $(i-1)m_p$, plus one subtraction. The bounds hold for $1\le i\le p$. (The paper also writes "$1\le t\le m_p$", but $t$ appears in neither inequality — apparently a typo.)

The paper notes that (3.3a) is the standard worst-case rounding bound, while (3.3b) holds under the probabilistic rounding model with probability at least $1-4\delta$ if $\Theta$ is a $(1/8,\,m^{-1}\binom{n}{d}^{-1}\delta,\,d)$ oblivious subspace embedding with $d=\mathcal O(\log(m/\delta))$.

**Meaning:** The computed block after subtracting the contributions from earlier blocks differs from the exact arithmetic result by a controlled rounding error. The second bound controls the same error after sketching. The constants express a worst-case rounding-error model.

### (3.4a) and (3.4b) — Error in the block QR factorization

The computed block QR step is assumed to satisfy

$$
|\widehat Q'_{(i)}-\widehat Q_{(i)}\widehat R_{(i,i)}|
\leq
u_{\mathrm{fine}}m
\left(
|\widehat Q'_{(i)}|
+
|\widehat Q_{(i)}||\widehat R_{(i,i)}|
\right)
\leq
0.1u_{\mathrm{crs}}
|\widehat Q_{(i)}||\widehat R_{(i,i)}|,
\tag{3.4a}
$$

and

$$
\left\|
\Theta\left(
\widehat Q'_{(i)}-\widehat Q_{(i)}\widehat R_{(i,i)}
\right)
\right\|_F
\leq
u_{\mathrm{fine}}m\|\Theta\|_F
\left\|
|\widehat Q'_{(i)}|
+
|\widehat Q_{(i)}||\widehat R_{(i,i)}|
\right\|
\leq
0.1u_{\mathrm{crs}}
\|\widehat Q_{(i)}\|\,\|\widehat R_{(i,i)}\|.
\tag{3.4b}
$$

Note the norms in (3.4b): only $\|\Theta(\cdot)\|_F$ and $\|\Theta\|_F$ are Frobenius; the middle term and the final bound $\|\widehat Q_{(i)}\|\,\|\widehat R_{(i,i)}\|$ use the spectral norm (a product of norms, not the norm of the product).

**Meaning:** The QR factorization of each block must be sufficiently accurate both in the original space and after applying the sketch. The fine-precision errors are required to be small compared with the coarse-precision error scale.

### (3.5) — Embedding quality for the intermediate block

If $\Theta$ is an $\varepsilon'$-embedding for the intermediate block $\widehat Q'_{(i)}$, the paper assumes

$$
1-0.1u_{\mathrm{crs}}\operatorname{cond}(\widehat Q'_{(i)})
\leq
\sigma_{\min}(\Theta\widehat Q_{(i)})
\leq
\sigma_{\max}(\Theta\widehat Q_{(i)})
\leq
1+0.1u_{\mathrm{crs}}\operatorname{cond}(\widehat Q'_{(i)}),
\tag{3.5}
$$

and

$$
\|\widehat Q_{(i)}\|\leq(1-\varepsilon')^{-1/2}\|\Theta\widehat Q_{(i)}\|.
$$

Note that the singular values are those of the sketch of the **output** block $\Theta\widehat Q_{(i)}$ (no prime); the condition number on the bounds is that of the **input** block $\widehat Q'_{(i)}$.

**Meaning:** (3.5) is a requirement on the inter-block QR routine in steps 4–5: whenever $\Theta$ embeds $\widehat Q'_{(i)}$, the routine must return a block whose sketch is nearly orthonormal, with a loss that grows only like $u_{\mathrm{crs}}\operatorname{cond}(\widehat Q'_{(i)})$. The paper states that the routines of Section 2.2 satisfy (3.4)–(3.5), e.g. single-vector RGS via (2.5).

**Why "possibly excluding (3.5)":** Theorem 3.2 does not need (3.5) at all — it works from the measured $\Delta^{(p)}$ and $\widetilde\Delta^{(p)}$, and its proof never invokes (3.5). Condition (3.5) matters for the a priori results (Theorem 3.4, Proposition 3.5). Separately, Section 3.1 notes that the implicit RCholeskyQR of Section 2.2.3 does not satisfy (3.5) directly, and handles it by a perturbation argument that reduces it to regular RCholeskyQR. These are two different points; don't merge them.

## 5. Proof of Theorem 3.2 — Detailed walkthrough of the paper's proof

The paper's proof is in Section 6. The reasoning has two main parts: first bound the singular values of $\widehat Q$; then bound the factorization residual.

### Part A: Bound the singular values of the computed basis

**Step 1 — Relate the computed sketch to the exact sketch of the computed basis.**

From the rounding assumptions for the sketching step, the paper obtains

$$
\|\Theta\widehat Q-\widehat S\|_F
\leq
1.02\,u_{\mathrm{fine}}\,n\,\|\Theta\|_F\|\widehat Q\|_F
\leq
1.02\sqrt{1+\varepsilon}\,
u_{\mathrm{fine}}n^{3/2}\|\widehat Q\|_F
\leq
0.02u_{\mathrm{crs}}\|\widehat Q\|
=:F_1.
\tag{6.1}
$$

The first inequality is the worst-case product bound (1.4) followed by $\|\Theta E\|_F\le\|\Theta\|_F\|E\|_F$; the second uses (3.2). The last one switches from $\|\widehat Q\|_F$ to the **spectral** norm $\|\widehat Q\|$ via $\|\widehat Q\|_F\le\sqrt m\,\|\widehat Q\|$; this $\sqrt m$ is exactly what the $m^{1/2}$ in the hypothesis $100m^{1/2}n^{3/2}u_{\mathrm{fine}}\le u_{\mathrm{crs}}$ absorbs ($1.02\sqrt{1.5}/100\approx0.0125\le0.02$).

**Why this matters:** The stored sketch $\widehat S$ is not exactly $\Theta\widehat Q$ because the computation is in floating-point arithmetic. Equation (6.1) bounds that discrepancy by $F_1$.

**Step 2 — Use a singular-value perturbation bound.**

If two matrices differ by a small amount in norm, their extreme singular values differ by no more than that amount. Applying this fact to $\Theta\widehat Q$ and $\widehat S$ gives

$$
\sigma_{\min}(\widehat S)-F_1
\leq \sigma_{\min}(\Theta\widehat Q)
\leq \sigma_{\max}(\Theta\widehat Q)
\leq \sigma_{\max}(\widehat S)+F_1.
$$

The proof then uses the definition

$$
\Delta^{(p)}=\|I-\widehat S^T\widehat S\|_F.
$$

A small $\Delta^{(p)}$ means that $\widehat S^T\widehat S$ is close to $I$. Therefore the singular values of $\widehat S$ are close to $1$; in particular,

$$
1-\Delta^{(p)}
\leq
\sqrt{1-\Delta^{(p)}}
\leq \sigma_{\min}(\widehat S)
\leq \sigma_{\max}(\widehat S)
\leq \sqrt{1+\Delta^{(p)}}
\leq 1+\Delta^{(p)}.
$$

With $\Delta^{(p)}\le0.1$, $\varepsilon\le1/2$ and the embedding bound of Corollary 1.4 (the paper cites "Corollary 1.3" here, but the singular-value relation used is Corollary 1.4), the proof obtains the **spectral-norm** bound $\|\widehat Q\|\leq1.57$ (roughly $\sqrt2\cdot1.1$). It is not a Frobenius bound: $\|\widehat Q\|_F\approx\sqrt m$.

**Step 3 — Transfer the bounds from sketch space to the original space.**

Because $\Theta$ is an $\varepsilon$-embedding for the relevant computed basis, it approximately preserves the lengths of vectors in that basis's column space. In singular-value terms, the embedding relates the singular values of $\widehat Q$ to those of $\Theta\widehat Q$, with distortion factors involving $1\pm\varepsilon$.

Combining Corollary 1.4 with Step 2, the proof obtains

$$
(1+\varepsilon)^{-1/2}(1-\Delta^{(p)}-F_1)
\leq\sigma_{\min}(\widehat Q)
\leq\sigma_{\max}(\widehat Q)
\leq(1-\varepsilon)^{-1/2}(1+\Delta^{(p)}+F_1).
$$

Since $F_1=0.02u_{\mathrm{crs}}\|\widehat Q\|\le0.02\cdot1.57\,u_{\mathrm{crs}}\approx0.031u_{\mathrm{crs}}\le0.1u_{\mathrm{crs}}$, replacing $F_1$ by $0.1u_{\mathrm{crs}}$ gives the statement of the theorem:

$$
(1+\varepsilon)^{-1/2}
\left(1-\Delta^{(p)}-0.1u_{\mathrm{crs}}\right)
\leq \sigma_{\min}(\widehat Q)
$$

and

$$
\sigma_{\max}(\widehat Q)
\leq
(1-\varepsilon)^{-1/2}
\left(1+\Delta^{(p)}+0.1u_{\mathrm{crs}}\right).
$$

This proves the theorem's singular-value statement. In plain language, the computed basis cannot be too far from having well-scaled, independent columns if the sketch is a good embedding, the sketch-orthogonality indicator is small, and the rounding errors are controlled.

### Part B: Bound the original-space factorization residual

**Step 4 — Start with the one-block case.**

For $p=1$, there are no previously processed blocks. The result follows directly from the assumed rounding-error bound for the block QR step.

**Step 5 — Bound the norm of the computed $R$ factor.**

For $p\geq2$, the proof uses the lower bound on $\sigma_{\min}(\widehat S)$ and the sketched factorization residual:

$$
\|\widehat R\|_F
\leq
\sigma_{\min}(\widehat S)^{-1}
\|\widehat S\widehat R\|_F.
$$

Since

$$
\|\widehat S\widehat R\|_F
\leq
\|\widehat P\|_F+
\|\widehat P-\widehat S\widehat R\|_F
=
(1+\widetilde{\Delta}^{(p)})\|\widehat P\|_F,
$$

the paper's chain is

$$
\|\widehat R\|_F
\leq\sigma_{\min}(\widehat S)^{-1}\|\widehat S\widehat R\|_F
\leq 1.12\|\widehat S\widehat R\|_F
\leq
1.12(1+\widetilde{\Delta}^{(p)})\|\widehat P\|_F
\leq 1.4(1+\widetilde{\Delta}^{(p)})\|\widehat W\|_F
\leq 1.6\|\widehat W\|_F,
\tag{6.2}
$$

using $\sigma_{\min}(\widehat S)\ge\sqrt{0.9}$ and $\|\widehat P\|_F\le1.02\|\Theta\widehat W\|_F\le1.25\|\widehat W\|_F$.

The constants here are the paper's simplified bounds under its assumptions; they are not universal constants for arbitrary matrices or sketches.

**Step 6 — Express the factorization error block by block.**

For each block $i$, the proof writes

$$
\widehat W_{(i)}
=
\widehat Q_{(1:i)}
\widehat R_{(1:i,i)}
+
\Delta W_{(i)},
$$

where $\Delta W_{(i)}$ collects the rounding errors from the projection/subtraction and block QR computations.

Taking Frobenius norms of (3.3a) and (3.4a) gives, for each block,

```math
\begin{aligned}
\|\Delta W_{(i)}\|_F
&\le\|\widehat Q'_{(i)}-\widehat Q_{(i)}\widehat R_{(i,i)}\|_F
+1.02u_{\mathrm{crs}}\big(\|\widehat W_{(i)}\|_F+i\,m_p\|\widehat Q_{(1:i-1)}\|_F\|\widehat R_{(1:i-1,i)}\|_F\big)\\
&\le 0.1u_{\mathrm{crs}}\|\widehat Q_{(i)}\|_F\|\widehat R_{(i,i)}\|_F
+1.02u_{\mathrm{crs}}\big(\|\widehat W_{(i)}\|_F+1.57\,i^{3/2}m_p^{3/2}\|\widehat R_{(1:i-1,i)}\|_F\big)\\
&\le 1.02u_{\mathrm{crs}}\big(1.26\,m_p^{1/2}\|\widehat W\|_F+1.57\,i^{3/2}m_p^{3/2}\|\widehat R_{(1:i-1,i)}\|_F\big),
\end{aligned}
```

where $\|\widehat Q_{(1:i-1)}\|_F\le1.57\sqrt{(i-1)m_p}$ turns $i\,m_p$ into $1.57\,i^{3/2}m_p^{3/2}$, and (6.2) bounds $\|\widehat R_{(i,i)}\|_F$. Squaring, summing over $i$, and using $\sum_i i^3m_p^3\le(pm_p)^3=m^3$ and $1.57\|\widehat R\|_F\le2.52\|\widehat W\|_F$:

$$
\|\Delta W\|_F^2\le1.02^2u_{\mathrm{crs}}^2\,2\big(1.26^2m+2.52^2m^3\big)\|\widehat W\|_F^2\le\big(4u_{\mathrm{crs}}m^{3/2}\|\widehat W\|_F\big)^2.
$$

So the $m^{3/2}$ in the final bound comes directly from the $i\,m_p$ factor in (3.3a) — another reason that factor must be in your statement of the assumption.

The resulting estimate in the paper is

$$
\|\widehat W-\widehat Q\widehat R\|_F
\leq
4u_{\mathrm{crs}}m^{3/2}\|\widehat W\|_F.
$$

This proves the second main conclusion of Theorem 3.2.

**Step 7 — Remark 3.3 (sketched residual).**

Repeating the blockwise argument with the sketched bounds (3.3b) and (3.4b) in place of (3.3a) and (3.4a) (constants 1.51 and 2 replace 1.26 and 1.57) gives the additional estimate

$$
\|\Theta(\widehat W-\widehat Q\widehat R)\|_F
\leq
5u_{\mathrm{crs}}m^{3/2}\|\widehat W\|_F.
$$

This is the extra statement in Remark 3.3, rather than one of the two principal conclusions printed in Theorem 3.2.

## 6. Intuition in simple words

The proof follows this chain:

1. **The sketch is computed accurately enough:** $\widehat S$ is close to $\Theta\widehat Q$.
2. **The sketch is nearly orthonormal:** small $\Delta^{(p)}$ means $\widehat S^T\widehat S$ is close to $I$.
3. **The sketch preserves geometry:** the embedding assumption lets the proof translate sketch-space information back to the original space.
4. **The sketched factorization is accurate:** small $\widetilde{\Delta}^{(p)}$ means $\widehat S\widehat R$ reproduces $\widehat P$.
5. **Rounding errors are controlled:** Assumptions 3.1 bound the errors accumulated by the algorithm's steps.
6. **Conclusion:** the computed basis has controlled extreme singular values, and $\widehat W\approx\widehat Q\widehat R$ with the stated residual bound.

## 7. What the theorem guarantees — and what it does not

### It guarantees, under all its hypotheses

- Bounds on the smallest and largest singular values of the computed basis.
- A bound on the original-space factorization residual.
- The additional sketched residual bound in Remark 3.3.
- An a posteriori stability certificate based on quantities computed from the sketches and $R$.

### It does not guarantee unconditionally

- It does **not** say that every RBGS run is stable regardless of sketch quality or rounding errors.
- It does **not** say that small indicators alone are enough if the embedding and rounding assumptions fail.
- It does **not** automatically prove exact orthogonality in the original Euclidean inner product; the conclusion is a bound, not exact equality $Q^TQ=I$.
- It does **not** provide the same kind of a priori bound for every input without additional assumptions. The paper's Theorem 3.4 adds stronger conditions to predict the indicators in advance.
- It does **not** make the algorithm immune to a poor sketch, an inappropriate precision configuration, or implementation errors.

## 8. Why the theorem is useful

The two indicators are relatively inexpensive to compute from the sketches and $R$. They let a numerical solver check whether the computed factorization appears reliable without repeatedly performing expensive calculations on the full high-dimensional matrix. In applications such as Krylov methods, this information can help decide whether re-orthogonalization or a restart is needed.

The paper also emphasizes that this a posteriori certification does not require the same assumptions on the condition number of $W$, the least-squares solver in step 2, and the stability of inter-block orthogonalization that are needed for the a priori result.



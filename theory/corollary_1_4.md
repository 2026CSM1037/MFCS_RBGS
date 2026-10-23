# Corollary 1.4: Singular-Value Bounds from an $\varepsilon$-Embedding

**Source:** Balabanov and Grigori, *Randomized Block Gram–Schmidt Process for the Solution of Linear Systems and Eigenvalue Problems*, Sec. 1.3, Corollary 1.4 (p. 4). The paper only refers to [4] for the proof, so the full argument is written out below.

## Statement

Let $V\in\mathbb{R}^{n\times m}$ and let $\Theta\in\mathbb{R}^{k\times n}$ be an $\varepsilon$-embedding for $V$ (that is, for $\operatorname{range}(V)$), with $0\le\varepsilon<1$. By Definition 1.1 this means

$$
\left|\langle x,y\rangle-\langle\Theta x,\Theta y\rangle\right|\le\varepsilon\,\lVert x\rVert\,\lVert y\rVert
\qquad\text{for all } x,y\in\operatorname{range}(V).
$$

Then the singular values of $V$ satisfy

$$
(1+\varepsilon)^{-1/2}\,\sigma_{\min}(\Theta V)\;\le\;\sigma_{\min}(V)\;\le\;\sigma_{\max}(V)\;\le\;(1-\varepsilon)^{-1/2}\,\sigma_{\max}(\Theta V).
$$

## Proof

Throughout, $\sigma_{\min}(A)=\min_{\lVert z\rVert=1}\lVert Az\rVert$ and $\sigma_{\max}(A)=\max_{\lVert z\rVert=1}\lVert Az\rVert$ for a tall matrix $A$. These are the standard variational descriptions of the extreme singular values, and the whole proof rests on them.

**Step 1: the embedding preserves norms.** Take $y=x$ in the definition. Since $\langle x,x\rangle=\lVert x\rVert^2$ and $\langle\Theta x,\Theta x\rangle=\lVert\Theta x\rVert^2$, we get $\left|\lVert x\rVert^2-\lVert\Theta x\rVert^2\right|\le\varepsilon\lVert x\rVert^2$. Rearranging gives

$$
(1-\varepsilon)\lVert x\rVert^2\le\lVert\Theta x\rVert^2\le(1+\varepsilon)\lVert x\rVert^2 .
$$

All terms are nonnegative because $\varepsilon<1$, so taking square roots is allowed.

**Step 2: pass to the matrix $V$.** For any $z\in\mathbb{R}^m$ the vector $Vz$ lies in $\operatorname{range}(V)$, so Step 1 applies to $x=Vz$:

$$
\sqrt{1-\varepsilon}\,\lVert Vz\rVert\;\le\;\lVert\Theta Vz\rVert\;\le\;\sqrt{1+\varepsilon}\,\lVert Vz\rVert
\qquad\text{for every } z\in\mathbb{R}^m. \tag{1}
$$

This is the only place where the embedding hypothesis enters.

**Step 3: the lower bound on $\sigma_{\min}(V)$.** Choose a unit vector $z_\ast$ at which $V$ is shortest, so that $\lVert Vz_\ast\rVert=\sigma_{\min}(V)$. Since $\sigma_{\min}(\Theta V)$ is the smallest value of $\lVert\Theta Vz\rVert$ over unit vectors, it cannot exceed the value at this particular $z_\ast$. Using the right-hand inequality of (1),

$$
\sigma_{\min}(\Theta V)\le\lVert\Theta Vz_\ast\rVert\le\sqrt{1+\varepsilon}\,\lVert Vz_\ast\rVert=\sqrt{1+\varepsilon}\;\sigma_{\min}(V).
$$

Dividing by $\sqrt{1+\varepsilon}$ gives $(1+\varepsilon)^{-1/2}\sigma_{\min}(\Theta V)\le\sigma_{\min}(V)$.

**Step 4: the upper bound on $\sigma_{\max}(V)$.** Now choose a unit vector $z^\ast$ at which $V$ is longest, so that $\lVert Vz^\ast\rVert=\sigma_{\max}(V)$. Since $\sigma_{\max}(\Theta V)$ is the largest value of $\lVert\Theta Vz\rVert$ over unit vectors, it is at least the value at $z^\ast$. Using the left-hand inequality of (1),

$$
\sigma_{\max}(\Theta V)\ge\lVert\Theta Vz^\ast\rVert\ge\sqrt{1-\varepsilon}\,\lVert Vz^\ast\rVert=\sqrt{1-\varepsilon}\;\sigma_{\max}(V).
$$

Dividing by $\sqrt{1-\varepsilon}>0$ gives $\sigma_{\max}(V)\le(1-\varepsilon)^{-1/2}\sigma_{\max}(\Theta V)$.

**Step 5: the middle inequality.** $\sigma_{\min}(V)\le\sigma_{\max}(V)$ holds for every matrix. Combining Steps 3 and 4 with this gives the claimed chain. $\qquad\blacksquare$

## Remarks

1. **Why the proof uses the "other" direction of (1).** Taking the minimum of the left side of (1) over unit $z$ gives the different inequality $\sigma_{\min}(\Theta V)\ge\sqrt{1-\varepsilon}\,\sigma_{\min}(V)$, and similarly $\sigma_{\max}(\Theta V)\le\sqrt{1+\varepsilon}\,\sigma_{\max}(V)$. Both are true, but they bound the sketch in terms of $V$. The corollary needs the reverse, because $\Theta V$ is the small matrix we can compute and the aim is to bound $V$ itself. This is why the proof evaluates (1) at the specific vectors $z_\ast$ and $z^\ast$ rather than optimizing over $z$.

2. **Rank is preserved.** Since $\varepsilon<1$, (1) shows that $\Theta Vz=0$ only if $Vz=0$.

3. **Consequence for the condition number.** Combining the two outer inequalities,
$$
\operatorname{cond}(V)\le\sqrt{\frac{1+\varepsilon}{1-\varepsilon}}\;\operatorname{cond}(\Theta V).
$$
If $\Theta V$ is nearly orthonormal, then $\operatorname{cond}(V)=1+O(\varepsilon)$. This is the fact quoted in Sec. 4.4 (p. 16) to justify that $\lVert H_{(p,1:p-1)}y\rVert$ is a reliable residual estimate in randomized Rayleigh–Ritz.

## Where the corollary is used

In the proof of Theorem 3.2 (p. 21), the algorithm controls the singular values of the small sketch $\hat S\approx\Theta\hat Q$ through the quantity $\Delta^{(p)}=\lVert I-\hat S^{\top}\hat S\rVert_F$. Corollary 1.4, applied with $V=\hat Q$, converts these into the bounds
$(1+\varepsilon)^{-1/2}(1-\Delta^{(p)}-F_1)\le\sigma_{\min}(\hat Q)\le\sigma_{\max}(\hat Q)\le(1-\varepsilon)^{-1/2}(1+\Delta^{(p)}+F_1)$
for the computed basis $\hat Q$ itself. The same step is repeated for $\hat W\hat R^{-1}$ in the proof of Proposition 3.5 (p. 25).

# Theorem 3.4 and Proposition 3.5

**Source:** Balabanov and Grigori, *Randomized Block Gram–Schmidt Process for the Solution of Linear Systems and Eigenvalue Problems*. Theorem 3.4 is on p. 12 and Proposition 3.5 on p. 13 (arXiv v2 pagination). Their proofs are in Sec. 6 (pp. 22–25) and are not reproduced here.

**Notation.** $\hat W$, $\hat Q$, $\hat R$ are the computed quantities, $m$ is the total number of columns and $n$ the row dimension. The stability coefficients are
$\Delta^{(p)}=\lVert I-\hat S^{\top}\hat S\rVert_F$ and $\tilde\Delta^{(p)}=\lVert\hat P-\hat S\hat R\rVert_F/\lVert\hat P\rVert_F$, where $\hat S$ and $\hat P$ are the computed sketches of $Q$ and $W$ (Sec. 3.2.1). The unit roundoffs are $u_{\mathrm{crs}}$ (coarse, step 3 of Algorithm 2.1) and $u_{\mathrm{fine}}$ (fine, all other steps).

## Theorem 3.4 (a priori stability of RBGS)

Consider Algorithm 2.1 with a backward-stable solver for step 2 (for example one based on Richardson iterations) that satisfies Assumptions 2.2. Under Assumptions 3.1, assume that $\Theta$ is an $\varepsilon$-embedding for $\hat Q_{(1:p-1)}$ and for $\hat W$, with $\varepsilon\le1/2$. If

$$
u_{\mathrm{crs}}\le10^{-3}\operatorname{cond}(\hat W)^{-1}m^{-2}\qquad\text{and}\qquad u_{\mathrm{fine}}\le10^{-2}m^{-1/2}n^{-3/2}u_{\mathrm{crs}},
$$

then $\Delta^{(p)}$ and $\tilde\Delta^{(p)}$ are bounded by

$$
\tilde\Delta^{(p)}\le4.2\,u_{\mathrm{crs}}\,m^{3/2}\,\lVert\hat W\rVert_F/\lVert\hat P\rVert_F\le6\,u_{\mathrm{crs}}\,m^{3/2}, \tag{3.7}
$$

$$
\Delta^{(p)}\le20\,u_{\mathrm{crs}}\,m^{2}\operatorname{cond}(\hat W). \tag{3.8}
$$

**Main idea and significance.** Theorem 3.2 is an a posteriori result: it says RBGS is stable whenever the computed coefficients $\Delta^{(p)}$ and $\tilde\Delta^{(p)}$ are at most $0.1$. Theorem 3.4 shows that these coefficients are small automatically, without having to check them, provided the least-squares solver in step 2 is backward stable, the embedding property holds, and the coarse roundoff satisfies $u_{\mathrm{crs}}\lesssim\operatorname{cond}(\hat W)^{-1}m^{-2}$, which means $\hat W$ is numerically full rank. With these conditions, (3.8) gives $\Delta^{(p)}\le0.02$ and (3.7) gives $\tilde\Delta^{(p)}\le6u_{\mathrm{crs}}m^{3/2}$, so Theorem 3.2 applies and $\hat Q$ is well conditioned with $\hat W\approx\hat Q\hat R$. The proof goes by induction on the number of blocks. The practical significance, as the authors emphasize, is that RBGS is stable unless the input is numerically rank deficient, which is the same behavior as other stable methods such as MGS, CGS2 and BCGS2, and that none of the conditions on the working precision involves the large dimension $n$. This makes it possible to run the dominant operations in low precision on very large problems.

## Proposition 3.5 (the embedding passes from $\hat W$ to $\hat Q$)

Consider Algorithm 2.1 with a backward-stable solver satisfying Assumptions 2.2, and suppose

$$
u_{\mathrm{crs}}\le10^{-3}\operatorname{cond}(\hat W)^{-1}m^{-2}\qquad\text{and}\qquad u_{\mathrm{fine}}\le10^{-2}m^{-1/2}n^{-3/2}u_{\mathrm{crs}}.
$$

Under Assumptions 3.1, if $\Theta$ is an $\varepsilon$-embedding for $\hat W$ with $\varepsilon\le1/4$, then $\Theta$ is an $\varepsilon'$-embedding for $\hat Q$, with

$$
\varepsilon'=2\varepsilon+180\,u_{\mathrm{crs}}\,m^{2}\operatorname{cond}(\hat W).
$$

**Main idea and significance.** The stability analysis of Sec. 3, including Theorems 3.2 and 3.4, needs $\Theta$ to be an $\varepsilon$-embedding for the computed basis $\hat Q$, but $\hat Q$ is produced by the algorithm from $\Theta$ and so is not independent of it. In contrast, the input $\hat W$ is a fixed matrix chosen independently of $\Theta$, so an oblivious $\ell_2$-subspace embedding (Definition 1.2) embeds it with high probability. The proposition closes this gap by showing that the embedding property transfers from $\hat W$ to $\hat Q$ at the price of a mild deterioration. The distortion roughly doubles, and a rounding term proportional to $u_{\mathrm{crs}}m^{2}\operatorname{cond}(\hat W)$ is added, which is tiny under the stated assumptions on $u_{\mathrm{crs}}$. The proof is again an induction over blocks. It shows that $\hat Q$ is close to $\hat W\hat R^{-1}$, whose singular values are controlled by the embedding property of $\hat W$ and by the near-orthonormality of the sketch $\hat S$, and then bounds the gap between $\lVert\hat Qa\rVert^2$ and $\lVert\Theta\hat Qa\rVert^2$. The result is the block counterpart of [4, Proposition 3.6] for single-vector RGS. It holds only when $\hat W$ is independent of $\Theta$. When $\hat W$ depends on $\Theta$, as for Krylov bases, the paper only offers an a posteriori certification with a second sketch (p. 13).

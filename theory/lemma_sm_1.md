# Lemma SM.1: Sketched and Ordinary Projectors

**Source:** Supplementary material of Balabanov and Grigori, RBGS paper, "brief analysis of the accuracy of the randomized RR approximation" (pp. 30–33 of the arXiv v2 file); Lemma SM.1 is on p. 31. It supports the Rayleigh–Ritz discussion in Sec. 4.4 (pp. 15–16).

## The two projectors

Let $\mathcal{K}\subset\mathbb{R}^n$ be an approximation space with basis matrix $Q$ (it need not be a Krylov space). The ordinary projector sends $w$ to the closest point of $\mathcal{K}$ in the $\ell_2$ norm:

$$
\Pi_{\mathcal{K}}\,w=\arg\min_{v\in\mathcal{K}}\lVert w-v\rVert .
$$

The sketched projector does the same thing, but measures the distance after applying $\Theta$:

$$
\Pi^{\Theta}_{\mathcal{K}}\,w=\arg\min_{v\in\mathcal{K}}\lVert\Theta(w-v)\rVert=Q\,(\Theta Q)^{\dagger}\,\Theta\,w .
$$

Both are projectors onto $\mathcal{K}$. The difference is the orthogonality condition: $\Pi_{\mathcal{K}}$ makes the error orthogonal to $\mathcal{K}$ in the usual inner product, whereas $\Pi^{\Theta}_{\mathcal{K}}$ makes it orthogonal in the sketched one, $(\Theta Q)^{\top}\Theta\,(w-\Pi^{\Theta}_{\mathcal{K}}w)=0$. In general $\Pi^{\Theta}_{\mathcal{K}}$ is not symmetric, so it is an *oblique* projector in the ordinary geometry.

## Statement (Lemma SM.1)

Let $w\in\mathbb{R}^n$. If $\Theta$ is an $\varepsilon$-embedding for $V=\mathcal{K}+\operatorname{span}(w)$, then

$$
\lVert(I-\Pi_{\mathcal{K}})w\rVert\;\le\;\lVert(I-\Pi^{\Theta}_{\mathcal{K}})w\rVert\;\le\;\sqrt{\frac{1+\varepsilon}{1-\varepsilon}}\;\lVert(I-\Pi_{\mathcal{K}})w\rVert, \tag{SM.6a}
$$

$$
\frac{1}{1+\varepsilon}\Big(\lVert\Pi_{\mathcal{K}}w\rVert^{2}-2\varepsilon\lVert w\rVert^{2}\Big)\;\le\;\lVert\Pi^{\Theta}_{\mathcal{K}}w\rVert^{2}\;\le\;\frac{1}{1-\varepsilon}\Big(\lVert\Pi_{\mathcal{K}}w\rVert^{2}+2\varepsilon\lVert w\rVert^{2}\Big), \tag{SM.6b}
$$

and if $w\in\mathcal{K}$, then

$$
\Pi^{\Theta}_{\mathcal{K}}\,w=w. \tag{SM.6c}
$$

In words: for a vector $w$ and a subspace $\mathcal{K}$ that $\Theta$ embeds, the sketched projection error is never smaller than the true best error and is larger by at most the factor $\sqrt{(1+\varepsilon)/(1-\varepsilon)}$. The size of the sketched projection is close to the size of the ordinary one, up to $O(\varepsilon)\lVert w\rVert^2$. Vectors already in $\mathcal{K}$ are left unchanged.

## Why the inequalities hold

Write $\Pi=\Pi_{\mathcal{K}}$ and $\Pi^{\Theta}=\Pi^{\Theta}_{\mathcal{K}}$.

**(SM.6a).** The left inequality holds because $\Pi^{\Theta}w\in\mathcal{K}$ and $\Pi w$ is the closest point of $\mathcal{K}$ to $w$ in the $\ell_2$ norm. For the right one, the vector $(I-\Pi^{\Theta})w$ lies in $V$, so the embedding property applies:
$$
\lVert(I-\Pi^{\Theta})w\rVert^2\le\tfrac{1}{1-\varepsilon}\lVert\Theta(I-\Pi^{\Theta})w\rVert^2\le\tfrac{1}{1-\varepsilon}\lVert\Theta(I-\Pi)w\rVert^2\le\tfrac{1+\varepsilon}{1-\varepsilon}\lVert(I-\Pi)w\rVert^2 .
$$
The middle step is the definition of $\Pi^{\Theta}$: it minimizes the sketched distance, so it does at least as well as $\Pi w$ in that measure. The first and last steps use the two sides of the embedding inequality.

**(SM.6b).** Because $\Theta(I-\Pi^{\Theta})w$ is orthogonal to $\Theta\mathcal{K}$, a Pythagoras identity holds in sketch space: $\lVert\Theta\Pi^{\Theta}w\rVert^2=\lVert\Theta w\rVert^2-\lVert\Theta(I-\Pi^{\Theta})w\rVert^2$. For the upper bound, use $\lVert\Theta w\rVert^2\le(1+\varepsilon)\lVert w\rVert^2$ together with the lower half of (SM.6a) and the embedding inequality, then replace $\lVert(I-\Pi)w\rVert^2=\lVert w\rVert^2-\lVert\Pi w\rVert^2$. This gives $\lVert\Theta\Pi^{\Theta}w\rVert^2\le\lVert\Pi w\rVert^2+2\varepsilon\lVert w\rVert^2$. The lower bound is symmetric. Since $\Pi^{\Theta}w\in\mathcal{K}\subset V$, the embedding property converts between $\lVert\Theta\Pi^{\Theta}w\rVert^2$ and $\lVert\Pi^{\Theta}w\rVert^2$ at the cost of the factors $1/(1\pm\varepsilon)$.

**(SM.6c).** If $w\in\mathcal{K}$, then $v=w$ makes $\lVert\Theta(w-v)\rVert$ zero, so it is the minimizer.

## Why this matters for eigenvalue computations

In classical Rayleigh–Ritz, the Galerkin condition $\langle v,r(u,\mu u)\rangle=0$ for all $v\in\mathcal{K}$ is the same as saying $(\mu,u)$ is an eigenpair of the *approximate operator* $\Pi_{\mathcal{K}}A\Pi_{\mathcal{K}}$ (Sec. 4.4, p. 16, and Suppl. p. 30). The randomized method imposes the sketched Galerkin condition (4.9) instead, which is the statement that $(\mu,u)$ is an eigenpair of $\Pi^{\Theta}_{\mathcal{K}}A\,\Pi^{\Theta}_{\mathcal{K}}$. Unlike the sketched minimal-residual condition in GMRES, the accuracy of this approximation does not follow immediately from the embedding property (p. 16), so the paper adapts the projector-based analysis of Saad [2, Sec. 4.3] to the sketched operator. Lemma SM.1 is the dictionary that makes this possible: whenever the classical argument needs a fact about $\Pi_{\mathcal{K}}$, the lemma gives the corresponding fact about $\Pi^{\Theta}_{\mathcal{K}}$ up to factors $1+O(\varepsilon)$.

It is used in two places on pp. 31–32:

* **Theorem SM.2** bounds the sketched residuals of an exact eigenvector $x$ with respect to $A_m=\Pi^{\Theta}_{\mathcal{K}}A\Pi^{\Theta}_{\mathcal{K}}$ by $\gamma\lVert(I-\Pi^{\Theta}_{\mathcal{K}})x\rVert$ and $\sqrt{\lambda^2+\gamma^2}\,\lVert(I-\Pi^{\Theta}_{\mathcal{K}})x\rVert$. By (SM.6a), the factor $\lVert(I-\Pi^{\Theta}_{\mathcal{K}})x\rVert$ can be replaced by $(1+O(\varepsilon))\lVert(I-\Pi_{\mathcal{K}})x\rVert$. The bound then depends only on how well $\mathcal{K}$ approximates the true eigenvector, as in the classical theory. Taking $\Theta=I$ recovers Saad's Theorem 4.3 exactly.
* **Theorem SM.3** bounds the residual of a computed pair $(\mu,u)$ with respect to the exact operator: $\lVert(A-\mu I)u\rVert\le(1+O(\varepsilon))\lVert(I-\Pi_{\mathcal{K}})A\Pi_{\mathcal{K}}\rVert\,\lVert u\rVert$. Consequently, if $\mathcal{K}$ is close to an invariant subspace the computed pairs have small residuals, and Corollary SM.4 states that when $\mathcal{K}$ is exactly invariant every pair satisfying (SM.4) is a true eigenpair.

## Limits of the result

The supplement is a brief analysis, not a full theory. A small residual does not guarantee an accurate eigenpair if the eigenvalue is badly conditioned. For symmetric $A$ the classical method can use the Min-Max principle to avoid this, but $\Pi^{\Theta}_{\mathcal{K}}A\Pi^{\Theta}_{\mathcal{K}}$ can be non-symmetric even when $A$ is symmetric, so the same route is not obviously available (p. 33). Remark 4.2 (p. 16) notes that the classical Galerkin projection can still be recovered by adding a Cholesky QR step to RBGS-Arnoldi.

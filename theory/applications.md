# How RBGS Theory Connects to GMRES and Rayleigh–Ritz

**Source:** Balabanov and Grigori, *Randomized Block Gram–Schmidt Process for the Solution of Linear Systems and Eigenvalue Problems*, Sec. 4.1–4.4 (pp. 14–16), with the supplementary analysis on pp. 30–33. Notation follows the paper: $AX=B$, $\Theta\in\mathbb{R}^{k\times n}$ is the sketching matrix, and $\hat Q$, $\hat H$ are the computed Arnoldi basis and Hessenberg matrix.

## 1. Randomized GMRES is optimal up to a factor $\mathrm{cond}(\hat Q)$ (p. 14)

**Setup.** RBGS-Arnoldi (Algorithm 4.1) builds a Krylov basis that is orthonormal in the sketched inner product. Under the stability conditions of RBGS, the computed quantities satisfy the Arnoldi identity (4.1),

$$
(A+\Delta A) \hat Q_{(1:p-1)}=\hat Q_{(1:p)} \hat H,
$$

where $\lVert\Delta A\rVert$ is close to machine precision and $\mathrm{cond}(\hat Q)=O(1)$. GMRES then computes the $i$-th column of the solution as

$$
u_i=\hat Q_{(1:p-1)}\arg\min_{z}\lVert\hat H z-\hat R_{(1:p,1)}e_i\rVert \qquad \text{(4.2)}
$$

so the least-squares problem is solved in the small coordinates $z$. The question is how good $u_i$ is in terms of the actual residual $\lVert Av-b_i\rVert$, since $\hat Q$ is only $\Theta$-orthonormal and not $\ell_2$-orthonormal.

**The argument.** The first block of the QR factorization of $[B,\ AQ]$ gives $b_i=\hat Q_{(1:p)}\hat R_{(1:p,1)}e_i$ (ignoring the small rounding term in $\hat W=\hat Q\hat R$). Together with (4.1), for every coefficient vector $z$,

$$
(A+\Delta A)\hat Q_{(1:p-1)}z-b_i=\hat Q_{(1:p)}(\hat Hz-\hat R_{(1:p,1)}e_i).
$$

A residual that has been multiplied by $\hat Q$ can be compared to the coordinate residual from above and below:

$$
\sigma_{\min}(\hat Q) \lVert\hat Hz-\hat R_{(1:p,1)}e_i\rVert \le \lVert(A+\Delta A)\hat Q_{(1:p-1)}z-b_i\rVert \le \lVert\hat Q\rVert \lVert\hat Hz-\hat R_{(1:p,1)}e_i\rVert .
$$

Now let $z_i$ be the minimizer in (4.2) and let $z$ be arbitrary. Using the upper bound at $z_i$, then minimality of $z_i$, then the lower bound at $z$:

$$
\lVert(A+\Delta A)u_i-b_i\rVert\le\lVert\hat Q\rVert \lVert\hat Hz_i-\hat R e_i\rVert\le\lVert\hat Q\rVert \lVert\hat Hz-\hat Re_i\rVert\le\frac{\lVert\hat Q\rVert}{\sigma_{\min}(\hat Q)} \lVert(A+\Delta A)\hat Q_{(1:p-1)}z-b_i\rVert .
$$

Since $\hat Q_{(1:p-1)}z$ ranges over all of $Q_{p-1}=\mathrm{range}(\hat Q_{(1:p-1)})=\mathcal{K}_{p-1}(A+\Delta A,B)$, this is the paper's conclusion:

$$
\lVert(A+\Delta A)u_i-b_i\rVert\le\mathrm{cond}(\hat Q)\min_{v\in Q_{p-1}}\lVert(A+\Delta A)v-b_i\rVert .
$$

**Where the embedding enters.** The least-squares problem (4.2) minimizes a Euclidean norm in coefficient space. Because $\hat Q$ is $\Theta$-orthonormal, this is the same as minimizing the *sketched* residual $\lVert\Theta(Av-b_i)\rVert$ over the Krylov subspace (Remark 4.1, p. 15). The $\varepsilon$-embedding property makes $\hat Q$ well conditioned, via Corollary 1.4 and Theorem 3.2, so that $\mathrm{cond}(\hat Q)=O(1)$ and the factor above is of order one.

**What this does and does not say.** The result is quasi-optimality, not equality with classical GMRES. Two qualifications matter. First, the comparison is made for the slightly perturbed matrix $A+\Delta A$, not for $A$ itself. Second, it relies on $\Theta$ being an $\varepsilon$-embedding for the computed Krylov space. The authors state in Sec. 7 (p. 26) that this property is observed in their experiments but is still unproven for randomized Arnoldi. The test in Sec. 5.2 (pp. 19–20) shows residual convergence comparable to BCGS2 with a well-conditioned $Q$.

## 2. The sketched Galerkin condition in Rayleigh–Ritz (pp. 15–16)

**Classical Rayleigh–Ritz.** Given a Krylov basis $Q_{(1:p)}$ from Arnoldi, the extreme eigenpairs $(\lambda,x)$ of $A$ are approximated by $(\mu,u)=(\mu,Q_{(1:p-1)}y)$, where $(\mu,y)$ is an eigenpair of the small matrix $H_{(1:p-1,1:p-1)}$ (eq. 4.6). With an $\ell_2$-orthonormal $Q$ one has $H_{(1:p-1,1:p-1)}=Q_{(1:p-1)}^{\top}AQ_{(1:p-1)}$, which is equivalent to the Galerkin orthogonality condition (4.7),

$$
\langle v,\ r(u,\mu u)\rangle=0\quad\text{for all }v\in Q_{p-1},\qquad r(u,\mu u)=Au-\mu u .
$$

**Randomized version.** RBGS-Arnoldi produces a basis $Q$ that is orthonormal with respect to $\langle\Theta\cdot,\Theta\cdot\rangle$, so $(\Theta Q_{(1:p-1)})^{\top}\Theta Q_{(1:p-1)}=I$ and

$$
H_{(1:p-1,1:p-1)}=(\Theta Q_{(1:p-1)})^{\top} \Theta A Q_{(1:p-1)} .
$$

An eigenpair of this matrix therefore satisfies the **sketched Galerkin condition** (4.9),

$$
\langle\Theta v,\ \Theta r(u,\mu u)\rangle=0\quad\text{for all }v\in Q_{p-1}.
$$

This is the classical condition with the inner product and norm replaced by their sketched counterparts. Equivalently, if $\Pi_{Q_{p-1}}$ is the $\ell_2$-orthogonal projector onto $Q_{p-1}$, classical Rayleigh–Ritz finds eigenpairs of the approximate operator $\Pi_{Q_{p-1}}A\Pi_{Q_{p-1}}$. The randomized method finds eigenpairs of $\Pi^{\Theta}_{Q_{p-1}}A \Pi^{\Theta}_{Q_{p-1}}$, where $\Pi^{\Theta}_{Q_{p-1}}=Q_{(1:p-1)}(\Theta Q_{(1:p-1)})^{\dagger}\Theta$ is the projector that is orthogonal in the sketched inner product.

**Residual estimate and accuracy.** For any pair, (4.8) gives
$\sigma_{\min}(Q)\lVert H_{(p,1:p-1)}y\rVert\le\lVert r(\mu,u)\rVert\le\sigma_{\max}(Q)\lVert H_{(p,1:p-1)}y\rVert$.
Since the embedding property gives $\mathrm{cond}(Q)=1+O(\varepsilon)$, the cheap quantity $\lVert H_{(p,1:p-1)}y\rVert$ is a reliable estimate of the true residual. Accuracy of the sketched Galerkin projection, however, does not follow directly from the embedding property, unlike for GMRES (p. 16). The paper handles it by rewriting everything with projection operators, as above, which is where Lemma SM.1 and Theorems SM.2–SM.3 enter (see `lemma_sm_1.md`). If the classical projection is preferred, Remark 4.2 (p. 16) shows how to get it by adding one Cholesky QR step.

## 3. How the two applications fit together

Both methods use the same ingredient, the $\Theta$-orthonormal Krylov basis from RBGS-Arnoldi. They differ in the condition imposed on the basis. GMRES minimizes the (sketched) residual over the subspace, and this is controlled by $\mathrm{cond}(\hat Q)$. Rayleigh–Ritz imposes a sketched orthogonality condition on the residual, and this is analysed through the sketched projector. In both cases the heavy work happens in the $k$-dimensional sketch space, and the embedding and stability results of Secs. 1.3 and 3 are what justify translating the result back to the original problem.
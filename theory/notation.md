# Notation

This page summarizes the mathematical notation used in the paper *Randomized Block Gram–Schmidt Process for the Solution of Linear Systems and Eigenvalue Problems* by Balabanov and Grigori.

## 1. Core QR Factorization Notation

| Symbol | Meaning |
|---|---|
| $W$ | Original input matrix to be factorized. |
| $Q$ | Matrix whose columns form a basis for the column space of $W$. Its columns are intended to be orthonormal or very well-conditioned, depending on the algorithm and precision. |
| $R$ | Upper-triangular factor in the QR factorization. |
| $W = QR$ | QR factorization of the original matrix. |

In an ideal $\ell_2$-orthonormal basis,

$$
Q^TQ = I,
$$

where $I$ is the identity matrix.

In floating-point computations, rounding errors may cause the computed columns of $Q$ to deviate from exact orthonormality. The paper's RBGS algorithm initially constructs a basis that is orthonormal with respect to the sketched inner product. An optional Cholesky QR post-processing step can make the basis $\ell_2$-orthonormal up to machine precision.

## 2. Random Sketching Notation

| Symbol | Meaning |
|---|---|
| $\Theta$ | Random sketching matrix that maps vectors into a lower-dimensional representation. |
| $S$ | Sketch of the computed basis, typically $S = \Theta Q$. |
| $P$ | Sketch of the input matrix, typically $P = \Theta W$. |
| $k$ | Number of rows in the sketching matrix $\Theta$. |
| $n$ | Number of rows in the original matrix $W$. |
| $m$ | Number of columns in the original matrix $W$. |
| $p$ | Number of column blocks in the partition of $W$. |
| $m_p$ | Number of columns in each block: $W_{(i)}\in\mathbb R^{n\times m_p}$, $1\le i\le p$, with $m=m_p\,p$ (paper, Section 1.2). All blocks have the same width. |

If $W \in \mathbb{R}^{n \times m}$ and $\Theta \in \mathbb{R}^{k \times n}$, then

$$
P = \Theta W \in \mathbb{R}^{k \times m}.
$$

Similarly,

$$
S = \Theta Q \in \mathbb{R}^{k \times m}.
$$

When $k \ll n$, the sketch has substantially fewer rows than the original matrix. This allows certain calculations to be performed in a lower-dimensional space.

## 3. Precision and Floating-Point Notation

| Symbol | Meaning |
|---|---|
| $\widehat{X}$ | A quantity or expression computed in finite-precision arithmetic; the paper writes $\widehat X$ or $\mathrm{fl}(X)$ (Section 1.1). |
| $u_{\mathrm{crs}}$ | Unit roundoff associated with the coarse precision used for selected computations. |
| $u_{\mathrm{fine}}$ | Unit roundoff associated with the fine precision used for selected computations. |

Unit roundoff measures the scale of rounding errors in floating-point arithmetic. Smaller unit roundoff generally corresponds to higher precision.

In Algorithm 2.1, the expensive, high-dimensional projection operation (Step 3) uses coarse precision. The other listed operations use fine precision, with

$$
u_{\mathrm{fine}} \leq u_{\mathrm{crs}}.
$$

This mixed-precision approach aims to reduce computational cost while maintaining stability.

## 4. Embedding and Error Notation

| Symbol | Meaning |
|---|---|
| $\varepsilon$ | Embedding accuracy parameter that controls the allowed distortion in relevant geometric quantities. |
| $\varepsilon$-embedding | A mapping that approximately preserves relevant norms or inner products within a specified error bound for vectors in the subspace of interest. |

**Definition 1.1 (paper).** For $\varepsilon<1$, $\Theta$ is an $\varepsilon$-embedding for a subspace $V$ if

$$
\left|\langle x,y\rangle-\langle\Theta x,\Theta y\rangle\right|
\leq \varepsilon \|x\|\,\|y\|
\qquad\forall x,y\in V.
\tag{1.2}
$$

**Definition 1.2 (paper).** A random $\Theta$ is an $(\varepsilon,\delta,d)$ oblivious $\ell_2$-subspace embedding if, for any fixed $d$-dimensional $V\subset\mathbb R^n$, $\mathbb P(\Theta\text{ is an }\varepsilon\text{-embedding for }V)\ge1-\delta$.

**Corollary 1.4 (paper).** If $\Theta$ is an $\varepsilon$-embedding for $V$, then $(1+\varepsilon)^{-1/2}\sigma_{\min}(\Theta V)\le\sigma_{\min}(V)\le\sigma_{\max}(V)\le(1-\varepsilon)^{-1/2}\sigma_{\max}(\Theta V)$. This is the step that transfers sketch-space bounds to the original space in Theorem 3.2.

## 4a. Norms and entrywise notation (paper, Section 1.1)

| Symbol | Meaning |
|---|---|
| $\Vert \cdot\Vert$ | $\ell_2$-norm for vectors; **spectral norm** $\sigma_{\max}(\cdot)$ for matrices. |
| $\Vert \cdot\Vert _F$ | Frobenius norm. Do not interchange with $\Vert \cdot\Vert$: e.g. $\Vert \widehat Q\Vert \le1.57$ but $\Vert \widehat Q\Vert _F\approx\sqrt m$. |
| $\vert X\vert$ | Entrywise absolute value. |
| $X\le Y$ | Entrywise inequality. |
| $\sigma_{\min},\sigma_{\max},\operatorname{cond}$ | Extreme singular values and condition number. |
| $X^\dagger$ | Moore–Penrose pseudoinverse. |
| $\pm$ | "Plus-minus" shorthand: LHS bounded above with $+$ and below with $-$. |

## 5. Block Notation

| Symbol | Meaning |
|---|---|
| $W_{(i)}$ | The $i$-th block of columns of the input matrix. |
| $Q_{(i)}$ | The computed basis block corresponding to $W_{(i)}$. |
| $Q'_{(i)}$ | Intermediate block obtained after subtracting components in previously computed basis directions. |
| $R_{(i,i)}$ | Diagonal block of the block upper-triangular factor $R$. |
| $R_{(1:i-1,i)}$ | Block of projection coefficients connecting previously processed blocks to the current block. |
| $S_{(i)}$ | Sketch of the computed basis block, $S_{(i)} = \Theta Q_{(i)}$. |
| $P_{(i)}$ | Sketch of the current input block, $P_{(i)} = \Theta W_{(i)}$. |

The block notation allows several columns to be processed together rather than one column at a time.

**Index convention.** Block indices are written as **subscripts**, as in the paper: $W_{(i)}$, $Q_{(1:i-1)}$, $R_{(1:i-1,i)}$, $R_{(i,i)}$, $S_{(i)}$, $P_{(i)}$. Superscripts are reserved for iteration counts such as $\Pi^{(i-1)}$ and $\Delta^{(p)}$.

**First block.** For $i=1$, $R_{(1:0,1)}$ is $0\times m_p$ and $Q_{(1:0)}$ is $n\times0$, so $Q'_{(1)}=W_{(1)}$.

## 6. Important Relationships

The following equations summarize the main relationships used in the randomized block Gram–Schmidt algorithm.

### QR factorization

$$
W = QR.
$$

Here, $Q$ spans the same column space as $W$, and $R$ is upper triangular. In finite precision, the computed factorization may include rounding errors.

### Sketch of the input matrix

$$
P = \Theta W.
$$

### Sketch of the computed basis

$$
S = \Theta Q.
$$

### Removal of previously computed directions

$$
Q'_{(i)}
=
W_{(i)}
-
Q_{(1:i-1)}R_{(1:i-1,i)}.
$$

This step removes the components of the current input block that lie in the directions represented by previously computed basis blocks.

### Orthonormalization of the current block

$$
Q'_{(i)} = Q_{(i)}R_{(i,i)}.
$$

This is the intended block QR relationship in Algorithm 2.1. In floating-point computation, rounding errors may make the relationship approximate.

The RBGS algorithm performs this orthonormalization with respect to the sketched inner product

$$
\langle x,y\rangle_{\Theta}
=
(\Theta x)^T(\Theta y).
$$

Therefore, sketch-orthonormality does not automatically imply exact $\ell_2$-orthonormality in the original space.

## 7. Important Note

This page is a notation reference based on the paper's Sections 1.1–1.3 and Algorithm 2.1 (checked against arXiv:2111.14641v2). When using later sections, keep in mind:

- The assumptions and bounds involving $u_{\mathrm{crs}}$ and $u_{\mathrm{fine}}$.
- Whether a stated orthogonality property refers to the sketched inner product or the original $\ell_2$ inner product.

These distinctions are important for understanding the algorithm's numerical stability and its use of randomized sketching.
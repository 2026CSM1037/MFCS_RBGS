
# Notation

This page summarizes the mathematical notation used in the paper *Randomized Block Gram–Schmidt Process for the Solution of Linear Systems and Eigenvalue Problems* by Balabanov and Grigori.

## 1. Core QR Factorization Notation

| Symbol | Meaning |
|---|---|
| \(W\) | Original input matrix to be factorized. |
| \(Q\) | Matrix whose columns form an orthonormal basis, or are approximately orthonormal in floating-point computation. |
| \(R\) | Upper-triangular factor in the QR factorization. |
| \(W = QR\) | QR factorization of the original matrix. |

For a matrix \(Q\) with orthonormal columns,

\[
Q^TQ = I,
\]

where \(I\) is the identity matrix.

## 2. Random Sketching Notation

| Symbol | Meaning |
|---|---|
| \(\Theta\) | Random sketching matrix that maps vectors into a lower-dimensional representation. |
| \(S\) | Sketch of the computed basis, typically \(S = \Theta Q\). |
| \(P\) | Sketch of the input matrix, typically \(P = \Theta W\). |
| \(k\) | Number of rows in the sketching matrix \(\Theta\). |
| \(n\) | Number of rows in the original matrix \(W\). |
| \(m\) | Number of columns in the original matrix \(W\). |

If \(W \in \mathbb{R}^{n \times m}\) and \(\Theta \in \mathbb{R}^{k \times n}\), then

\[
P = \Theta W \in \mathbb{R}^{k \times m}.
\]

When \(k \ll n\), the sketch has substantially fewer rows than the original matrix.

## 3. Precision and Floating-Point Notation

| Symbol | Meaning |
|---|---|
| \(\widehat{X}\) | Computed or approximated value of \(X\), with the hat indicating a floating-point computed quantity when used that way in the paper. |
| \(u_{\mathrm{crs}}\) | Unit roundoff associated with the coarse precision used for selected computations. |
| \(u_{\mathrm{fine}}\) | Unit roundoff associated with the fine precision used for selected computations. |

Unit roundoff measures the scale of rounding errors in floating-point arithmetic. Smaller unit roundoff generally corresponds to higher precision.

The precise use of coarse and fine precision depends on the algorithm and its assumptions.

## 4. Embedding and Error Notation

| Symbol | Meaning |
|---|---|
| \(\varepsilon\) | Embedding accuracy parameter controlling the allowed distortion in relevant geometric quantities. |
| \(\varepsilon\)-embedding | A mapping that approximately preserves inner products or norms for vectors in the relevant subspace, within a specified error bound. |

A typical inner-product embedding condition is

\[
\left|x^Ty-(\Theta x)^T(\Theta y)\right|
\leq \varepsilon \|x\|_2\|y\|_2.
\]

Here, \(x\) and \(y\) belong to the subspace for which the embedding guarantee applies.

## 5. Block Notation

| Symbol | Meaning |
|---|---|
| \(W^{(i)}\) | The \(i\)-th block of columns of the input matrix. |
| \(Q^{(i)}\) | The orthonormalized version of the \(i\)-th block. |
| \(Q'^{(i)}\) | Intermediate block obtained after removing components in previously computed directions. |
| \(R_{i,i}\) | Diagonal block of the block upper-triangular factor \(R\). |
| \(R_{1:i-1,i}\) | Block of projection coefficients connecting previously processed blocks to the current block. |
| \(S^{(i)}\) | Sketch of the computed block, \(S^{(i)} = \Theta Q^{(i)}\). |
| \(P^{(i)}\) | Sketch of the current input block, \(P^{(i)} = \Theta W^{(i)}\). |

The block notation allows several columns to be processed together rather than one column at a time.

## 6. Important Relationships

The following equations summarize the main relationships used in the randomized block Gram–Schmidt algorithm:

### QR factorization

\[
W = QR.
\]

### Sketch of the input matrix

\[
P = \Theta W.
\]

### Sketch of the computed basis

\[
S = \Theta Q.
\]

### Removal of previously computed directions

\[
Q'^{(i)}
=
W^{(i)} - Q_{1:i-1}R_{1:i-1,i}.
\]

### Orthonormalization of the current block

\[
Q'^{(i)} = Q^{(i)}R_{i,i}.
\]

The randomized method uses the sketch to estimate projection coefficients and to perform certain calculations efficiently, while retaining the basis vectors in the original space.

## 7. Important Note

This page is a working notation reference. Before final submission, each definition, precision convention, and equation should be checked against the paper's notation and Algorithm 2.1. In particular, the precise definition of \(m_p\), the use of hats, and the roles of \(u_{\mathrm{crs}}\) and \(u_{\mathrm{fine}}\) must follow the paper exactly.

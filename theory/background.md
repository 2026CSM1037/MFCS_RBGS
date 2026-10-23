
# Background: From Classical Gram–Schmidt to RBGS

## 1. Why Is Orthogonalization Needed?

In numerical linear algebra, QR factorization represents a matrix $W$ as

$$
W = QR,
$$

where the columns of $Q$ are orthonormal and $R$ is upper triangular.

Orthonormal columns are important in numerical algorithms because they provide a stable basis for representing vectors and subspaces. QR factorization is used in applications such as solving linear systems and computing eigenvalues.

## 2. Why Can Classical Gram–Schmidt Fail?

Classical Gram–Schmidt (CGS) constructs an orthogonal basis by subtracting projections of each new vector onto previously computed basis vectors.

In exact arithmetic, this process produces orthogonal vectors when the input vectors are linearly independent.

However, computers perform calculations using finite-precision floating-point arithmetic. Rounding errors can accumulate during projection and subtraction. Consequently, the computed basis may lose orthogonality, meaning that

$$
\widehat{Q}^{T}\widehat{Q} \neq I.
$$

The problem can become more severe when the input vectors are nearly linearly dependent. In such cases, small rounding errors may become significant relative to the remaining vector after projection.

Loss of orthogonality can affect the reliability of subsequent numerical computations.

## 3. Block Gram–Schmidt Methods

Block Gram–Schmidt methods process several columns together instead of processing only one vector at a time.

### 3.1 Block Modified Gram–Schmidt (BMGS)

BMGS performs orthogonalization by processing blocks and removing components in previously computed directions. Its organization supports efficient matrix operations and can take advantage of modern computing hardware.

### 3.2 Block Classical Gram–Schmidt with Two Passes (BCGS2)

BCGS2 applies a second orthogonalization pass to improve the orthogonality of the computed basis. This additional pass can help correct residual components left behind by the first pass.

Both methods are designed to construct useful orthogonal bases, but their numerical behavior and computational costs depend on the input matrix and implementation.

## 4. The Main Idea of Randomized Block Gram–Schmidt (RBGS)

Randomized Block Gram–Schmidt (RBGS) uses a random sketching matrix $\Theta$ to create a lower-dimensional representation of the input vectors.

Instead of performing every operation directly on the original high-dimensional vectors, RBGS uses the sketch to estimate projection coefficients and carry out selected calculations in a smaller space.

The resulting coefficients are then used to update the original vectors. This approach aims to reduce computational and communication costs while maintaining the accuracy needed for numerical applications.

## 5. Why Is RBGS Useful?

RBGS is designed for large-scale numerical computations in which orthogonalization can be expensive.

Its potential advantages include:

- Reducing the dimensions of selected calculations through sketching.
- Processing columns in blocks to exploit efficient matrix operations.
- Reducing communication and synchronization in distributed computations.
- Supporting applications such as iterative linear-system solvers and eigenvalue algorithms.

These benefits depend on the sketch size, embedding properties, floating-point precision, and computational environment.

## 6. Summary

Classical Gram–Schmidt can lose orthogonality because of floating-point rounding errors. Block methods organize orthogonalization around groups of columns, while BCGS2 adds a second pass to improve orthogonality. RBGS introduces randomized sketching to estimate projections in a smaller space, aiming to make large-scale orthogonalization more efficient.

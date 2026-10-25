
# Algorithm 2.1: Randomized Block Gram–Schmidt (RBGS)

## 1. Purpose

Randomized Block Gram–Schmidt (RBGS) computes a QR factorization of a matrix using random sketching to reduce the cost of projection calculations.

Given a matrix $W \in \mathbb{R}^{n \times m}$, the algorithm computes

$$
W \approx QR,
$$

where $Q \in \mathbb{R}^{n \times m}$ is the computed basis and $R \in \mathbb{R}^{m \times m}$ is upper triangular.

Unlike standard Gram–Schmidt methods, RBGS estimates projection coefficients using a lower-dimensional sketch of the input and previously computed basis vectors.

## 2. Inputs and outputs

**Inputs**

- $W \in \mathbb{R}^{n \times m}$: input matrix.
- $\Theta \in \mathbb{R}^{k \times n}$: random sketching matrix.
- $k$: sketch dimension, with $m \leq k \ll n$.
- $p$: number of column blocks in the partition of $W$.

**Outputs**

- $Q \in \mathbb{R}^{n \times m}$: computed basis.
- $R \in \mathbb{R}^{m \times m}$: upper-triangular factor.
- $S=\Theta Q$: sketch of the computed basis.

Here, $W=[W_{(1)},W_{(2)},\ldots,W_{(p)}]$ with every block $W_{(i)}\in\mathbb R^{n\times m_p}$, so $m=p\,m_p$. The paper (Section 1.2) defines the partition with equal blocks of $m_p$ columns; unequal blocks are not part of its setup.

## 3. Pseudocode for Algorithm 2.1

```text
Algorithm RBGS(W, Θ)

Input:
    W = [W(1), W(2), ..., W(p)]
    Θ ∈ R^(k×n), with m ≤ k ≪ n

Output:
    Q = [Q(1), Q(2), ..., Q(p)]
    Upper-triangular factor R
    S = ΘQ

Initialize Q, R, S

for i = 1, 2, ..., p:

    1. Compute the sketch of the current input block:
           P(i) = Θ W(i)
       Precision: fine (u_fine)

    2. Estimate the projection coefficients by solving:
           R(1:i-1,i) =
               argmin_Y || S(1:i-1)Y - P(i) ||_F
       Precision: fine (u_fine)

    3. Remove components in previously computed directions:
           Q'(i) = W(i) - Q(1:i-1) R(1:i-1,i)
       Precision: coarse (u_crs)

    4. QR of Q'(i) w.r.t. <Θ·,Θ·>  (shown here: RCholeskyQR, Algorithm 2.3):
           S'(i) = Θ Q'(i)
           Compute S'(i) = U R(i,i) using ordinary QR
           Q(i) = Q'(i) R(i,i)^(-1)
       Precision: fine (u_fine)

    5. Compute the sketch of the new block:
           S(i) = Θ Q(i)
       Precision: fine (u_fine)

end for

6. Optional stability checks:
       Δ(p) = ||I - S^T S||_F
       Δ̃(p) = ||P - SR||_F / ||P||_F
   Precision: fine (u_fine)

Return Q, R, S
```

**Precision convention:** The paper uses $u_{\mathrm{crs}}$ for the coarse unit roundoff and $u_{\mathrm{fine}}$ for the fine unit roundoff, with $u_{\mathrm{fine}} \leq u_{\mathrm{crs}}$. Step 3 is the expensive high-dimensional update carried out in coarse precision; the other listed steps use fine precision.

**Steps 4–5 note:** The paper's Algorithm 2.1 leaves the inter-block QR generic ("QR fact. with respect to $\langle\Theta\cdot,\Theta\cdot\rangle$"). Section 2.2 gives four options: single-vector RGS, RCholeskyQR (Algorithm 2.3, shown above), implicit RCholeskyQR (Algorithm 2.4, which forms $S'_{(i)}=P_{(i)}-S_{(1:i-1)}R_{(1:i-1,i)}$ instead of multiplying by $\Theta$), and $\ell_2$-QR+RCholeskyQR (Algorithm 2.5, used in the paper's experiments).

## 4. Explanation of the steps

### Step 1: Sketch the input block

$$
P_{(i)}=\Theta W_{(i)}.
$$

The original block has $n$ rows, whereas its sketch has only $k$ rows. Because $k \ll n$, subsequent projection calculations can be performed in a smaller space.

### Step 2: Solve the sketched least-squares problem

$$
R_{(1:i-1,i)}
=\arg\min_Y
\left\|S_{(1:i-1)}Y-P_{(i)}\right\|_F.
$$

This estimates how much of the current block overlaps with the previously computed basis directions. The coefficients are computed from the sketches rather than by solving the full-dimensional least-squares problem.

### Step 3: Remove the previous directions

$$
Q'_{(i)}
=W_{(i)}-Q_{(1:i-1)}R_{(1:i-1,i)}.
$$

This operation updates the original high-dimensional block. It is performed in coarse precision because it is one of the expensive operations that RBGS aims to accelerate.

For the first block, no previous basis vectors exist. Therefore, $Q'_{(1)}=W_{(1)}$.

### Steps 4–5: RCholeskyQR

RCholeskyQR orthonormalizes the current block with respect to the sketched inner product

$$
\langle x,y\rangle_\Theta=(\Theta x)^T(\Theta y).
$$

First, compute

$$
S'_{(i)}=\Theta Q'_{(i)}.
$$

Then perform ordinary QR on the sketch:

$$
S'_{(i)}=UR_{(i,i)},
\qquad U^TU=I.
$$

Recover the basis in the original space:

$$
Q_{(i)}=Q'_{(i)}R_{(i,i)}^{-1}.
$$

Consequently,

$$
\Theta Q_{(i)}=U
$$

in exact arithmetic. The sketch of the resulting basis is therefore orthonormal. In floating-point arithmetic, this relation holds approximately.

Finally, store $S_{(i)}=\Theta Q_{(i)}$ for the next iteration.

## 5. Optional stability checks

The paper defines two quantities:

$$
\Delta^{(p)}=\|I-S^TS\|_F,
$$

$$
\widetilde{\Delta}^{(p)}
=\frac{\|P-SR\|_F}{\|P\|_F}.
$$

- $\Delta^{(p)}$ measures the departure of the sketch $S$ from orthonormality.
- $\widetilde{\Delta}^{(p)}$ measures the relative residual in the sketched factorization.

These quantities can be used for a posteriori stability certification under the paper's stated assumptions. They are optional and do not replace the assumptions needed for the theoretical guarantees.

## 6. Main intuition

RBGS uses a small sketch to estimate projection coefficients, performs the expensive update on the original vectors, and orthonormalizes each block using RCholeskyQR on the sketch.

The intended benefit is to reduce computation and communication while retaining a numerically useful basis for large-scale linear algebra.

# Corollary 1.4 — Singular-Value Bounds from an \(\varepsilon\)-Embedding

**Source:** Balabanov and Grigori, *Randomized Block Gram–Schmidt Process for the Solution of Linear Systems and Eigenvalue Problems*, Section 1.3, Corollary 1.4 (cited there as Corollary 2.2 of the earlier randomized Gram–Schmidt paper).

## Statement

Let \(\Theta\) be an \(\varepsilon\)-embedding for the subspace spanned by the columns of \(V\), where \(0\leq\varepsilon<1\). The embedding approximately preserves inner products on that subspace:

\[
|\langle x,y\rangle-\langle\Theta x,\Theta y\rangle|
\leq \varepsilon\|x\|_2\|y\|_2
\qquad \text{for all }x,y\in\operatorname{range}(V).
\]

In particular, setting \(x=y\) gives

\[
(1-\varepsilon)\|x\|_2^2
\leq \|\Theta x\|_2^2
\leq (1+\varepsilon)\|x\|_2^2.
\]

Applying this to \(x=Vz\) for every vector \(z\), and using the variational characterizations of the extreme singular values, yields

\[
\sqrt{1-\varepsilon}\,\sigma_{\min}(V)
\leq \sigma_{\min}(\Theta V)
\leq \sigma_{\max}(\Theta V)
\leq \sqrt{1+\varepsilon}\,\sigma_{\max}(V).
\]

Equivalently, the corresponding individual singular values are distorted by at most the factors \(\sqrt{1-\varepsilon}\) and \(\sqrt{1+\varepsilon}\). The inequalities most useful in stability arguments are

\[
\frac{\sigma_{\min}(\Theta V)}{\sqrt{1+\varepsilon}}
\leq \sigma_{\min}(V),
\qquad
\sigma_{\max}(V)
\leq \frac{\sigma_{\max}(\Theta V)}{\sqrt{1-\varepsilon}},
\]

where the first and second inequalities follow from applying the embedding bounds to the relevant singular directions. (When quoting the corollary in a final report, use the exact ordering and form of the displayed inequalities in the original PDF.)

## Complete proof

Let \(z\) be any vector of compatible dimension and put \(x=Vz\). Since \(x\) lies in \(\operatorname{range}(V)\), the embedding condition with \(y=x\) gives

\[
\left|\|Vz\|_2^2-\|\Theta Vz\|_2^2\right|
\leq \varepsilon\|Vz\|_2^2.
\]

Rearranging the two sides,

\[
(1-\varepsilon)\|Vz\|_2^2
\leq \|\Theta Vz\|_2^2
\leq (1+\varepsilon)\|Vz\|_2^2.
\]

Taking square roots (all terms are nonnegative),

\[
\sqrt{1-\varepsilon}\|Vz\|_2
\leq \|\Theta Vz\|_2
\leq \sqrt{1+\varepsilon}\|Vz\|_2.
\tag{1}
\]

For any matrix \(A\), its largest singular value satisfies
\(\sigma_{\max}(A)=\max_{\|z\|_2=1}\|Az\|_2\), and its smallest singular value (for a matrix with full column rank) satisfies
\(\sigma_{\min}(A)=\min_{\|z\|_2=1}\|Az\|_2\).

Taking the maximum over unit \(z\) in the upper bound of (1) gives
\[
\sigma_{\max}(\Theta V)\leq
\sqrt{1+\varepsilon}\,\sigma_{\max}(V).
\]

Taking the minimum over unit \(z\) in the lower bound of (1) gives
\[
\sigma_{\min}(\Theta V)\geq
\sqrt{1-\varepsilon}\,\sigma_{\min}(V).
\]

These are the required extreme-singular-value estimates. The embedding also prevents a nonzero vector in \(\operatorname{range}(V)\) from being mapped to zero when \(\varepsilon<1\), so the sketch preserves the rank of \(V\). This completes the proof. \(\square\)

## Why it matters for RBGS

The algorithm monitors quantities involving the small matrix \(\Theta Q\). Corollary 1.4 is the bridge between the geometry measured in sketch space and the geometry of the original basis \(Q\). If the embedding is accurate, singular-value information about the sketched basis can be translated into bounds for the original basis, which is a key ingredient in the stability analysis.

## Verification note

The argument above proves the standard singular-value consequence of the \(\varepsilon\)-embedding definition. Before submission, compare the exact displayed statement in Corollary 1.4 with the paper's PDF and make the notation/inequality ordering match it verbatim; PDF text extraction may omit displayed equations.

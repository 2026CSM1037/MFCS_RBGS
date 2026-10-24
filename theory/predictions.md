# Experiment Predictions for RBGS

## 1. Purpose

This document records **theory-based predictions to make before running
experiments** for the Randomized Block Gram–Schmidt (RBGS) process
described in the paper *Randomized Block Gram–Schmidt Process for the
Solution of Linear Systems and Eigenvalue Problems*.

These predictions are hypotheses, not experimental results. After
implementation, the results should be compared with them and any
differences explained.

The main quantities to examine are:

- orthogonality and conditioning of the computed basis;
- the residual/error indicators used by the paper;
- runtime and memory/work required;
- sensitivity to sketch size, sketch construction, the number of
  Richardson iterations in the step-2 solver, and arithmetic precision.

Use the same input matrices and the same stopping criteria when
comparing configurations, unless the experiment is specifically designed
to vary those factors.

------------------------------------------------------------------------

## 2. What should be measured?

Where supported by the implementation, record the following for every
run.

| Quantity | What it tells us |
|---|---|
| Sketch size $k$ | Number of rows of $\Theta$, i.e. of the sketch $\Theta W$. The method requires $m \le k \ll n$. |
| Orthogonality error $\lVert I-Q^TQ\rVert$ | How close the computed basis is to orthonormal in the original Euclidean inner product. |
| Condition number $\operatorname{cond}(Q)$ | The quantity the paper plots in Figures 1b, 2b and 3b. |
| Sketched orthogonality $\Delta^{(p)}=\lVert I-S^TS\rVert_F$, with $S=\Theta Q$ | How close the basis is to orthonormal in the sketched inner product. |
| Sketched residual $\widetilde\Delta^{(p)}=\lVert P-SR\rVert_F/\lVert P\rVert_F$, with $P=\Theta W$ | How accurately the factorization holds in sketch space. |
| Factorization error $\lVert W-QR\rVert/\lVert W\rVert$ | Accuracy in the original space (paper, Figure 1b). |
| Number of Richardson iterations | Iterations used by the step-2 least-squares solver, if Richardson is the solver. |
| Runtime | The computational cost of each configuration. |
| Failure/instability | Whether factorization fails, values become non-finite, or the output violates expected tolerances. |

**Important:** The paper's a posteriori indicators and the Euclidean
orthogonality error are not interchangeable. A small error measured in
sketch space does not, by itself, prove that $Q$ is highly orthogonal in
the original space. Measure both when possible.

Use the matrix norm and exact diagnostic definitions required by the
paper or the assignment. Do not silently substitute one norm for
another.

------------------------------------------------------------------------

## 3. Prediction A — Effect of sketch size $k$

### Hypothesis

With the input matrix, sketch family, block size, precision, and
iteration settings fixed, increasing $k$ should generally make the
sketch a more reliable representation of the relevant column-space
geometry. This is expected to reduce the chance that the sketch distorts
lengths or angles enough to damage the computation.

### Expected observations

-   **Small $k$:** lower sketching cost, but a greater risk that the
    sketch poorly represents the column space. Orthogonality or residual
    diagnostics may be worse or more variable, and some runs may fail.
-   **Moderate $k$:** a likely compromise between cost and numerical
    reliability.
-   **Large $k$:** generally more reliable geometric information, but
    greater sketch storage and multiplication cost. Improvements may
    eventually become small compared with the additional cost.

### Why

A sketch compresses information from the original $n$-dimensional space
into $k$ dimensions. If $k$ is too small for the problem and required
accuracy, important geometric information may not be preserved
adequately. Increasing $k$ gives the embedding more room to preserve
that information, at the cost of additional work.

### How to test it fairly

Choose several valid $k$ values, from relatively small to larger values
allowed by the method. Keep the input, block size, sketch type,
arithmetic precision, and number of Richardson iterations fixed. For randomized
sketches, repeat runs with multiple random seeds and report the mean and
spread, not just one run.

### What would support the hypothesis?

A trend toward more reliable diagnostics or fewer failures as $k$
increases, accompanied by increased runtime or memory use, would support
the prediction. A perfectly monotonic improvement is **not** guaranteed
for every random seed or finite-precision run.

------------------------------------------------------------------------

## 4. Prediction B — Effect of the number of Richardson iterations (step-2 solver)

### Hypothesis

If Richardson iterations are used as the step-2 least-squares solver,
additional iterations should initially reduce the error in the computed
coefficients $R_{(1:i-1,i)}$, and hence in the diagnostics, provided the refinement is in its stable/convergent regime. After
some point, improvement may level off because of finite-precision
effects, a stopping tolerance, or other sources of error.

### Expected observations

-   **Zero or few iterations:** lower cost, but potentially larger
    residual/error.
-   **A suitable number of iterations:** smaller residual/error if the
    refinement is converging.
-   **Too many iterations:** extra runtime with little additional
    benefit; in an unfavourable numerical regime, the error may stop
    improving or may worsen.

### What Richardson does in this paper

In the paper, Richardson iterations are the **step-2 least-squares
solver** (Section 2.1, Algorithm SM.1):
X ← X + S(1:i-1)ᵀ (P(i) − S(1:i-1) X). They are not a refinement of the
final factorization. With l = 1 iteration, RBGS is exactly BCGS in the
sketched inner product; l iterations correspond to BCGS with l
re-orthogonalizations. In exact arithmetic the error contracts
geometrically, ‖x_j − y_j‖ ≤ (Δ^(i−1))^l ‖y_j‖ (SM.2), so a few
iterations should suffice whenever Δ^(i−1) is small. The paper uses 5
iterations in Sections 5.2–5.3; in Section 5.1, 20 CG iterations were
insufficient in unique precision but sufficient in multi-precision.
Theorem 3.4 needs the step-2 solver to be backward stable
(Assumptions 2.2); Theorem 3.2 does not.

### Why

An iterative refinement method applies corrections to reduce a
discrepancy. If each correction is effective, the error decreases over
the early iterations. Finite precision and the method's convergence
conditions limit how far this process can be expected to help.

### How to test it fairly

Hold the matrix, sketch, sketch size, block size, and precision fixed.
Run with a small range of iteration limits, including the
baseline/no-refinement case if the implementation permits it. Record the
diagnostic after each setting and the runtime. Use the same
convergence/stopping tolerance in all comparable runs.

### Interpretation caution

Do not claim that more iterations must always improve the result. The
prediction is conditional on the refinement method being applicable and
convergent for that configuration. Also distinguish an iteration
**limit** from the number of iterations actually executed.

------------------------------------------------------------------------

## 5. Prediction C — Effect of sketch type

### Hypothesis

Different sketch constructions may trade off computational cost against
embedding quality and implementation overhead. A sketch that is cheaper
to apply may improve runtime, while a sketch with stronger practical
embedding behaviour for the tested matrices may produce more reliable
numerical diagnostics.

### Expected observations

-   Sketches that are inexpensive to construct and apply may be faster.
-   Sketches with less favourable embedding behaviour for a particular
    input may produce less reliable diagnostics or require a larger $k$.
-   There may be no single sketch type that is fastest and most accurate
    for every matrix and machine.

### Why

The sketch determines how the original matrix is mapped into a
lower-dimensional representation. Different constructions have different
costs, sparsity patterns, and embedding properties. Their practical
performance also depends on matrix dimensions, data, implementation, and
hardware.

### How to test it fairly

Compare only sketch types supported by the paper and the implementation.
Keep $k$, input data, block size, precision, and number of Richardson iterations
fixed. If sketch types have different setup costs, report setup time
separately from the main algorithm time where possible. Repeat
randomized runs using multiple seeds.

### What the paper reports

The paper uses Rademacher and SRHT sketches and states that they gave
very similar results in all its experiments (Section 5). Note also that
the method requires m ≤ k; the paper used k = 3000 for m = 300
(Section 5.1) and k = 5000 or 7500 for m = 3100 (Section 5.2).

### Interpretation caution

Do not declare one sketch universally superior from a single matrix or a
single random seed. Report both numerical diagnostics and cost.

------------------------------------------------------------------------

## 6. Prediction D — Effect of arithmetic precision

### Hypothesis

Using higher precision for numerically sensitive operations should
generally reduce rounding error and may improve orthogonality or
residual diagnostics, but it can increase runtime and memory traffic.
The effect depends on which steps use each precision and on the
conditioning of the input.

### Expected observations

-   Lower precision may be faster but more sensitive to ill-conditioning
    and rounding.
-   Higher precision may improve numerical reliability, particularly in
    sensitive operations, but may cost more.
-   Mixed precision can offer a useful compromise if the
    coarse-precision steps remain accurate enough for the algorithm's
    assumptions and the fine-precision corrections are effective.

### Why

Floating-point arithmetic introduces rounding errors. These errors can
accumulate during matrix products, projections, and factorization.
Higher precision reduces the unit roundoff, but it does not eliminate
errors due to ill-conditioning, an inadequate sketch, or an unsuitable
algorithmic configuration.

### How to test it fairly

Compare the precision configurations actually permitted by the algorithm
and implementation. Keep all other parameters fixed. Record the
precision used in each algorithm step, not just a single label such as
"mixed precision." Compare numerical diagnostics and runtime.

### Interpretation caution

Do not assume that higher precision always fixes a poor sketch or an
ill-conditioned problem. Precision and sketch quality are different
factors.

------------------------------------------------------------------------

## 7. Prediction E — Effect of block size

Include this experiment only if block size is an adjustable parameter in
your implementation or is explicitly required by the assignment.

### Hypothesis

Block size can affect both numerical behaviour and computational
efficiency. Larger blocks may make matrix operations more efficient on
some hardware, but they can change the amount of work done at each
orthogonalization step and the memory requirements.

### Expected observations

-   Very small blocks may have more per-block overhead.
-   Larger blocks may use optimized matrix operations more effectively,
    but may require more memory and can change the numerical behaviour.
-   The best block size is likely to depend on the input dimensions and
    implementation.

### How to test it fairly

Vary block size while holding the input, sketch size, sketch type,
precision, and number of Richardson iterations fixed. Do not include this experiment
if block size is outside your assigned scope or cannot be varied in the
available code.

------------------------------------------------------------------------

## 8. Suggested experiment matrix

This is a planning table, not a requirement to run every combination.
Begin with a baseline and vary one factor at a time.

| Experiment | Change one factor | Keep fixed | Main outputs |
|---|---|---|---|
| Baseline | None | Choose and record all parameters | All diagnostics, runtime, failures |
| Sketch-size study | $k$ | Input, sketch type, block size, precision, Richardson iterations | Error indicators, orthogonality, runtime |
| Richardson study | Number of Richardson iterations | Input, $k$, sketch type, block size, precision | Error versus iterations, runtime |
| Sketch-type study | Sketch construction | Input, $k$, block size, precision, Richardson iterations | Diagnostics, setup and total runtime |
| Precision study | Precision configuration | Input, $k$, sketch type, block size, Richardson iterations | Diagnostics, runtime |
| Block-size study (optional) | Block size $m_p$ | Input and remaining settings | Diagnostics, runtime, memory if available |

If computational resources are limited, prioritize the factors
explicitly required by the assignment and the parameters exposed by the
existing implementation. Do not build extra infrastructure solely to run
optional studies.

------------------------------------------------------------------------

## 9. How to report the results later

After implementation and experiments, add a results section to the
project report or the appropriate existing results file. For each
experiment:

1.  State the hypothesis **before** showing the results.
2.  Specify the parameter that changed and the settings held fixed.
3.  Present a table or plot of measured diagnostics and runtime.
4.  State whether the observation supported, partly supported, or
    contradicted the hypothesis.
5.  Explain plausible reasons for the outcome, separating evidence from
    speculation.
6.  Mention variability across random seeds when applicable.

Example wording:

> **Prediction:** Increasing $k$ will usually improve the reliability of
> the sketch, but increase sketching cost.\
> **Observation:** To be filled in after the experiments.\
> **Interpretation:** To be written from the measured results; do not
> fill this in before running the code.

Do not invent numerical values or write that a prediction was confirmed
before observing the results.

------------------------------------------------------------------------

## 10. Is this an ablation study?

The prediction document and an ablation study are related but not
identical.

-   **Prediction:** a theory-based expectation written before running
    the experiment.
-   **Parameter study:** systematically changes a setting such as $k$,
    iteration count, or precision and observes its effect.
-   **Ablation study:** removes or changes a component of a method to
    assess its contribution, usually against a baseline.

For example, varying $k$ is a parameter study. Comparing the full method
against a version with a component disabled may be an ablation, if such
a comparison is meaningful and permitted by the algorithm.

For this assigned task, write the predictions now. Validate them after
the coding and experimental work. Do not turn this document into a full
ablation report unless that is explicitly part of the project
requirements.

------------------------------------------------------------------------

## 11. Final checklist

-   [ ] Predictions are written before inspecting experimental outcomes.
-   [ ] Each prediction has a mathematical or algorithmic rationale.
-   [ ] The parameters to vary and the parameters to hold fixed are
    stated.
-   [ ] Numerical quality and computational cost are both considered.
-   [ ] The paper's sketch-space indicators are distinguished from
    Euclidean orthogonality.
-   [ ] No experimental outcomes are claimed before runs are completed.
-   [ ] Later results will be compared with these predictions honestly.
-   [ ] Optional experiments are omitted if they are outside the
    assigned scope.

**Scope note:** This file specifies expected trends, not guaranteed
outcomes. The exact trends depend on the assumptions of the paper, input
matrix, implementation, and hardware.

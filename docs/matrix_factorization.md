<!-- Copyright 2026 DeepMind Technologies Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

# Matrix Factorization Notation

<!-- disableFinding(LINE_OVER_80) -->

{mod}`jax_privacy.matrix_factorization` implements correlated-noise
("DP-FTRL") mechanisms: define, optimize, instantiate, and apply a matrix
factorization of a workload. This page records the notation used by that
public API. For a runnable overview, see
[`dpmf_strategy_optimization.py`](https://github.com/google-parfait/jax_privacy/blob/main/examples/dpmf_strategy_optimization.py).

The full name (left column) is generally used for kwargs on public functions.
Single-letter names appear in implementations and match the papers cited
below.

## Notation and symbols

| Full name / API name | Symbol or short name | Description |
|----------------------|----------------------|-------------|
| `workload` | $A$ | Workload matrix, for example `np.tri` (the lower-triangular matrix of ones) for prefix sums. |
| `strategy_matrix` | $C$ | Strategy matrix, also sometimes called the encoder matrix. |
| `noising_matrix` | $C^{-1}$ or `C_inv` | In a factorization $A = BC$, the noise added on iteration $i$ is conceptually `(C_inv @ Z)[i, :]`. Production code usually applies this implicitly. |
| `decoder_matrix` | $B$ | Decoder matrix $B$ in the factorization $A = BC$. |
| `n` | $n$ | Number of iterations the mechanism supports, equal to the number of rows and columns in $A$. |
| `max_participations` | $k$ | Number of epochs, or maximum participations for one datum or user. |
| `min_sep` | $b$ | Used both for `num_bands` (banded matrices) and min-separation participation. |
| `sensitivity` | `sens` | L2 sensitivity of $C$ under a [participation schema](#participation-schemas). |
| `error` | `error` | Variance (squared error) of $C^{-1}$ under unit noise, ignoring sensitivity. |
| `loss` | `loss` | Total loss, including sensitivity: $\mathrm{loss}(C, B) = \mathrm{sens}(C)^2 \cdot \mathrm{error}(B)$. |

## Types of loss and error

By default the library operates on squared losses and errors.
`per_query_error` is the vector of errors for each query (row) of the
workload $A$. This is usually reduced to a scalar: `max_error` (maximum
per-query error) or `mean_error` (mean per-query error). Papers often report
`sqrt(mean_error)`, also called `rmse` (root-mean-squared error).

(participation-schemas)=
## Sensitivity and participation schemas

The L2 sensitivity of a matrix mechanism is a function of $C$. See
[Choquette-Choo et al. (2023)](https://arxiv.org/abs/2211.06530) and
[Choquette-Choo et al. (2023)](https://arxiv.org/abs/2306.08153).

Definitions are with respect to an abstract privacy unit. With appropriate
bundling (for example FedAvg-style), per-example notions convert to per-user
notions. This page may use `example`, `user`, and `datum` interchangeably
under that understanding.

Sensitivity depends on which iterations each example participates in:

| Name in code | Full name | Description |
| ------------ | --------- | ----------- |
| (implicit) | single-participation or single-epoch | Each example participates at most once in training. This is the default if unspecified. See [Denisov et al. (2022)](https://arxiv.org/abs/2202.08312). |
| `min_sep` | $b$ min-separation | Examples are at least $b$ steps apart. If an example participates in iterations $i$ and $j$, then $\|i - j\| \ge b$. $(k, b)$-participation below is a special case. See [Choquette-Choo et al. (2023)](https://arxiv.org/abs/2306.08153). |
| `fixed_epoch` | $(k, b)$ participation | There are exactly $n = k \cdot b$ iterations, and participations are separated by multiples of $b$: if a client participates on $i$ and $j$, then $\|i - j\| \bmod b = 0$. See [Choquette-Choo et al. (2023)](https://arxiv.org/abs/2211.06530). |

## Parameterized matrix classes

Parameterizing $C$ and $C^{-1}$ makes sensitivity and error cheaper to
compute, and makes online noise generation efficient (for example in
DP-FTRL).

### General banded matrices

Implemented in {mod}`jax_privacy.matrix_factorization.banded`, introduced in
[Choquette-Choo et al. (2023)](https://arxiv.org/abs/2306.08153).

A matrix $X$ is $b$-banded if $i, j \in [n]$ and $|i - j| \ge b$ implies
$X_{[i,j]} = 0$. This is off-by-one from the usual bandwidth ($X$ has
bandwidth $b-1$), but it matches $b$-banded matrices with $b$-min-separation
participation. For $b$-banded lower-triangular matrices, $b$ is the number of
bands.

### Banded Toeplitz matrices

Implemented in {mod}`jax_privacy.matrix_factorization.toeplitz`. $C \in
\mathbb{R}^{n \times n}$ is a lower-triangular Toeplitz matrix, parameterized
by `n` (the size of $C$) and `coef` (the nonzero Toeplitz coefficients of
$C$). See [McKenna (2024)](https://arxiv.org/abs/2405.15913).

TODO: b/329444015 - Add a markdown example where n = 4 and coef = [1, 0.5].

### Buffered linear Toeplitz (BLT) matrices

Implemented in {mod}`jax_privacy.matrix_factorization.buffered_toeplitz`. BLT
mechanisms were introduced in [Dvijotham et al. (2024)](https://arxiv.org/abs/2404.16706)
and extended to multiple participations in
[Choquette-Choo et al. (2024)](https://arxiv.org/abs/2408.08868).

### LTI Toeplitz matrices

These matrices are parameterized by a single constant $\nu$ that defines the
recursive relationship for each new entry in a column. They are not
optimized, though $\nu$ can be swept with respect to some error definition.
They are optimal for an LTI error definition. See
[Choquette-Choo et al. (2024)](https://arxiv.org/abs/2310.06771).

## References

-   [Denisov et al. (2022)](https://arxiv.org/abs/2202.08312). *Improved
    Differential Privacy for SGD via Optimal Private Linear Operators on
    Adaptive Streams.* NeurIPS 2022.
-   [Henzinger, Upadhyay, and Yeo (2023)](https://arxiv.org/abs/2202.11205).
    *Constant Matters: Fine-grained Complexity of Differentially Private
    Continual Observation.* ICML 2023.
-   [Choquette-Choo et al. (2023)](https://arxiv.org/abs/2211.06530).
    *Multi-Epoch Matrix Factorization Mechanisms for Private Machine
    Learning.* ICML 2023.
-   [Choquette-Choo et al. (2023)](https://arxiv.org/abs/2306.08153).
    *(Amplified) Banded Matrix Factorization: A unified approach to private
    training.* NeurIPS 2023.
-   [Choquette-Choo et al. (2024)](https://arxiv.org/abs/2310.06771).
    *Correlated Noise Provably Beats Independent Noise for Differentially
    Private Learning.* ICLR 2024.
-   [Dvijotham et al. (2024)](https://arxiv.org/abs/2404.16706). *Efficient
    and Near-Optimal Noise Generation for Streaming Differential Privacy.*
    FOCS 2024.
-   [Kalinin and Lampert (2024)](https://arxiv.org/abs/2405.13763). *Banded
    Square Root Matrix Factorization for Differentially Private Model
    Training.* NeurIPS 2024.
-   [McKenna (2024)](https://arxiv.org/abs/2405.15913). *Scaling up the
    Banded Matrix Factorization Mechanism for Differentially Private ML.*
    ICLR 2025.
-   [Choquette-Choo et al. (2024)](https://arxiv.org/abs/2408.08868). *A
    Hassle-free Algorithm for Private Learning in Practice: Don't Use Tree
    Aggregation, Use BLTs.* EMNLP Industry Track 2024.

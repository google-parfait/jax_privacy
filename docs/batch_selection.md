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

# Batch Selection Strategies

<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LINK_ID) -->

This document overviews the research behind different batch selection
(subsampling) strategies for differentially private training and summarizes
their support in `jax_privacy` and `dp_accounting`.

## Background

Differentially private training algorithms like DP-SGD
([Abadi+ 2016](#abadi-2016))
rely on two key ingredients: (1) gradient clipping to bound each example's
contribution, and (2) additive noise to mask individual examples.
The **batch selection strategy** determines which subset of examples
participates in each training step, and the analysis of this selection
strategy is critical for quantifying the overall privacy guarantee.

### Noise addition mechanisms

JAX Privacy supports two noise addition paradigms:

*   **DP-SGD** (independent noise): Gaussian noise is added independently at
    each step.
    See {func}`~jax_privacy.noise_addition.gaussian_privatizer` in
    {mod}`~jax_privacy.noise_addition`.

*   **DP-MF** (correlated noise via Matrix Factorization): Noise is correlated
    across steps using a lower-triangular matrix, which can reduce total error.
    See [Pillutla+ 2025](#pillutla-2025) for an introduction and survey, the
    {mod}`~jax_privacy.matrix_factorization` module, and
    {func}`~jax_privacy.noise_addition.matrix_factorization_privatizer` in
    {mod}`~jax_privacy.noise_addition`.

### API tiers and batch selection

JAX Privacy provides [three API tiers](api-tiers) with different levels of
built-in DP assurance; see the [Overview](overview) for the full description.
In the context of batch selection:

*   **Tier 1 (end-to-end training loops)** and **Tier 2
    ({class}`~jax_privacy.execution_plan.DPExecutionPlan` via
    {class}`~jax_privacy.execution_plan.BandMFConfig`)**: Batch selection is
    configured automatically and coupled with noise addition and accounting by
    construction. Currently supports
    {class}`~jax_privacy.batch_selection.CyclicPoissonSampling` variants
    (including truncated Poisson and fixed-order multi-epoch). See
    [`dp_sgd_transformer.py`](https://github.com/google-parfait/jax_privacy/tree/main/examples/dp_sgd_transformer.py)
    and
    [`dp_logistic_regression.py`](https://github.com/google-parfait/jax_privacy/tree/main/examples/dp_logistic_regression.py)
    for examples.

*   **Tier 3 (Core API)**: Users compose {mod}`~jax_privacy.batch_selection`,
    {mod}`~jax_privacy.clipping`, {mod}`~jax_privacy.noise_addition`, and
    {mod}`~jax_privacy.accounting` modules directly. **Any**
    {class}`~jax_privacy.batch_selection.BatchSelectionStrategy` can be used
    with **any** noise addition scheme and **any** `DpEvent` — including Monte
    Carlo accounting via {mod}`~jax_privacy.experimental.monte_carlo` for
    combinations where PLD/RDP accounting is not available. See
    [`jax_api_example.py`](https://github.com/google-parfait/jax_privacy/tree/main/examples/jax_api_example.py)
    and
    [`balls_in_bins_accounting.py`](https://github.com/google-parfait/jax_privacy/tree/main/examples/balls_in_bins_accounting.py)
    for examples.

---

## Summary table

The table below lists all batch selection strategies, both those implemented in
jax_privacy and those discussed in the literature. Support levels are:

*   ✅ **Supported**: Implemented in jax_privacy with accounting support.

*   📄 **Analyzed**: Privacy analysis exists in the literature but accounting is
    not yet implemented in jax_privacy.

*   ❓ **Not analyzed**: No known privacy analysis for this combination.

*   N/A: Not applicable (e.g., DP-SGD is a special case of DP-MF with the
    identity matrix, so these entries are simply not relevant rather than
    invalid).

Privacy accounting is done via the
[dp_accounting](https://github.com/google/differential-privacy/tree/main/python/dp_accounting)
package by consuming `DpEvent` objects. There are two main accountants: **PLD**
and **RDP**. PLD is generally tighter and also supports additional methods
beyond standard $(\varepsilon, \delta)$-DP:

*   `get_true_positive_rates(false_positive_rates)`: Upper bounds on TPR at
    given FPR values, using the full PLD (not just an $(\varepsilon, \delta)$
    summary). Important for membership inference analysis, especially at very
    low FPR.

*   `get_gdp_parameter_estimate()`: Estimates the $\mu$-GDP parameter from the
    PLD.

These methods are **PLD-only** — they are not available via the RDP accountant.
See the [Accounting support matrix](#accounting-support-matrix) below for
details on which `DpEvent` types support each accountant.

For supported combinations, the accounting approach is noted as **PLD**,
**RDP**, or **MC** (Monte Carlo, experimental).

<!-- mdformat off -->

| Strategy | jax_privacy class | DP-SGD | DP-MF | Accounting | Key papers |
|---|---|---|---|---|---|
| Poisson sampling | {class}`~jax_privacy.batch_selection.CyclicPoissonSampling` (cycle_length=1) | ✅ | ✅ (Tier 2 via {class}`~jax_privacy.execution_plan.BandMFConfig`) | PLD via {func}`~jax_privacy.accounting.dpsgd_event` / {func}`~jax_privacy.accounting.amplified_bandmf_event` | [Abadi+ 2016](#abadi-2016), [Mironov+ 2019](#mironov-2019), [Liew+ 2022](#liew-2022) |
| Cyclic Poisson (BandMF) | {class}`~jax_privacy.batch_selection.CyclicPoissonSampling` (cycle_length > 1) | N/A | ✅ (Tier 2 via {class}`~jax_privacy.execution_plan.BandMFConfig`) | PLD via {func}`~jax_privacy.accounting.amplified_bandmf_event` | [Choquette-Choo+ 2023](#choquette-choo-2023), [McKenna 2024](#mckenna-2024) |
| Truncated Poisson | {class}`~jax_privacy.batch_selection.CyclicPoissonSampling` (truncated_batch_size set) | ✅ | ✅ (Tier 2 via {class}`~jax_privacy.execution_plan.BandMFConfig`) | PLD via {func}`~jax_privacy.accounting.truncated_dpsgd_event` / {func}`~jax_privacy.accounting.truncated_amplified_bandmf_event` | [Chua+ 2024a](#chua-2024a), [Ganesh 2025](#ganesh-2025) |
| Fixed-order + multi-epoch | {class}`~jax_privacy.batch_selection.CyclicPoissonSampling` (sampling_prob=1) | N/A | ✅ (Tier 2 via {class}`~jax_privacy.execution_plan.BandMFConfig`) | PLD via {func}`~jax_privacy.accounting.amplified_bandmf_event` | [Choquette-Choo+ 2022](#choquette-choo-2022) |
| Fixed-size (with replacement) | {class}`~jax_privacy.batch_selection.FixedBatchSampling` (replace=True) | ✅ | ❓ | DpEvent created but no accountant support | [Balle+ 2018](#balle-2018) |
| Fixed-size (without replacement) | {class}`~jax_privacy.batch_selection.FixedBatchSampling` (replace=False) | ✅ | ❓ | RDP via {func}`~jax_privacy.accounting.fixed_dpsgd_event` | [Balle+ 2018](#balle-2018) |
| Balls-in-bins | {class}`~jax_privacy.batch_selection.BallsInBinsSampling` | 📄 | ✅ (MC) | PLD for $k=1$ via {func}`~jax_privacy.accounting.random_allocation_dpsgd_event`; MC for DP-MF via `balls_in_bins_accounting.py` | [Choquette-Choo+ 2024](#choquette-choo-2024), [Chua+ 2024b](#chua-2024b) |
| Random allocation ($k$-out-of-$t$), a.k.a. Balanced Iteration Subsampling (BIS) | {class}`~jax_privacy.batch_selection.RandomAllocationSampling` (cycle_length=1) | ✅ | 📄 | PLD via {func}`~jax_privacy.accounting.random_allocation_dpsgd_event` | [Feldman+ 2025](#feldman-2025), [Dong+ 2025](#dong-2025), [Dong+ 2026b](#dong-2026b), [Schuchardt+ 2026](#schuchardt-2026), [Feldman+ 2026](#feldman-2026) |
| Cyclic random allocation (BandMF) | {class}`~jax_privacy.batch_selection.RandomAllocationSampling` (cycle_length > 1) | N/A | ✅ (Tier 2 via {class}`~jax_privacy.execution_plan.BandMFConfig`) | PLD via {func}`~jax_privacy.accounting.random_allocation_bandmf_event` | [Choquette-Choo+ 2023](#choquette-choo-2023), [Feldman+ 2025](#feldman-2025), [Feldman+ 2026](#feldman-2026) |
| $b$-min-sep ($b > 1$) | {class}`~jax_privacy.batch_selection.BMinSepSampling` | 📄 | ✅ (MC) | MC via {func}`experimental.monte_carlo <jax_privacy.experimental.monte_carlo.sample_generation.generate_sample>` | [Dong+ 2026a](#dong-2026a) |
| User-level wrapper | {class}`~jax_privacy.batch_selection.UserSelectionStrategy` | ✅ | ✅ | Inherits from base strategy | — |
| Shuffling | {class}`~jax_privacy.batch_selection.CyclicPoissonSampling` (partition_type=EQUAL_SPLIT) | 📄 | ❓ | No accounting support | [Chua+ 2024a](#chua-2024a) |
| Multi-owner $b$-min-sep | {class}`~jax_privacy.batch_selection.MultiOwnerMinSepSampling` | 📄 | 📄 | None implemented | [Ganesh+ 2025](#ganesh-2025-multi-user), [Dong+ 2026a](#dong-2026a) |

<!-- mdformat on -->

### Key observations

1.  **Poisson + Cyclic Poisson via
    {class}`~jax_privacy.batch_selection.CyclicPoissonSampling`** is the most
    versatile strategy class. With different parameter settings it covers
    standard Poisson (`cycle_length=1`), BandMF-style cyclic partitioning
    (`cycle_length > 1`), fixed-order multi-epoch (`sampling_prob=1`), and
    truncated variants (`truncated_batch_size` set). All of these have PLD
    accounting, meaning `get_true_positive_rates` and
    `get_gdp_parameter_estimate` are available.

2.  **Balls-in-bins** is a special case of
    {class}`~jax_privacy.batch_selection.CyclicPoissonSampling` with
    `partition_type=INDEPENDENT` and `sampling_prob=1` (included as an alias
    for better discoverability). It has an example of Monte Carlo accounting
    with DP-MF, but no PLD/RDP accounting. The near-exact privacy
    amplification framework of
    [Choquette-Choo+ 2024](#choquette-choo-2024)
    provides the theoretical foundation.

3.  **$b$-min-sep** is implemented in jax_privacy with Monte Carlo accounting
    support via {mod}`~jax_privacy.experimental.monte_carlo`
    ([Dong+ 2026a](#dong-2026a)).

4.  **Random allocation** is supported in `dp_accounting` via
    `RandomAllocationDpEvent` (implementing [Feldman+ 2026](#feldman-2026)),
    with exact PLD accounting via {class}`~dp_accounting.pld.PLDAccountant`.
    In jax_privacy the batch selection strategy is
    {class}`~jax_privacy.batch_selection.RandomAllocationSampling`, whose
    `cycle_length` parameter selects between the two accounting paths:
    {func}`~jax_privacy.accounting.random_allocation_dpsgd_event` for
    `cycle_length=1`, and
    {func}`~jax_privacy.accounting.random_allocation_bandmf_event` for
    `cycle_length > 1` with a banded strategy matrix. Both are configured
    end-to-end via {class}`~jax_privacy.execution_plan.BandMFConfig` with
    `sub_strategy=CyclicInnerSamplingStrategy.RANDOM_ALLOCATION`.

5.  **Shuffling** is the most common batch selection strategy in practice but
    has a *strictly worse* privacy guarantee than Poisson subsampling
    ([Chua+ 2024a](#chua-2024a)). It can be parameterized via
    {class}`~jax_privacy.batch_selection.CyclicPoissonSampling` with
    `partition_type=EQUAL_SPLIT`, but has no accounting support. Use Poisson
    or truncated Poisson sampling instead.

---

(accounting-support-matrix)=
## Accounting support matrix

The table below maps
[DpEvent types](https://github.com/google/differential-privacy/blob/main/python/dp_accounting/dp_accounting/dp_event.py)
to accountant support and available privacy metrics.

<!-- mdformat off -->

| DpEvent type | PLD | RDP | Neighboring relation | `get_epsilon` | `get_true_positive_rates` | `get_gdp_parameter_estimate` |
|---|---|---|---|---|---|---|
| `PoissonSampledDpEvent(GaussianDpEvent)` | ✅ | ✅ | ADD_OR_REMOVE_ONE, REPLACE_SPECIAL | ✅ (both) | ✅ (PLD) | ✅ (PLD) |
| `TruncatedSubsampledGaussianDpEvent` | ✅ | ❌ | REPLACE_ONE, REPLACE_SPECIAL | ✅ (PLD) | ✅ (PLD) | ✅ (PLD) |
| `RandomAllocationDpEvent(GaussianDpEvent)` | ✅ | ❌ | ADD_OR_REMOVE_ONE | ✅ (PLD) | ✅ (PLD) | ✅ (PLD) |
| `SampledWithoutReplacementDpEvent` | ❌ | ✅ | REPLACE_ONE only | ✅ (RDP) | ❌ | ❌ |
| `SampledWithReplacementDpEvent` | ❌ | ❌ | — | ❌ | ❌ | ❌ |
| `MixtureOfGaussiansDpEvent` | ✅ | ❌ | ADD_OR_REMOVE_ONE, REPLACE_SPECIAL | ✅ (PLD) | ✅ (PLD) | ✅ (PLD) |

<!-- mdformat on -->

---

## Research appendix

(abadi-2016)=
### Abadi+ 2016: Deep Learning with Differential Privacy

*   **Paper**: [Abadi, Chu, Goodfellow, McMahan, Mironov, Talwar, Zhang, 2016](https://arxiv.org/abs/1607.00133)

*   **Strategy**: Poisson subsampling — each example is included independently
    with probability $q = B/N$ (batch_size / dataset_size). **Algorithm 1**
    defines DP-SGD with Poisson subsampling. **Section 3** introduces the
    Moments Accountant for tracking cumulative privacy loss under subsampled
    mechanisms.

*   **MF compatibility**: Not discussed (independent noise, $C = I$).

*   **dp_accounting support**: `PoissonSampledDpEvent(GaussianDpEvent)` in PLD
    and RDP accountants (the RDP accountant generalizes and improves on the
    moments accountant introduced in this work). Full support for TPR/GDP via
    PLD.

*   **jax_privacy support**: ✅
    {class}`~jax_privacy.batch_selection.CyclicPoissonSampling` `(cycle_length=1)`
    and {func}`~jax_privacy.accounting.dpsgd_event`.

(balle-2018)=
### Balle+ 2018: Privacy Amplification by Subsampling: Tight Analyses via Couplings and Divergences

*   **Paper**: [Balle, Barthe, and Gaboardi, 2018](https://arxiv.org/abs/1807.01647)

*   **Strategy**: Unified framework for analyzing privacy amplification across
    subsampling methods (for choosing a single batch). **Theorem 8** covers
    Poisson sampling, **Theorem 9** covers sampling fixed-size batches without
    replacement, and **Theorem 10** covers sampling fixed-size batches with
    replacement. **Section 3** develops the analytical tools (couplings,
    divergences, privacy profiles) and **Section 4** derives the main
    amplification bounds.

*   **MF compatibility**: Not discussed.

*   **dp_accounting support**: Full support for Poisson sampling.
    `SampledWithoutReplacementDpEvent` — RDP only, REPLACE_ONE only. No PLD
    support, no TPR/GDP.

*   **jax_privacy support**: ✅

    *   Poisson sampling: Full support.

    *   Without replacement:
        {class}`~jax_privacy.batch_selection.FixedBatchSampling`
        `(replace=False)` and
        {func}`~jax_privacy.accounting.fixed_dpsgd_event`, RDP accounting only.

    *   With replacement:
        {class}`~jax_privacy.batch_selection.FixedBatchSampling`
        `(replace=True)`, no accounting support.

(mironov-2019)=
### Mironov+ 2019: Rényi Differential Privacy of the Sampled Gaussian Mechanism

*   **Paper**: [Mironov, Talwar, Zhang, 2019](https://arxiv.org/abs/1908.10530)

*   **Strategy**: Characterizes RDP for the Sampled Gaussian Mechanism
    (composition of Poisson subsampling + Gaussian noise). **Theorem 4**
    reduces the multi-dimensional case to 1D Gaussian mixtures. **Theorem 11**
    gives a closed-form bound: $\varepsilon = 2q^2\alpha/\sigma^2$ under
    conditions on $q$, $\sigma$, and $\alpha$. **Section 3.3** provides a
    numerically stable procedure for exact RDP computation.

*   **MF compatibility**: Not discussed.

*   **dp_accounting support**: Core reference for the RDP accountant
    implementation of `PoissonSampledDpEvent(GaussianDpEvent)`.

*   **jax_privacy support**: ✅ Full support for Poisson sampling.

(liew-2022)=
### Liew+ 2022: Privacy Amplification via Shuffled Check-Ins

*   **Paper**: [Liew, Hasegawa, Takahashi, 2022](https://arxiv.org/abs/2206.03151)

*   **Strategy**: Shuffled check-in protocol where clients independently decide
    to participate in each iteration/batch with probability $\gamma$ (Poisson
    sampling), after which the reports are shuffled. This in-batch shuffling
    provides additional protections against an adversarial aggregator, but does
    not influence the DP guarantee for the released model. **Section 3**
    defines the protocol (Section 3.1), provides the RDP analysis (Section 3.2,
    Theorem 2), and introduces a numerical method for tracking privacy of
    Gaussian mechanisms under shuffling (Section 3.3). The full protocol is
    Algorithm 1 in Appendix C. **Section 4** derives a lower bound for the
    Gaussian mechanism.

*   **MF compatibility**: Not discussed.

*   **dp_accounting support**: Not implemented.

*   **jax_privacy support**: ✅ Full support for Poisson sampling, but
    shuffling and distributed computation are currently out of scope for
    jax_privacy (though could be built on top).

(choquette-choo-2022)=
### Choquette-Choo+ 2022: Multi-Epoch Matrix Factorization Mechanisms for Private Machine Learning

*   **Paper**: [Choquette-Choo et al., 2022](https://arxiv.org/abs/2211.06530)

*   **Strategy**: Introduces $(k, b)$-participation schema where each example
    participates at most $k$ times with spacing $b$ between participations.
    **Section 2** formalizes $(k, b)$-participation and defines sensitivity for
    multi-participation adaptive streams. **Section 3** covers optimal matrix
    factorization for multiple epochs.

*   **MF compatibility**: ✅ Core paper for combining fixed-order sampling with
    banded Toeplitz matrices.

*   **dp_accounting support**: Yes: the mechanism is equivalent to a single
    application of the Gaussian mechanism.

*   **jax_privacy support**: The
    {func}`~jax_privacy.matrix_factorization.dense.optimize` function supports
    optimizing $(k, b)$-participation strategy matrices by setting `epochs=k`.
    However, for centralized training applications where sampling is possible,
    newer amplified approaches are generally preferred.

(choquette-choo-2023)=
### Choquette-Choo+ 2023: (Amplified) Banded Matrix Factorization: A Unified Approach to Private Training

*   **Paper**: [Choquette-Choo et al., 2023](https://arxiv.org/abs/2306.08153)

*   **Strategy**: Introduces the cyclic Poisson sampling scheme for BandMF.
    **Section 3** defines the $b$-min-sep participation schema for FL.
    **Section 4** covers optimization of banded matrices. **Section 5**
    introduces cyclic Poisson subsampling (Algorithms 2 and 6) and proves the
    privacy amplification result (**Theorem 1** is the informal statement):
    $b$-banded BandMF on a dataset of $m$ examples run for $n$ iterations
    achieves the same privacy as DP-SGD with $n/b$ iterations and Poisson
    sampling probability $\text{batch\_size} \cdot b / m$. Theorems 4 and 5
    give the formal results. The formal results are modular and do not
    reference Poisson sampling directly. Roughly, they say the privacy cost can
    be computed by only looking at $k$ applications of an arbitrary mechanism
    to a dataset of size $m / b$. Using Poisson sampling as the chosen mechanism
    then yields the main result. The standard analysis of DP-SGD with Poisson
    sampling is a special case of this result with $b=1$.

*   **MF compatibility**: ✅ Central paper, introducing $b$-banded strategy
    matrices $C$.

*   **dp_accounting support**: `PoissonSampledDpEvent(GaussianDpEvent)` via
    reduction to per-group independent Poisson DP-SGD.

*   **jax_privacy support**: ✅
    {class}`~jax_privacy.batch_selection.CyclicPoissonSampling`
    `(cycle_length > 1)` and
    {func}`~jax_privacy.accounting.amplified_bandmf_event` +
    {class}`~jax_privacy.execution_plan.BandMFConfig` (Tier 2).

(mckenna-2024)=
### McKenna 2024: Scaling up the Banded Matrix Factorization Mechanism for Differentially Private ML

*   **Paper**: [McKenna, 2024](https://arxiv.org/abs/2405.15913)

*   **Strategy**: Scales up DP-BandMF
    ([Choquette-Choo+ 2023](#choquette-choo-2023)) along two axes:

    1.  *strategy optimization* (**Sections 3.1 and 3.2**): more efficient
        algorithms for optimizing general and Toeplitz banded strategy matrices;

    2.  *distributed noise generation* (**Section 3.3**): shards correlated
        noise across accelerators, enabling support for large models with
        negligible overhead. Does not change the batch selection strategy
        (inherits cyclic Poisson sampling) or the privacy analysis which
        follows from [Choquette-Choo+ 2023](#choquette-choo-2023).

*   **MF compatibility**: ✅ Central focus. Scalable optimization of banded
    Toeplitz strategy matrices and distributed noise generation for DP-BandMF.

*   **dp_accounting support**: Same as
    [Choquette-Choo+ 2023](#choquette-choo-2023):
    `PoissonSampledDpEvent(GaussianDpEvent)` via reduction to amplified DP-SGD.

*   **jax_privacy support**: ✅
    {meth}`BandMFConfig.default() <jax_privacy.execution_plan.BandMFConfig.default>`
    uses
    {func}`toeplitz.optimize_banded_toeplitz() <jax_privacy.matrix_factorization.toeplitz.optimize_banded_toeplitz>`,
    and `sharding_utils.py` implements the distributed noise generation.

### Beltran+ 2024: Efficient and Scalable Implementation of Differentially Private Deep Learning without Shortcuts

*   **Paper**: [Beltran et al., 2024](https://arxiv.org/abs/2406.17298)

*   **Strategy**: Addresses efficient implementation of Poisson subsampling in
    JAX. **Section 3** proposes "masked DP-SGD" that avoids recompilation by
    rounding up the batch size and masking extra gradients. **Lemma 1** proves
    that decomposing Poisson subsampling into a Binomial draw + WOR is
    equivalent to standard Poisson subsampling. Referenced for implementation
    correctness in
    {class}`~jax_privacy.batch_selection.CyclicPoissonSampling`.

*   **MF compatibility**: N/A.

*   **dp_accounting support**: N/A.

*   **jax_privacy support**: Masked DP-SGD is implemented, see the jax_privacy
    documentation on [Handling Variable Batch Sizes](sharp_edges_variable_batch_sizes);
    jax_privacy generally refers to this as "padding" batches to a fixed size
    rather than "masking".

(choquette-choo-2024)=
### Choquette-Choo+ 2024: Near Exact Privacy Amplification for Matrix Mechanisms

*   **Paper**: [Choquette-Choo et al., 2024](https://arxiv.org/abs/2410.06266)

*   **Strategy**: Balls-in-bins sampling — each example is independently
    assigned to a "bin" uniformly at random. **Section 2** introduces Monte
    Carlo accounting via PLD and the Estimate-Verify-Release framework.
    **Section 3** defines balls-in-bins batching (**Definition 3.1**) and
    provides the dominating pair via **Lemma 3.2** (dimension reduction).
    **Section 4** describes how to do Monte Carlo accounting for correlated
    noise (matrix factorization) with balls-in-bins. Enables joint optimization
    of correlation matrix $C$ with amplification for arbitrary lower-triangular
    non-negative $C$.

*   **MF compatibility**: ✅ Core paper for combining balls-in-bins with
    arbitrary matrix mechanisms (not limited to banded matrices).

*   **dp_accounting support**: Near-exact privacy parameters computable, but
    not integrated into dp_accounting. Monte Carlo accounting available via
    {mod}`~jax_privacy.experimental.monte_carlo`.

*   **jax_privacy support**: ✅ Batch selection via
    {class}`~jax_privacy.batch_selection.BallsInBinsSampling`. Accounting via
    Monte Carlo (see
    [`balls_in_bins_accounting.py`](https://github.com/google-parfait/jax_privacy/tree/main/examples/balls_in_bins_accounting.py)).

(chua-2024a)=
### Chua+ 2024a: Scalable DP-SGD: Shuffling vs. Poisson Subsampling

*   **Paper**: [Chua et al., 2024](https://arxiv.org/abs/2411.04205)

*   **Main contribution**: Establishes the privacy gap between shuffling and
    Poisson subsampling.

*   **MF compatibility**: Gives a reduction from the privacy of any
    mechanism applied to a sampled and truncated batch to the privacy of the
    mechanism on the untruncated batch, including BandMF with Cyclic Poisson.

*   **dp_accounting support**: `TruncatedSubsampledGaussianDpEvent` in PLD
    accountant.

*   **jax_privacy support**: ✅
    {class}`~jax_privacy.batch_selection.CyclicPoissonSampling`
    `(truncated_batch_size=B)` and
    {func}`~jax_privacy.accounting.truncated_dpsgd_event`.

*   **Additional strategies from this paper not in jax_privacy**: Shuffling
    (not implemented; strictly worse privacy than Poisson).

(chua-2024b)=
### Chua+ 2024b: Balls-and-Bins Sampling for DP-SGD

*   **Paper**: [Chua et al., 2024](https://arxiv.org/abs/2412.16802)

*   **Strategy**: Balls-and-bins as in
    [Choquette-Choo+ 2024](#choquette-choo-2024). **Section 3** defines the
    balls-in-bins sampling scheme. **Section 4** provides the privacy analysis
    via Monte Carlo accounting (estimating hockey-stick divergence). Shows this
    achieves privacy amplification comparable to Poisson subsampling while being
    practical like shuffling.

*   **MF compatibility**: Focused on DP-SGD (independent noise). MF extension
    is via [Choquette-Choo+ 2024](#choquette-choo-2024).

*   **dp_accounting and jax_privacy support**: ✅ See
    [Choquette-Choo+ 2024](#choquette-choo-2024).

(feldman-2025)=
### Feldman+ 2025: Privacy Amplification by Random Allocation

*   **Paper**: [Feldman & Shenfeld, 2025](https://arxiv.org/abs/2502.08202v4)

*   **Strategy**: Provides theoretical (RDP) and numerical privacy analysis for
    $k$-out-of-$t$ random allocation — user's data is used in $k$ steps chosen
    uniformly at random from $t$ total steps. The balls-in-bins sampling scheme
    is a special case with $k=1$.

*   **MF compatibility**: Not addressed.

*   **dp_accounting support**: Not integrated.

*   **jax_privacy support**: ✅ Batch selection via
    {class}`~jax_privacy.batch_selection.RandomAllocationSampling`.
    Accounting not implemented.

(dong-2025)=
### Dong+ 2025: Leveraging Randomness in Model and Data Partitioning for Privacy Amplification

*   **Paper**: [Dong, Chen, Ozgur, 2025](https://arxiv.org/abs/2503.03043)

*   **Strategy**: Concurrently introduced random allocation, which they call
    Balanced Iteration Subsampling (BIS, **Section 3.3**): each sample
    participates in exactly $k$ iterations out of $t$. **Section 3.1** provides
    the core RDP bound (Theorem 3.1) for the dominating pair.

*   **MF compatibility**: Not discussed.

*   **dp_accounting and jax_privacy support**: See
    [Feldman+ 2025](#feldman-2025).

(ganesh-2025-multi-user)=
### Ganesh+ 2025: It's My Data Too: Private ML for Datasets with Multi-User Training Examples

*   **Paper**: [Ganesh, McKenna, McMahan, Smith, Wu, 2025](https://arxiv.org/abs/2503.03622)

*   **Strategy**: Multi-attribution model where each example may be associated
    with multiple users. **Section 2** defines fixed-graph DP for
    multi-attribution. **Section 3** discusses batch selection strategies for
    multi-attribution DP-SGD/DP-MF: Poisson sampling with group privacy (Section
    3.2), $(k, b)$-min-sep with BandMF, and cyclic Poisson. **Section 4**
    proposes greedy algorithms for contribution bounding.

*   **MF compatibility**: Discussed in context of multi-owner DP-SGD.

*   **dp_accounting support**: For DP-SGD with Poisson sampling, the analysis
    can be reduced to group privacy which is supported by the
    `MixtureOfGaussiansDpEvent`, but currently jax_privacy does not provide a
    utility to do this easily. With a contribution bound of 1, then you get user
    to example reduction and all example-level batch selection strategies
    immediately apply at the user level. Unamplified DP-MF using
    $b$-min-separation is also immediately supported, as the mechanism behaves
    as a single application of the Gaussian mechanism.

*   **jax_privacy support**: ✅ Batch selection via
    {class}`~jax_privacy.batch_selection.MultiOwnerMinSepSampling`,
    {class}`~jax_privacy.batch_selection.MultiOwnerGraph`, and
    {func}`~jax_privacy.batch_selection.greedy_contribution_bound` (see
    `_multi_owner.py`). Accounting not implemented.

(pillutla-2025)=
### Pillutla+ 2025: Correlated Noise Mechanisms for Differentially Private Learning

*   **Paper**: [Pillutla et al., 2025](https://arxiv.org/abs/2506.08201)

*   **Main Contribution**: Comprehensive monograph on correlated noise
    mechanisms (matrix mechanisms / factorization mechanisms / DP-FTRL). Not a
    batch selection paper per se, but foundational for the noise addition side
    of DP-MF. Covers $(k, b)$-participation schemas, block-cyclic Poisson
    subsampling, cyclic participation, $b$-min-sep, and other batch selection
    methods as they relate to correlated noise mechanisms.

*   **MF compatibility**: ✅ This is the central focus.

*   **dp_accounting support**: General framework; specific `DpEvent` types
    depend on the batch selection strategy used.

*   **jax_privacy support**: ✅ Many mechanisms implemented in the
    {mod}`~jax_privacy.matrix_factorization` module and supported by
    {mod}`~jax_privacy.noise_addition`.

(ganesh-2025)=
### Ganesh 2025: Tighter Privacy Analysis for Truncated Poisson Sampling

*   **Paper**: [Ganesh, 2025](https://arxiv.org/abs/2508.15089)

*   **Strategy**: Truncated Poisson — Poisson sampling with probability $p$,
    truncated to maximum batch size $B$, with random subselection if
    $|\text{sampled}| > B$. **Section 2** defines the mechanism formally.
    **Section 3** provides the privacy analysis, which is tighter than
    absorbing truncation probability into $\delta$ as in
    [Chua+ 2024a](#chua-2024a). Generalizes to adaptive vector queries via
    post-processing and composition.

*   **MF compatibility**: Analysis extends to DP-MF via
    {func}`~jax_privacy.accounting.truncated_amplified_bandmf_event`.

*   **dp_accounting and jax_privacy support**: ✅ See
    [Chua+ 2024a](#chua-2024a).

(schuchardt-2026)=
### Schuchardt+ 2026: Sampling-Free Privacy Accounting for Matrix Mechanisms under Random Allocation

*   **Paper**: [Schuchardt & Kalinin, 2026](https://arxiv.org/abs/2601.21636)

*   **Strategy**: Random allocation with matrix mechanisms. Develops
    sampling-free (as opposed to Monte Carlo) privacy bounds via Rényi
    divergence (Section 3) and Conditional Composition (Section 4). **Lemma
    3.2** computes the Rényi divergence for random-allocation matrix mechanisms;
    it is generally intractable, but Algorithm 3 provides a dynamic programming
    approach that can be practical for $p$-banded strategy matrices $C$ when $p$
    is relatively small (Lemma 3.3).

*   **MF compatibility**: ✅ Central focus.

*   **dp_accounting support**: Not integrated.

*   **jax_privacy support**: Referenced in
    {class}`~jax_privacy.batch_selection.RandomAllocationSampling` docstring.
    Integration could be useful for random allocation + DP-MF users, though the
    tighter method's privacy analysis has runtime exponential in the number of
    bands.

(dong-2026a)=
### Dong+ 2026a: Privacy Amplification for BandMF via $b$-Min-Sep Subsampling

*   **Paper**: [Dong & Ganesh, 2026](https://arxiv.org/abs/2602.09338)

*   **Strategy**: $b$-min-sep subsampling selects each example that has not
    participated in the past $b$ iterations with a fixed Poisson sampling
    probability, ensuring any given example does not participate more than once
    within any window of $b$ iterations. Standard Poisson sampling corresponds
    to $b=1$, and balls-in-bins corresponds to (warm-start) $b$-min-sep with
    sampling probability $p=1$. **Section 4** defines "pure" $b$-min-sep
    subsampling in Algorithm 1 (which suffers from overly large batches
    initially, especially if $p$ is large), and the practically preferred
    warm-start variation that ensures all iterations have the same expected
    batch size (Algorithm 2). **Section 5** presents Monte Carlo accounting
    exploiting Markovian structure via dynamic programming. **Section 6**
    discusses multi-attribution extension and **Appendix B** covers extensions
    to non-banded matrices.

*   **MF compatibility**: ✅ The primary motivation is enabling improved privacy
    amplification for BandMF.

*   **dp_accounting support**: Not integrated; Monte Carlo accounting only.

*   **jax_privacy support**: ✅ Batch selection via
    {class}`~jax_privacy.batch_selection.BMinSepSampling`. Accounting via
    {mod}`~jax_privacy.experimental.monte_carlo`.

(feldman-2026)=
### Feldman+ 2026: Efficient Privacy Loss Accounting for Subsampling and Random Allocation

*   **Paper**: [Feldman & Shenfeld, 2026](https://arxiv.org/abs/2602.17284)

*   **Strategy**: Demonstrates that the PLD of random allocation applied to any
    DP algorithm can be computed efficiently. Introduces **PLD realization** as
    a new accounting tool (Section 3). **Theorem 4.6** provides the efficient
    algorithm. The results show a privacy-utility tradeoff for random allocation
    is at least as good as Poisson subsampling.

*   **MF compatibility**: Not directly addressed.

*   **dp_accounting support**: ✅ Supported via
    `dp_accounting.dp_event.RandomAllocationDpEvent` and
    {class}`~dp_accounting.pld.PLDAccountant`. Computes exact PLD bounds for
    $k$-out-of-$t$ allocation with Gaussian noise based on Theorem 4.6.

*   **jax_privacy support**: ✅ Batch selection supported via
    {class}`~jax_privacy.batch_selection.RandomAllocationSampling` (with
    `cycle_length > 1` for banded matrix mechanisms), accounting via
    {func}`~jax_privacy.accounting.random_allocation_dpsgd_event` and
    {func}`~jax_privacy.accounting.random_allocation_bandmf_event`, and
    execution plans via {class}`~jax_privacy.execution_plan.BandMFConfig` with
    `sub_strategy=CyclicInnerSamplingStrategy.RANDOM_ALLOCATION`.

(dong-2026b)=
### Dong+ 2026b: Less Random, More Private: What is the Optimal Subsampling Scheme for DP-SGD?

*   **Paper**: [Dong & Özgür, 2026](https://arxiv.org/abs/2605.07072)

*   **Strategy**: Proposes random allocation (which they call Balanced
    Iteration Subsampling (BIS), see also
    [Feldman+ 2025](#feldman-2025), [Dong+ 2025](#dong-2025)) as an optimal
    alternative to Poisson subsampling. **Algorithm 1** formally defines BIS.
    **Section 3** proves BIS is optimal at both $\sigma \to 0$ and $\sigma \to
    \infty$ extremes via a layered analysis. **Section 4** introduces a
    near-exact Monte Carlo accountant. Shows up to 9.6% reduction in required
    noise multiplier vs. Poisson.

*   **MF compatibility**: Not directly addressed.

*   **dp_accounting support**: Not integrated.

*   **jax_privacy support**: ✅ Batch selection via
    {class}`~jax_privacy.batch_selection.RandomAllocationSampling`.
    Accounting not implemented.

---

## Future work

The following combinations are analyzed in the academic literature but not yet
fully supported in jax_privacy, and represent good candidates for future
integration:

*   **Random allocation ($k$-out-of-$t$) + DP-MF / matrix mechanisms**:
    `dp_accounting` supports $k$-out-of-$t$ random allocation for standard
    DP-SGD (Gaussian noise) via `RandomAllocationDpEvent`, and jax_privacy
    extends this to $b$-banded matrix mechanisms via the `cycle_length`
    parameter of
    {class}`~jax_privacy.batch_selection.RandomAllocationSampling`, which
    enforces a minimum separation of $b$ so that the mechanism reduces to
    random allocation of a single Gaussian mechanism. An open frontier is
    sampling-free accounting for random allocation combined with general
    (non-banded) matrix mechanisms, and tighter analyses of the banded case, as
    studied by [Schuchardt+ 2026](#schuchardt-2026). In addition,
    [Dong+ 2026b](#dong-2026b) provides a Monte Carlo accounting scheme for
    $k \ge 1$ that could be added to jax_privacy's Monte Carlo estimation
    libraries.

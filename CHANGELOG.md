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

# Changelog

All notable changes to this project will be documented in this file.

The format is based on https://keepachangelog.com/en/1.1.0/

## [Unreleased]

### Added

-   **[`DPTrainer`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.training.html)
    precompilation and auto-tuning**: Added `init()` and `precompile()`
    ahead-of-time compilation with batch-size deduping, and automated
    `microbatch_size` tuning to maximize throughput and optimize device memory
    during DP-SGD training, while minimizing compilation overhead and manual
    tuning of performance knobs.
-   **[Monte Carlo (MC) accounting](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.accounting.html)**:
    Added cyclic Poisson sampling (with independent partitions) to the MC
    accounting library, and added support for hybrid mechanism calibration
    composing events analyzed via MC accounting with events analyzed via
    Privacy Loss Distributions (PLD).
-   **Slack-based adaptive clipping**: Added optional `slack` argument to
    [`clipped_fun`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.clipping.html)
    and
    [`clipped_grad`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.clipping.html)
    to output a sensitivity-bounded slack indicator from the gradient norm,
    enabling downstream adaptive clipping-bound controllers. This is a building
    block for the [SLAClip](https://icml.cc/virtual/2026/poster/66390)
    approach. ([#343](https://github.com/google-parfait/jax_privacy/pull/343))
-   **Documentation improvements**:
    -   **API documentation rendering**: Greatly improved documentation across
        `.md`, `.rst`, and `.py` files for better rendering of math, code, and
        tables on
        [jax-privacy.readthedocs.io](https://jax-privacy.readthedocs.io/).
    -   **DP pitfalls guide**: Added a comprehensive guide on
        [Common Pitfalls in DP Training](https://jax-privacy.readthedocs.io/en/latest/sharp_edges_dp_training_pitfalls.html)
        covering pitfalls across data loading, clipping, composition, and model
        architecture, and explaining how the design of `jax_privacy` prevents
        or mitigates them.
        ([#362](https://github.com/google-parfait/jax_privacy/pull/362))
    -   **Batch selection**: Added the
        [Batch Selection Strategies guide](https://jax-privacy.readthedocs.io/en/latest/batch_selection.html)
        documenting supported batch selection strategies and their privacy
        accounting implications.
    -   **Contributor documentation guide**: Updated `CONTRIBUTING.md` and added
        a dedicated
        [Documentation Guide](https://jax-privacy.readthedocs.io/en/latest/contribution_guide.html#documentation-guide)
        establishing ReadTheDocs rendering quality as a primary priority for
        Python docstrings and Markdown documentation, complete with local
        preview workflows.
    -   **Matrix factorization notation page**: Moved the notation and
        participation-schema tables from
        `jax_privacy/matrix_factorization/README.md` to a
        [Matrix Factorization notation](https://jax-privacy.readthedocs.io/en/latest/matrix_factorization.html)
        page, and linked that page from the generated Public API docs.
        ([#198](https://github.com/google-parfait/jax_privacy/issues/198))
    -   **Sphinx intersphinx mappings**: Added cross-reference mappings for
        `jax` and `optax` for native cross-references in Sphinx documentation.

### Fixed

-   **[`DPTrainer.fit`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.training.html)
    preemption resume**: Fixed an issue where resuming from preemption did not
    advance the `batch_iterator` to match the restored `TrainingState`,
    correctly synchronizing the data stream.
-   **[`clipped_fun`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.clipping.html)
    Formal Guarantees**: Replaced the incorrect claim that sensitivity is
    always `1.0` / `l2_clip_norm` with guidance to use `.sensitivity()` /
    `.l2_norm_bound`, matching `clipped_grad`. The old text ignored
    `normalize_by` and per-layer clipping. Added contract tests.

### Internal / Cleanups

-   **Dependency management with `uv` and `pylock.toml`**: Migrated lockfile
    and development workflow to PEP 751 `pylock.toml` using `uv`, enforced via
    GitHub Actions.
-   **Removed `pytype` from CI**: Removed `pytype` checks from CI workflows.

## [2.2.0] - 2026-06-15

### Added

-   **Multi-owner DP support**: Added `greedy_contribution_bound` and
    `MultiOwnerGraph` dataclass for multi-owner differential privacy based on
    [arXiv:2503.03622](https://arxiv.org/abs/2503.03622).
-   **Banded-inverse matrix factorization methods**: New strategies for
    `BandInvMF` and `BISR` including
    `banded_inverse_square_root_noising_coefs()`,
    `compute_banded_inverse_sensitivity()`, and
    `optimize_banded_inverse_toeplitz()`. Based on
    [arXiv:2505.12128](https://arxiv.org/abs/2505.12128).
    ([#218](https://github.com/google-parfait/jax_privacy/pull/218))
-   **Experimental `pre_clipping_transforms` API**: Allows optimizers to define
    custom pre-clipping transforms.
-   **Truncation support for b-min-sep sampling**: Added to both sample
    generation and privacy loss computation.
-   **Keras Poisson opt-in example**: Demonstrates Poisson sampling support
    using the Keras API.
    ([#196](https://github.com/google-parfait/jax_privacy/pull/196))

### Changed

-   **Promoted experimental modules to public API**: Moved
    `experimental/accounting.py`, `experimental/execution_plan.py`, and
    `experimental/optimizers.py` to top-level `jax_privacy` modules.
-   **Faster Bernoulli sample generation**: Performance improvement to batch
    selection sampling.
-   **Documentation overhaul**:
    -   Rewrote installation docs to recommend GitHub head install and explain
        co-developed dependency requirements (DP Accounting, Optax).
    -   Simplified README.md by removing sections now covered by Read the Docs.
    -   Split `sharp_edges.md` into three focused pages: variable batch sizes,
        VMap + sharding, and mixed precision training.
    -   Updated documentation for `clipped_grad` and `batch_selection`.

### Fixed

-   **`clipped_fun` `return_norms` output contract**: Fixed incorrect return
    shape `(value, ((), norms))` → `(value, norms)` when `has_aux=False`.
    Regression test added.
    ([#224](https://github.com/google-parfait/jax_privacy/pull/224))
-   **`ZeroDivisionError` in zCDP accounting**: Fixed crash when
    `noise_multiplier=0` by guarding the division in the zCDP branch and
    returning `ZCDpEvent(inf)` (rho → ∞, i.e., no privacy). Includes regression
    test. ([#255](https://github.com/google-parfait/jax_privacy/pull/255))
-   **JAX v0.9.0 / NumPy 2.4.0 compatibility**: Fixed implicit JAX array to
    dtype conversion warnings that would become errors in future versions.
-   **Keras Poisson example regression**: Increased batch size 128→512, seeded
    Keras init for reproducibility, aligned Poisson sampling with accountant.
    ([#222](https://github.com/google-parfait/jax_privacy/pull/222))
-   **Flaky `test_dp_training_e2e_work`**: Seeded Keras/JAX RNG to eliminate
    test flakiness.
    ([#264](https://github.com/google-parfait/jax_privacy/pull/264))

### Internal / Cleanups

-   **Centralized `_validate` module**: Introduced `_validate.py` as the single
    source of truth for parameter validation, replacing pydantic-based
    validation. Provides generic validators (`non_negative`, `positive`,
    `in_range`) with no external dependency.
-   **Removed pydantic dependency**: Replaced by the centralized `_validate`
    module.
-   **Converted to relative imports**: All intra-package imports in core library
    modules now use Python relative imports (`from . import X`).
-   **Removed `skip_checks` args from `toeplitz.py`**: Extended
    `toeplitz.minsep_sensitivity_squared` to support potentially increasing
    toeplitz coefficients natively.
-   **Deleted `jax_privacy/accounting/` directory**: Removed in favor of the
    promoted top-level accounting module.

## [2.0.0] - 2026-04-16

Since version 1.0.0, the **jax-privacy** library has undergone a major
structural transformation, transitioning from a framework-coupled design to a
modular, extensible architecture. This includes a complete overhaul of batch
selection strategies, the introduction of experimental Monte Carlo accounting,
and expanded support for matrix factorization mechanisms.

### 1. API Changes

*   **Decoupling and Deprecation**: The monolithic `dp_sgd`,
    `training`, and `experiments` modules have been removed. The library now
    prioritizes decoupled modules like `clipping.py`, `noise_addition.py`, and
    `batch_selection.py`, which can be integrated into custom training loops
    without inheriting framework-specific dependencies.

### 2. Experimental Features

*   **DPExecutionPlan**: Introduced an experimental `DPExecutionPlan` that
    encapsulates the "three pillars" of a differentially private mechanism:
    batch selection, clipping, and noise addition. This allows for
    framework-agnostic execution of DP-SGD and other mechanisms.
*   **Monte Carlo Accounting**: A new experimental framework for privacy
    accounting using Monte Carlo simulations was added. This includes utilities
    for generating privacy loss distribution (PLD) samples and calculating
    overall deltas for complex mechanisms that are difficult to analyze
    analytically.

### 3. New Features, Components, and Mechanisms

*   **Expanded Batch Selection Strategies**: Batch Selection API: Reorganized
    the batch selection API to support diverse strategies, including fixed-size,
    cyclic Poisson, balls-in-bins, and b-min-sep sampling. It also includes a
    UserSelectionStrategy for user-level DP.
*   **Auditing Enhancements**:
    *   **One-Run Auditing**: Added support for "Auditing f-Differential Privacy
        in One Run," allowing empirical privacy estimation from a single
        training run.
    *   **Threshold Strategies**: Refactored `CanaryScoreAuditor` to support
        multiple threshold selection strategies, including `Split` and
        `MultiSplit` for epsilon auditing.
*   **Matrix Factorization and BandMF**: Expanded support for private matrix
    factorization with implementations for `Banded`, `Dense`, and `Toeplitz`
    matrices, including specific accountants for BandMF with Cyclic Poisson
    sampling.

### 4. Examples and Documentation

*   **Advanced Training Examples**:
    *   **Secure Noise**: Demonstrates the use of cryptographic-strength
        randomness for noise addition and batch selection.
    *   **Transformers and LLMs**: Added examples for training Transformers with
        DP-SGD and fine-tuning **Gemma3** using LoRA.
    *   **Synthetic Data**: Provided a notebook for using DP-SGD to generate
        synthetic data for Gemma3.
*   **Enhanced Documentation**:
    *   Added a "Library Design" document and a "Sharp Edges" guide covering
        pitfalls like mixed-precision training.
    *   Improved discoverability of batch selection strategies and their
        partitioning logic.

### 5. Bug Fixes

*   **Auxiliary Output Alignment**: Fixed a bug where auxiliary outputs had an
    incorrect extra dimension when `keep_batch_dim=True` was used; the
    implementation now correctly squeezes this artificial batch dimension.
*   **Mixed Precision Support**: Added an optional `dtype` argument to noise
    addition and clipping functions to ensure consistency during low-precision
    or mixed-precision training.
*   **Type Safety and Consistency**: `clip_pytree` was refactored to guarantee
    that the output dtype matches the input dtype, preventing issues where
    `jax.lax.cond` could produce inconsistent return types.

### 6. Internal Refactorings and Code Cleanups

*   **Optax Migration**: Internal microbatching utilities were migrated to the
    public `optax.microbatching` API to reduce redundancy, and the
    experimental `microbatching.py` file was subsequently removed.
*   **Unified Noise Addition**: Refactored the noise addition pipeline to
    support customizable intermediate sharding strategies and factored out
    general-purpose sharding utilities into dedicated files, improving the
    maintainability of distributed noise generation.

## [1.0.0] - 2025-07-10

### Added

-   First official stable release of JAX-Privacy — a library for
    privacy-preserving machine learning in
    JAX.[(#24)](https://github.com/google-parfait/jax_privacy/pull/24)
-   Full support for DP-SGD training with JAX, including integration with:
    -   Raw JAX training pipelines
    -   Flax Linen API
    -   Keras API for high-level model fine-tuning workflows.
-   End-to-end examples and Jupyter notebooks demonstrating usage of the
    library.
-   Core implementations of DP algorithms, including:
    -   Differentially Private Stochastic Gradient Descent (DP-SGD)
    -   Utilities for DP accounting (computing noise scales given privacy
        parameters) integrated into training pipelines.

### Changed

-   Library APIs designed for a more production-ready and modular experience
    compared with earlier research code.

### Fixed

-   Stability and packaging improvements to support publication on PyPI as
    version 1.0.0.

<!-- disableFinding(LINK_UNUSED_ID) -->

[unreleased]: https://github.com/google-parfait/jax_privacy/compare/v2.2.0...HEAD.
[2.2.0]: https://github.com/google-parfait/jax_privacy/compare/v2.0.0...v2.2.0.
[2.0.0]: https://github.com/google-parfait/jax_privacy/compare/v1.0.0...v2.0.0.

<!-- enableFinding(LINK_UNUSED_ID) -->

[1.0.0]: https://github.com/google-parfait/jax_privacy/compare/v0.2...v1.0.0.

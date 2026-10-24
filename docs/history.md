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

# History of JAX Privacy

## Overview

When JAX Privacy was initially released alongside
[De et al. (2022)](https://arxiv.org/abs/2204.13650) and
[Berrada et al. (2023)](https://arxiv.org/abs/2303.05555), a primary engineering
goal was making DP-SGD fast and scalable on modern accelerators. At the time,
existing frameworks like
[TensorFlow Privacy](https://github.com/tensorflow/privacy) made per-example
gradient clipping slow and difficult to speed up at scale. By harnessing JAX's
vectorized transformations ({func}`jax.vmap` and {func}`jax.pmap`) and gradient
accumulation, the original package enabled large-scale DP-SGD training with
large batch sizes, augmentation multiplicity, and long training horizons. Those
papers demonstrated that standard DP-SGD could achieve strong privacy-utility
tradeoffs on competitive vision benchmarks, putting DP-SGD on the map as a
viable method in practice; indeed, that recipe remains a very strong baseline
for the tasks considered and has proven difficult to beat. To support that
research, the initial repository was designed as an end-to-end experimental
*framework* for reproducing those experiments on the DeepMind stack of the time
(Haiku, Jaxline, and {func}`jax.pmap`).

Since those papers were published, the differentially private machine learning
literature has matured significantly. Many newer techniques now achieve strong
privacy-utility tradeoffs in lower-compute or single-pass regimes where large
batch sizes and data augmentation multiplicity are impractical. These advances
span different components of the mechanism (batch selection, noise addition,
clipping, optimization, and accounting), and supporting them cleanly called for
decoupling those components into standalone primitives rather than an end-to-end
training harness.

Around the same time, teams across Google were training differentially private
models across a variety of internal codebases. Directly integrating gradient
clipping and noise addition inside each training framework (for example, inside
the [Pax](https://github.com/google/paxml) codebase) proved brittle and
difficult to maintain across stacks. Even large-scale efforts like
[VaultGemma 1B](https://research.google/blog/vaultgemma-the-worlds-most-capable-differentially-private-llm/),
which built on earlier versions of JAX Privacy, reused its gradient clipping
module while implementing custom batch selection and accounting to handle
padding and truncation in their own training pipeline (see
[Chua et al., 2024](https://arxiv.org/abs/2411.04205)). To provide a shared,
framework-agnostic foundation for both modern JAX training stacks and newer DP
mechanisms, JAX Privacy evolved from an experimental *framework* (v0.x) into a
modular *library* (v2.x) of composable primitives.

## Scope

As JAX Privacy transitioned from an end-to-end experiment framework into a
general-purpose library, paper-specific models, dataset loaders, and experiment
runners were moved out of the core package, and the privacy primitives were
organized into six flat pillar modules.

<!-- mdformat off -->
<!-- markdownlint-disable MD013 -->

<table class="table" style="width: 100%; table-layout: fixed;">
  <colgroup>
    <col style="width: 22%;">
    <col style="width: 39%;">
    <col style="width: 39%;">
  </colgroup>
  <thead>
    <tr>
      <th class="text-left">Dimension</th>
      <th class="text-left">Original Framework (v0.x)</th>
      <th class="text-left">Current Library (v2.x)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Design Philosophy</strong></td>
      <td>Self-contained experimental framework built to reproduce large-scale DP vision papers.</td>
      <td>Composable general-purpose library of lower-level DP ingredients (plus a high-level <a href="_autosummary_output/jax_privacy.training.html"><code>training</code></a> API) that plug into any JAX loop.</td>
    </tr>
    <tr>
      <td><strong>Core Primitives</strong></td>
      <td>Fast per-example clipping and Gaussian noise in <a href="https://github.com/google-parfait/jax_privacy/blob/95870e4d6b9999f849f5426ba3dfab82a20a2317/jax_privacy/dp_sgd/gradients.py">GradientComputer</a>, paired with dataset iterators in the experiment pipeline.</td>
      <td>Six flat pillar modules: <a href="_autosummary_output/jax_privacy.clipping.html"><code>clipping</code></a>, <a href="_autosummary_output/jax_privacy.noise_addition.html"><code>noise_addition</code></a>, <a href="_autosummary_output/jax_privacy.batch_selection.html"><code>batch_selection</code></a>, <a href="_autosummary_output/jax_privacy.accounting.html"><code>accounting</code></a>, <a href="_autosummary_output/jax_privacy.auditing.html"><code>auditing</code></a>, and <a href="_autosummary_output/jax_privacy.optimizers.html"><code>optimizers</code></a>.</td>
    </tr>
    <tr>
      <td><strong>Model &amp; Modality</strong></td>
      <td>Tailored for supervised image classification, augmentation multiplicity, and Haiku <code>(params, network_state)</code> signatures.</td>
      <td>Modality- and model-agnostic; operates on arbitrary PyTrees of JAX arrays with a leading batch axis.</td>
    </tr>
    <tr>
      <td><strong>Dependencies</strong></td>
      <td>15 packages including Haiku, Jaxline, TensorFlow, TensorFlow Datasets, and scikit-learn.</td>
      <td>7 core requirements centered on JAX, Optax, and DP Accounting.</td>
    </tr>
    <tr>
      <td><strong>Models &amp; Data Loaders</strong></td>
      <td><strong>In Scope:</strong> Bundled Haiku vision models (NFNets, WideResNets) and TFDS loaders (ImageNet, CIFAR, CheXpert, MIMIC-CXR).</td>
      <td><strong>Out of Scope:</strong> Defining model architectures, tokenizers, and dataset loaders is left to the user's stack.</td>
    </tr>
    <tr>
      <td><strong>Domain Pipelines</strong></td>
      <td><strong>In Scope:</strong> Bundled end-to-end vision pipelines (EMA averaging, auto-tuning, evaluators).</td>
      <td><strong>Out of Scope:</strong> Domain-specific pipelines live in downstream libraries (e.g., <a href="https://github.com/google-parfait/dpsynth">DPSynth</a>).</td>
    </tr>
    <tr>
      <td><strong>Binaries &amp; Runners</strong></td>
      <td><strong>In Scope:</strong> Included Jaxline CLI entrypoints, launch scripts, and paper-specific experiment configs.</td>
      <td><strong>Out of Scope:</strong> Pure importable library; runnable scripts live in a standalone <a href="https://github.com/google-parfait/jax_privacy/tree/main/examples">examples/</a> directory.</td>
    </tr>
  </tbody>
</table>

## Key Design Changes

Click any pillar below to expand its technical details and Before / After
comparisons:

*   <details>
    <summary><strong>Gradient Clipping (<code>clipping.py</code>):</strong> Generalizes <a href="https://github.com/google-parfait/jax_privacy/blob/95870e4d6b9999f849f5426ba3dfab82a20a2317/jax_privacy/dp_sgd/gradients.py">DpsgdGradientComputer</a> into a <code>jax.grad</code>-style <code>clipped_grad</code> primitive supporting sum-based aggregation, microbatching, and pre-clipping transforms.</summary>

    *   **Sum-Based Aggregation:** Originally,
        [gradient clipping](https://github.com/google-parfait/jax_privacy/blob/95870e4d6b9999f849f5426ba3dfab82a20a2317/jax_privacy/dp_sgd/grad_clipping.py)
        averaged per-example gradients over the batch axis for fixed-size
        batches. To support variable-size batches (such as under Poisson
        subsampling, where dividing by a random batch size $|B|$ changes the
        sensitivity of the aggregate), {func}`~jax_privacy.clipping.clipped_grad`
        computes a sum over clipped gradients and returns a
        {class}`~jax_privacy.clipping.BoundedSensitivityCallable` whose
        sensitivity metadata can be inspected directly when calibrating noise
        (see [Variable Batch Sizes](sharp_edges_variable_batch_sizes.md) and
        [Common Pitfalls](sharp_edges_dp_training_pitfalls.md)).
    *   **Microbatching (`vmap` $\leftrightarrow$ `scan`):** The original
        version offered separate vectorized ({func}`jax.vmap`) and loop
        ({func}`jax.lax.scan`) clipping modes alongside trainer-level gradient
        accumulation. In the rewrite,
        {func}`~jax_privacy.clipping.clipped_grad` builds `microbatch_size`
        directly into the gradient function, interpolating between `vmap` and
        `scan` in a single call.
    *   **Pre-Clipping Transforms:** To support newer adaptive DP optimizers
        like {func}`~jax_privacy.optimizers.scale_then_privatize` that
        precondition per-example gradients prior to clipping while preserving
        standard Euclidean $L_2$ sensitivity bounds, `clipped_grad` provides a
        dedicated `pre_clipping_transform` hook that transforms per-example
        gradients *before* $L_2$ clipping.
    *   **Standard `jax.grad` Ergonomics:** Previously, `DpsgdGradientComputer`
        was designed around Haiku's stateful trainer signature
        `(params, network_state, rng_per_example, inputs)`. Today,
        {func}`~jax_privacy.clipping.clipped_grad` adopts standard
        `jax.grad(fun, argnums=0, has_aux=False)` conventions so it can be used
        as a drop-in replacement for {func}`jax.grad`:

        ```python
        # BEFORE (v0.x): 4-argument trainer wrapper and DpsgdGradientComputer
        from jax_privacy.dp_sgd import grad_clipping, gradients, typing

        def loss_fn(param, X, y):
          return 0.5 * jnp.mean((X @ param - y) ** 2)

        def wrapped_loss_fn(param, unused_network_state, unused_rng, inputs):
          return loss_fn(param, *inputs), (None, typing.Metrics())

        grad_computer = gradients.DpsgdGradientComputer(
            clipping_norm=1.0,
            noise_multiplier=0.0,
            rescale_to_unit_norm=False,
            per_example_grad_method=grad_clipping.VECTORIZED,
            global_norm_fn=optax.global_norm,
        )
        _, _, _, grads = grad_computer.loss_and_clipped_gradients(
            loss_fn=wrapped_loss_fn,
            params=param,
            network_state=None,
            rng_per_example=jax.random.PRNGKey(0),
            inputs=(X, y),
        )
        ```

        ```python
        # AFTER (v2.x): Drop-in replacement for jax.grad
        import jax_privacy

        def loss_fn(param, x, y):
          return 0.5 * (x @ param - y) ** 2

        grad_fn = jax_privacy.clipped_grad(loss_fn, l2_clip_norm=1.0)
        grads = grad_fn(param, X, y)
        ```

    </details>

*   <details>
    <summary><strong>Noise Addition (<code>noise_addition.py</code>):</strong> Decouples noise addition from clipping and expresses stateful correlated noise as a standard <code>optax.GradientTransformation</code>.</summary>

    *   **Decoupled Clipping and Noise:** Originally,
        [GradientComputer](https://github.com/google-parfait/jax_privacy/blob/95870e4d6b9999f849f5426ba3dfab82a20a2317/jax_privacy/dp_sgd/gradients.py)
        combined per-example gradient clipping with i.i.d. Gaussian noise
        addition in a single class, which was natural for standard DP-SGD. In
        the rewrite, {mod}`~jax_privacy.clipping` and
        {mod}`~jax_privacy.noise_addition` are decoupled into independent
        primitives so i.i.d. Gaussian noise and stateful correlated
        {mod}`~jax_privacy.matrix_factorization` mechanisms can be swapped
        seamlessly.
    *   **Three-Line Framework Integrations (Case Study: Tunix):** External
        training frameworks like [Tunix](https://github.com/google/tunix) manage
        their own training loops, checkpointing, and optimizer state containers,
        where plugging in an Optax transformation is much simpler than adopting a
        separate
        [dp_updater.py](https://github.com/google-parfait/jax_privacy/blob/95870e4d6b9999f849f5426ba3dfab82a20a2317/jax_privacy/training/dp_updater.py)
        or manually threading custom privatizer state. Because the current
        library expresses stateful correlated noise as a standard
        {class}`optax.GradientTransformation`, Tunix chains the privatizer with
        any base Optax optimizer in **three lines of code**:

        ```python
        privatizer = self._dp_plan.noise_addition_transform
        noisy_optimizer = optax.chain(privatizer, optimizer)
        self.optimizer = nnx.Optimizer(self.model, noisy_optimizer, wrt=wrt)
        ```

        The framework's optimizer and checkpoint manager automatically
        initialize, shard, update, and checkpoint the stateful BandMF noise
        buffer alongside the base optimizer state.

    </details>

*   <details>
    <summary><strong>Batch Selection (<code>batch_selection.py</code>):</strong> Introduces first-class <code>BatchSelectionStrategy</code> classes in the core library, aligning batch construction with privacy accounting assumptions.</summary>

    Following standard practice in the DP-ML literature at the time, the
    original package paired Poisson or replacement accounting with standard
    shuffled or fixed-size data loaders rather than providing a dedicated batch
    selection module. Subsequent research showed that mismatches between
    accounting assumptions and batch construction can materially affect privacy
    guarantees (see [Common Pitfalls](sharp_edges_dp_training_pitfalls.md)).

    In the rewrite, {mod}`jax_privacy.batch_selection` provides first-class
    {class}`~jax_privacy.batch_selection.BatchSelectionStrategy` implementations
    ({class}`~jax_privacy.batch_selection.CyclicPoissonSampling`,
    {class}`~jax_privacy.batch_selection.BallsInBinsSampling`,
    {class}`~jax_privacy.batch_selection.BMinSepSampling`, and user-level
    grouping), as well as random-access and streaming partitioning utilities
    that handle padding and truncation cleanly. See
    [Batch Selection](batch_selection.md) for details.

    </details>

*   <details>
    <summary><strong>Mechanism Definition &amp; Configuration (<code>execution_plan.py</code>):</strong> Generalizes <a href="https://github.com/google-parfait/jax_privacy/blob/95870e4d6b9999f849f5426ba3dfab82a20a2317/jax_privacy/training/algorithm_config.py">AlgorithmConfig</a> and <a href="https://github.com/google-parfait/jax_privacy/blob/95870e4d6b9999f849f5426ba3dfab82a20a2317/jax_privacy/training/algorithm_config.py">DpsgdConfig</a> into <code>BandMFConfig</code> and a verified <code>DPExecutionPlan</code>.</summary>

    While a single noise multiplier was convenient for parameterizing simple
    DP-SGD sweeps in the original package, it does not by itself characterize a
    DP-SGD mechanism: the formal guarantee depends jointly on the noise
    multiplier, the number of training iterations, the sampling probability, and
    the clipping norm.

    In the rewrite, {class}`~jax_privacy.execution_plan.BandMFConfig` (which
    subsumes standard DP-SGD when using a single band) captures the complete
    mechanism definition and constructs a verified
    {class}`~jax_privacy.execution_plan.DPExecutionPlan` that couples the
    privacy accounting event, batch selection strategy, clipped gradient
    function, and noise addition transformation by construction.

    </details>

*   <details>
    <summary><strong>Privacy Accounting (<code>accounting.py</code>):</strong> Streamlines the <a href="https://github.com/google-parfait/jax_privacy/blob/95870e4d6b9999f849f5426ba3dfab82a20a2317/jax_privacy/accounting/accountants.py">accounting</a> and <a href="https://github.com/google-parfait/jax_privacy/blob/95870e4d6b9999f849f5426ba3dfab82a20a2317/jax_privacy/accounting/calibrate.py">calibration</a> subpackage into lightweight <code>dp_accounting.DpEvent</code> builders.</summary>

    The original package maintained a five-file accounting subpackage with
    experiment-specific accountant classes, caching hierarchies, and custom
    binary search routines for calibrating noise multipliers, update counts, and
    batch sizes. As upstream `dp_accounting` matured to provide general-purpose
    calibration utilities, the rewrite consolidated {mod}`jax_privacy.accounting`
    into a lightweight module of `dp_accounting.DpEvent` builders paired directly
    with upstream `dp_accounting.calibrate_dp_mechanism`.

    </details>

<!-- markdownlint-enable MD013 -->
<!-- mdformat on -->

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

# Overview

JAX Privacy provides robust, scalable, pure-JAX implementations of the
building blocks necessary to develop DP mechanisms for machine learning
applications, alongside high-level training APIs ({mod}`~jax_privacy.training`
and the [Keras API](keras_api.rst)) and worked [examples](examples_guide). For
a high-level summary of what is in and out of scope for the library, see
[Scope and Design Philosophy](https://github.com/google-parfait/jax_privacy#scope-and-design-philosophy)
in the repository README and the
[accompanying paper](https://arxiv.org/abs/2602.17861).

The key building blocks that constitute a DP Mechanism include:

1.  **(Gradient) Clipping** ({mod}`jax_privacy.clipping`):
    Compute bounded-sensitivity aggregations over a batch of examples (usually
    minibatch gradients).

2.  **Noise Addition** ({mod}`jax_privacy.noise_addition`): Add
    (possibly correlated or adaptive) noise to clipped minibatch gradients,
    providing formal guarantees for the output.

3.  **Batch Selection** ({mod}`jax_privacy.batch_selection`): Construct a
    (usually non-deterministic) sequence of example batches to apply gradient
    clipping and noise addition to.

4.  **Accounting** ({mod}`jax_privacy.accounting`): To compute the privacy
    budget required to run a given mechanism or calibrate the mechanism
    parameters (like noise multiplier) to achieve a given privacy target.

5.  **Auditing** ({mod}`jax_privacy.auditing`): To provide
    empirical measures of the privacy of a model, usually used in conjunction
    with canary insertion (either at the example level or the gradient level).

The design of the JAX Privacy core library aims to make these different building
blocks work together seamlessly, while allowing the user to pick and choose
which ones they need for their application. While end-to-end DP mechanisms
generally need to be concerned with at least the first four building blocks,
there are some cases when one might want to bypass some of our utilities:

*   When doing convex optimization over Lipschitz-continuous losses, the
    gradient already has bounded sensitivity, and so gradient clipping could in
    principle be bypassed.

*   When doing research, it is common practice to read the data in a
    deterministic or shuffled order, rather than adhering to the exact batch
    selection strategy required to obtain the desired formal DP guarantees.
    While we don't recommend doing this, our batch selection strategies can be
    ignored in such cases.

---

(api-tiers)=
## API Tiers

JAX Privacy provides three levels of API, each trading flexibility for stronger
built-in DP assurance. Each tier builds on the one below it: the higher-level
tiers compose the building blocks for you, so that privacy-critical coupling
decisions are made once in the library rather than once per user.

```{tip}
**Design principle:** You should not be able to configure any individual
JAX Privacy utility in a way that breaks its stated guarantees. If you can,
that is a bug. We attempt to enforce this at the API level, even when it
sacrifices some flexibility.
```

### Tier 1: End-to-end training loops

*   [Keras API](keras_api.rst), {mod}`~jax_privacy.training`

These consume a {class}`~jax_privacy.execution_plan.DPExecutionPlan` and write
the entire training loop for you — batch selection, gradient computation, noise
addition, parameter updates, and privacy accounting. The resulting training
satisfies the stated DP guarantee unconditionally. It is a design goal of
JAX Privacy that you should not need to reason about how the components
interact; the library handles this for you.

Currently, the Keras API supports DP-SGD with internally Poisson-sampled batches
built from random-access per-example arrays (with accounting done using
the same Poisson-sampling assumption); arbitrary DP `ExecutionPlan`s are not
yet supported.

### Tier 2: `DPExecutionPlan`

*   {mod}`~jax_privacy.execution_plan` (currently
    {class}`~jax_privacy.execution_plan.BandMFConfig` and
    {class}`~jax_privacy.execution_plan.NonPrivateConfig`)

A single config object bundles batch selection, clipped gradient computation,
noise addition, and the corresponding `DpEvent` into a verified
{class}`~jax_privacy.execution_plan.DPExecutionPlan`. When you use the plan's
components as documented to write your own training loop, the resulting loop
inherits the stated DP guarantee *by construction*. See
[`dp_sgd_transformer.py`](https://github.com/google-parfait/jax_privacy/tree/main/examples/dp_sgd_transformer.py)
and
[`dp_logistic_regression.py`](https://github.com/google-parfait/jax_privacy/tree/main/examples/dp_logistic_regression.py)
for examples.

### Tier 3: Core API (low-level building blocks)

*   {mod}`~jax_privacy.batch_selection`, {mod}`~jax_privacy.clipping`,
    {mod}`~jax_privacy.noise_addition`, {mod}`~jax_privacy.accounting`

Users compose the individual modules directly. **Any**
{class}`~jax_privacy.batch_selection.BatchSelectionStrategy` can be used,
**any** noise addition scheme can be composed, and **any** `DpEvent` can be used
for privacy analysis — including Monte Carlo accounting via
{mod}`~jax_privacy.experimental.monte_carlo` for combinations where PLD/RDP
accounting is not available. Each individual component is designed so that you
should not be able to configure it in a way that breaks its own *local formal
guarantees* — and if you can, that is a bug. Importantly, these local guarantees
are not DP guarantees: individual components do not satisfy DP by themselves.
They are properties like sensitivity bounds and per-example isolation that serve
as the building blocks for *proving* DP for the higher-level compositions. The
risk at this tier is in *composition*: wiring components together incorrectly
(e.g., calibrating noise to the wrong sensitivity, or using an accounting method
that does not match the batch selection strategy). See
[`jax_api_example.py`](https://github.com/google-parfait/jax_privacy/tree/main/examples/jax_api_example.py)
and
[`balls_in_bins_accounting.py`](https://github.com/google-parfait/jax_privacy/tree/main/examples/balls_in_bins_accounting.py)
for examples.

### Your responsibilities at each tier

Which parts of DP correctness you own depends on the tier you build on:

-   **Tier 1 (end-to-end training loops):** Nothing, as far as component
    composition goes. Configure the mechanism and the library handles batch
    selection, clipping, noise, and accounting so the training loop satisfies
    the stated guarantee.

-   **Tier 2 ({class}`~jax_privacy.execution_plan.DPExecutionPlan`):** Use the
    plan's components as documented. The privacy-critical pieces are already
    coupled consistently; your job is to wire them into your training loop as
    described.

-   **Tier 3 (Core API):** You own composition. You are responsible for
    calibrating noise to the right sensitivity and using an accounting method
    that matches your batch selection. The components are designed so that
    assembling a basic DP-SGD loop is straightforward; going beyond that (e.g.,
    custom mechanisms or accounting) requires genuine DP expertise, and that is
    a deliberate, acceptable trade-off for the flexibility this tier provides.

This is a direct consequence of JAX Privacy's **flat, auditable design**:
privacy-critical logic is not buried across nested abstraction layers. Each
component (clipping, noise addition, batch selection, accounting) stands alone
and can be understood, tested, and audited in isolation; coupling happens only
at the higher-level API layer, where the joint guarantees are stated explicitly.
See the [Common Pitfalls](sharp_edges_dp_training_pitfalls) page for how each
pitfall interacts with the tier system.

```{important}
**What JAX Privacy can and cannot control.** JAX Privacy governs what it
*computes*: the sensitivity of a gradient, the noise added, the accounting of
a mechanism. It cannot govern what you *release*. It will not stop you from
logging a PRNG seed, or checkpointing raw training state to unencrypted storage.
You must always reason about what leaves your trust boundary, and to whom. This
is a property of *your* deployment, not of the library.
```

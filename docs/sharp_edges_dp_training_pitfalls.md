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

# Common Pitfalls in DP Training

<!-- disableFinding(LINK_ID) -->

*Design decisions JAX Privacy makes to prevent them*

*Authors: Ryan McKenna, H. Brendan McMahan, and Tudor Cebere*

Building a correct differentially private training pipeline is hard. A single
misplaced operation — dividing by the wrong constant, using the wrong batch
sampling strategy, or miscalibrating noise — can silently break the formal
privacy guarantee while producing models that *look* fine.

JAX Privacy is designed to make correctness the path of least resistance.
The design decisions described on this page are based on years of experience
building and integrating DP training pipelines — both our own implementations
and integrations with internal and external libraries. Each one addresses a
specific pitfall we have encountered in practice.

---

## API Tiers and Pitfall Ownership

JAX Privacy provides [three API tiers](api-tiers) — end-to-end training loops,
`DPExecutionPlan`, and low-level building blocks — each trading flexibility for
stronger built-in DP assurance. Which pitfalls on this page you need to worry
about depends on the tier you build on:

-   **Tier 1 (end-to-end training loops):** The library handles component
    composition for you. None of the composition pitfalls below apply.

-   **Tier 2 (`DPExecutionPlan`):** The privacy-critical pieces are already
    coupled consistently; your job is to wire them into your training loop as
    described.

-   **Tier 3 (Core API):** You own composition. You are responsible for
    calibrating noise to the right sensitivity and using an accounting method
    that matches your batch selection. The pitfalls below are most relevant at
    this tier.

See the [API Tiers](api-tiers) section of the Overview for the full description
of each tier, its capabilities, and your responsibilities.

---

(how-severe-is-each-pitfall)=
## How Severe Is Each Pitfall?

Not all pitfalls are equally dangerous. It helps to classify them by *what
actually happens to your privacy guarantee* when you hit one. We use three
severity classes throughout this page.

First, a word on what we mean by "severity." These classes describe *how much
the formal DP guarantee is affected*, not how likely a real-world privacy harm
is. Whether a formal weakening translates into actual harm is deeply use-case-
and infrastructure-specific — it depends on your other data controls, the
sensitivity of the data, and your threat model — and is largely outside what
JAX Privacy can speak to. A **Critical** pitfall means *the released outputs do
not satisfy the DP guarantee you intended to claim*; it does **not** mean a
real privacy violation is likely. In fact the outputs often still satisfy
*some* (weaker) guarantee than the one you reported.

```{note}
For simplicity we phrase everything in terms of $(\epsilon, \delta)$-DP,
with $\epsilon$ as the primary parameter and $\delta$ fixed to some small
value. The statements apply equally to other DP formulations (e.g. RDP or
$\mu$-GDP), or to direct bounds on an attacker's true-positive rate at a
fixed false-positive rate.
```

```{note}
These severity classes are orthogonal to the [API tiers](api-tiers). The tiers
describe *who is responsible* for correctness (you vs. the library); the
severity classes describe *how badly the guarantee breaks* if correctness
fails.
```

```{important}
**Critical -- silent, formal invalidation.** An ordinary implementation or
composition mistake means the released outputs *do not satisfy the DP guarantee
you intended to claim*: the stated $\epsilon$ no longer holds (though a weaker
guarantee often still does). No adversary is required, and the code runs without
errors, so nothing warns you. These are the most dangerous pitfalls precisely
because a careful, well-intentioned user can trigger them by accident.
*Example: calibrating noise to the wrong sensitivity, or using an accounting
method that does not match the batch selection strategy.*
```

```{important}
**Theoretical -- a formal break only under an adversarial, largely theoretical
threat model.** The failure *can* void the formal guarantee (provably, in the
worst case), but reaching that worst case requires an active adversary and
conditions that are, today, largely theoretical. Whether you need to address it
depends on your threat model.
```

```{important}
**Negligible -- bounded, negligible degradation.** Even in the worst case you
lose only a negligible amount of privacy: the realized guarantee is
$\epsilon' = \epsilon + \text{tiny}$ rather than $\epsilon$. The guarantee
still holds; it is simply a hair weaker than reported, typically because of
finite-precision arithmetic. *Example: numerical error in accounting, or a
clipped gradient whose norm floating-point-rounds a hair above the clip norm
$C$.*
```

The *same* underlying phenomenon can appear in different classes.
Floating-point arithmetic, for instance, is **Theoretical** when viewed as an
exploitable side channel (a formal break under a strong adversary) but
**Negligible** when viewed as ordinary rounding (a negligible overshoot of the
sensitivity bound). The distinction is the severity of the realistic outcome,
not the root cause.

Every pitfall below is tagged with its severity class in the summary table.

---

## Common Pitfalls at a Glance

Pitfall                                                                                                                   | Severity                    | How JAX Privacy Handles It
:------------------------------------------------------------------------------------------------------------------------ | :-------------------------- | :-------------------------
[Division by batch size](#division-by-batch-size)                                                                         | Critical                    | Computes a sum, not a mean; optional `normalize_by` reflected in sensitivity
[Sensitivity alignment](#sensitivity-alignment)                                                                           | Critical                    | Returned callable exposes `.sensitivity()` — calibrate noise against it
[Accounting and batch selection mismatch](#accounting-and-batch-selection-mismatch)                                       | Critical                    | `DPExecutionPlan` couples these by construction
[Public vs. private metadata](#public-vs-private-metadata)                                                                | Critical                    | Dataclass fields are public config; sensitive values are method arguments
[Neighboring relation clarity](#neighboring-relation-clarity)                                                             | Critical                    | Explicit `NeighboringRelation` enum; `.sensitivity()` parameterized by it
[Error handling and data-dependent failures](#zero-sized-batches-and-non-finite-gradients)                                | Critical                    | Handles empty batches and non-finite per-example gradients; arbitrary user-code and runtime failures remain the caller's responsibility
[Gradient accumulation](#gradient-accumulation)                                                                           | Critical                    | No manual accumulation needed — on-device microbatching instead
[Cross-example operations (incl. batch norm and MoE)](#cross-example-operations-incl-batch-normalization-and-moe-routing) | Critical                    | `vmap` isolates examples automatically, but this silently changes model semantics — a utility pitfall
[Data augmentation](#data-augmentation-and-the-privacy-unit)                                                              | Critical                    | Augment inside a per-unit composite loss and clip once per original privacy unit
[Auxiliary information](#auxiliary-information)                                                                           | Critical                    | Per-example returns; aggregation is the caller's responsibility
[Randomness and RNG injection](#randomness-and-rng-injection)                                                             | Theoretical                 | Explicit RNG parameters; supports cryptographically secure sources
[Floating point robustness](#floating-point-robustness)                                                                   | Theoretical                 | Opt-in discrete Gaussian mechanism with integer-domain clipping
[Finite-precision numerical error](#finite-precision-numerical-error)                                                     | Negligible                  | Inherent and negligible in the default path; discrete Gaussian path removes it entirely
[Privacy auditing: canary sampling](#privacy-auditing-experimental-design-sharp-edges)                                    | Critical for audit validity | Caller must use IID or exchangeable canaries
[Privacy auditing: multi-run vs. one-run bounds](#privacy-auditing-one-run-vs-multi-run)                                  | Critical for audit validity | Provides one-run auditing methods only
[Privacy auditing: threshold selection](#privacy-auditing-threshold-selection)                                            | Critical for audit validity | Provides explicit, Bonferroni, split, and multi-split strategies

*Severity key:* **Critical** = the released outputs do not satisfy the DP
guarantee you intended to claim; **Theoretical** = a formal break only under an
adversarial, largely theoretical threat model; **Negligible** = bounded,
negligible degradation ($\epsilon' = \epsilon + \text{tiny}$). See
[How severe is each pitfall?](#how-severe-is-each-pitfall) for definitions.
**Critical for audit validity** means the reported empirical lower bound may be
invalid; it does not weaken the trained mechanism's DP guarantee.

Beyond these failure modes, JAX Privacy follows several cross-cutting design
principles, and there are related concerns that fall outside its scope. See
[Design Principles and Scope](#design-principles-and-scope).

---

(division-by-batch-size)=
### Division by Batch Size

**Severity: Critical.**

**The pitfall.** DP mechanisms naturally privatize *sums*: the noise is
calibrated to the sensitivity of a *sum* of per-example gradients. But in
standard ML, the loss — and therefore the gradient — is usually defined as an
*average* over the batch: `loss = sum(losses) / batch_size`. If the noise is
calibrated to a sum but the gradient is divided by the batch size (or vice
versa), you can end up adding too much or too little noise — off by a factor of
the batch size — silently miscalibrating the privacy guarantee.

This is further complicated under Poisson sampling, where the batch size is
itself a *random variable* that depends on which examples were selected, making
it technically sensitive: dividing by the actual (random) batch size can itself
leak information about the batch composition.

**How JAX Privacy handles it.** The {func}`~jax_privacy.clipping.clipped_grad`
function computes a *sum* of clipped per-example gradients, not a mean. This
avoids any dependence on the random batch size. If you want to recover a mean
gradient (e.g., for compatibility with standard optimizers), you can pass a
`normalize_by` value — typically the *expected* batch size, which is a public,
deterministic quantity:

```python
# User writes a per-example loss -- no batch-size division needed.
def loss_fn(params, x, y):
    logits = model.apply(params, x)
    return cross_entropy(logits, y)

# JAX Privacy computes a sum of clipped gradients, optionally normalized.
grad_fn = jax_privacy.clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    normalize_by=expected_batch_size,  # Public, deterministic quantity.
)
```

Crucially, the `normalize_by` value is reflected in the sensitivity reported
by the returned callable's `.sensitivity()` method, so downstream noise
calibration stays correct automatically.

---

(sensitivity-alignment)=
### Sensitivity Alignment

**Severity: Critical.**

**The pitfall.** There are many ways to configure gradient clipping: global
clipping, per-layer clipping, rescaling to unit norm, integer grid rounding
for the discrete Gaussian mechanism, and more. Each configuration produces a
different sensitivity for the clipped gradient function. If you calibrate your
noise multiplier based on an *assumed* sensitivity that does not match the
*actual* sensitivity of your clipping configuration, the privacy guarantee is
invalidated.

**How JAX Privacy handles it.** The callable returned by
{func}`~jax_privacy.clipping.clipped_grad` (and
{func}`~jax_privacy.clipping.clipped_fun`) exposes a `.sensitivity()` method
that reports the exact L2 sensitivity of the clipped output, accounting for all
configuration options:

```python
grad_fn = jax_privacy.clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    rescale_to_unit_norm=True,
    normalize_by=expected_batch_size,
)

# Query the sensitivity -- it accounts for clipping, rescaling, and normalization.
sensitivity = grad_fn.sensitivity()
```

The {class}`~jax_privacy.execution_plan.DPExecutionPlan` uses this method
internally to calibrate noise, ensuring that the noise multiplier and the
clipping configuration are always aligned. If you are using the lower-level
building blocks directly, we recommend always calibrating noise against the
`.sensitivity()` of the clipped gradient callable rather than computing the
sensitivity by hand.

---

(accounting-and-batch-selection-mismatch)=
### Accounting and Batch Selection Mismatch

**Severity: Critical.**

**The pitfall.** Privacy accounting makes assumptions about how batches are
formed (e.g., Poisson sampling with a specific probability). If the actual
batch selection strategy does not match these assumptions, the computed privacy
guarantee is invalid. This is one of the most common and most dangerous
mistakes in DP training, because the training loop runs without errors and the
resulting model looks normal.

**How JAX Privacy handles it.** The
{class}`~jax_privacy.execution_plan.DPExecutionPlan` bundles the batch
selection strategy, clipped gradient computation, noise addition, and the
corresponding `DpEvent` into a single object. The `DpEvent` is derived from
the same parameters that configure the batch selection and noise addition,
ensuring they are consistent by construction:

```python
config = jax_privacy.BandMFConfig.default(
    num_bands=1,
    iterations=1000,
    expected_participations=100,
).calibrate(epsilon=1.0, delta=1e-5)

plan = config.make()

# These are guaranteed to be consistent:
plan.batch_selection_strategy  # Poisson sampling with matching probability
plan.noise_addition_transform  # Noise calibrated to matching sensitivity
plan.dp_event                  # DpEvent derived from the same parameters
```

If you drop down to the low-level building blocks and assemble batch selection,
noise, and accounting yourself, this consistency is *your* responsibility:
there is no cross-check that the accounting method you choose actually matches
the batch selection strategy you run. Keep them in sync deliberately, or use a
{class}`~jax_privacy.execution_plan.DPExecutionPlan` (or a higher tier) so that
they are coupled by construction.

---

(public-vs-private-metadata)=
### Public vs. Private Metadata

**Severity: Critical.**

**The pitfall.** DP training pipelines handle a mix of public quantities
(hyperparameters, expected batch size, number of iterations) and
data-dependent quantities (the actual dataset size, which examples were
selected, the random state used for noise). The question that matters for
privacy is not what you *store* internally, but what you *release* — what
leaves the trust boundary under your DP guarantee. If public and data-dependent
quantities are not clearly separated in the API, it is easy to accidentally
treat a data-dependent value as public and release it (for example, by baking
the actual dataset size into an artifact that is published alongside the
model).

What counts as "released" depends on your threat model. Often the only thing
released from DP training is the model itself, but in a stricter model you may
want the guarantee to cover anything the engineer running the training can see.
JAX Privacy stays agnostic about how a given deployment handles logging,
storage, and access control — those are deployment design choices — and instead
focuses on making the public/data-dependent distinction structural and
explicit.

**How JAX Privacy handles it.** JAX Privacy uses a deliberate structural
separation. In the batch selection API, for example, the
{class}`~jax_privacy.batch_selection.BatchSelectionStrategy` is a frozen
dataclass whose *fields* are all public configuration (sampling probability,
number of iterations, cycle length). Data-dependent values — the number of
examples and the random number generator — are only consumed as *arguments* to
the {meth}`~jax_privacy.batch_selection.BatchSelectionStrategy.batch_iterator`
method:

```python
# Fields are public configuration -- data-independent, safe to release.
strategy = jax_privacy.CyclicPoissonSampling(
    sampling_prob=0.01,
    iterations=1000,
    cycle_length=1,
)

# Data-dependent values are only consumed at call time.
for batch_indices in strategy.batch_iterator(
    num_examples=len(dataset),  # Sensitive under add-or-remove-one DP.
    rng=rng,                    # Privacy-critical randomness.
):
    ...
```

This separation maps onto two distinct serialization scenarios you will
encounter in practice:

- **Pre-execution configuration.** The objects that define your mechanism and
  its hyperparameters — and that characterize/calibrate the `DpEvent` — are
  data-independent. They can be freely serialized and saved before training
  begins, or even produced by a separate offline job (useful when calibration
  is expensive, e.g., Monte Carlo accounting).
- **Runtime training state.** For fault tolerance, production runs checkpoint
  the evolving training state. This object mixes public fields (model
  parameters, optimizer state, step count) with data-dependent ones (the RNG
  state and noise state). JAX Privacy cannot control how this checkpoint is
  stored: it is your responsibility to handle it safely (for example, using
  encrypted storage in a production setting), consistent with what your threat
  model allows you to release.

```{note}
**Note on neighboring relations:** Under some neighboring relations (e.g.,
zero-out or replace-one), the number of examples is public information. In
these cases, batch selection strategies may define it as a field on the
dataclass itself. The API design makes this distinction explicit rather than
leaving it to implicit convention.
```

---

(neighboring-relation-clarity)=
### Neighboring Relation Clarity

**Severity: Critical.**

**The pitfall.** Different DP neighboring relations (add-or-remove-one,
replace-one, zero-out) imply different sensitivity bounds and different
assumptions about what information is public. If the neighboring relation is
not explicitly stated and programmatically enforced, it is easy for different
parts of the pipeline to make incompatible assumptions — for example, the
clipping code assumes replace-one (sensitivity = 2C) while the accounting
code assumes add-or-remove-one (sensitivity = C), silently doubling the
actual privacy loss.

A subtler version of this problem arises from *parameterization choices*. In
research code, it is common to parameterize a mechanism in terms of the batch
size or the expected batch size. But doing so makes an implicit assumption
that you know the number of records in the dataset — and if the dataset size
is sensitive (as it is under add-or-remove-one), that assumption changes what
your DP guarantee actually covers. This does not automatically *invalidate* a
guarantee stated on the released model — the dataset size may be revealed
through your hyperparameter choices rather than through the model parameters —
but it does mean the formal statement must be explicit about which
data-touches are in scope. Our recommendation is to get this right in the
formal analysis rather than rely on heuristic arguments: parameterize
mechanisms in terms of quantities that are public under your chosen
neighboring relation (e.g., sampling probability and iteration count), so that
the guarantee you state is the guarantee you actually have.

**How JAX Privacy handles it.** JAX Privacy supports three neighboring
relations — `ADD_OR_REMOVE_ONE`, `REPLACE_ONE`, and `REPLACE_SPECIAL`
(zero-out) — and makes the choice explicit and first-class throughout the
API. Rather than recommending one relation over another, the higher-level
APIs *determine* the neighboring relation automatically from how you configure
the mechanism. For example, with
{class}`~jax_privacy.execution_plan.BandMFConfig` the default preset
(`num_examples=None`) produces a
{class}`~jax_privacy.execution_plan.DPExecutionPlan` under
`ADD_OR_REMOVE_ONE`; specifying `num_examples` (and optionally
`truncated_batch_size`) instead produces one under `REPLACE_SPECIAL`, where the
dataset size is treated as public.

The key property is that the neighboring relation is an *explicit field* on the
{class}`~jax_privacy.execution_plan.DPExecutionPlan` dataclass. It can be
inspected, reported, and programmatically enforced alongside the precise
`DpEvent` — in contrast to the common situation where the neighboring relation
is an implicit, undocumented assumption baked into the code. When you use the
low-level components directly, none of this is automatic: choosing a relation
and keeping sensitivity, batch selection, and accounting consistent with it
requires care and DP expertise (JAX Privacy does not do it for you at this
level).

The neighboring relation is an explicit parameter at every level:

- The `.sensitivity()` method on {func}`~jax_privacy.clipping.clipped_grad`'s
  returned callable is parameterized by a `NeighboringRelation` enum, so you
  always know exactly which neighboring relation your sensitivity bound
  corresponds to:

  ```python
  from dp_accounting import NeighboringRelation

  grad_fn = jax_privacy.clipped_grad(loss_fn, l2_clip_norm=1.0)

  # Explicit -- no ambiguity about what "sensitivity" means.
  s_add_remove = grad_fn.sensitivity(NeighboringRelation.ADD_OR_REMOVE_ONE)
  s_replace = grad_fn.sensitivity(NeighboringRelation.REPLACE_ONE)
  # s_replace == 2 * s_add_remove
  ```

- The {class}`~jax_privacy.execution_plan.DPExecutionPlan` stores the
  `neighboring_relation` as a field, and uses it consistently across batch
  selection, noise calibration, and privacy accounting.

JAX Privacy also takes the philosophy of parameterizing mechanisms directly —
the way you would in a paper — rather than in terms of derived quantities
like batch size that carry implicit assumptions about what is public. This
connects to the [public vs. private metadata](#public-vs-private-metadata)
separation: by parameterizing in terms of fundamental quantities (sampling
probability, number of iterations) rather than derived ones (batch size),
the API avoids baking in assumptions about the neighboring relation.

---

(zero-sized-batches-and-non-finite-gradients)=
### Error Handling and Data-Dependent Failures

**Severity: Critical.**

**The pitfall.** A DP mechanism's output does not traditionally account for
crashes or errors that occur during execution. If neighboring datasets exhibit
different failure behavior, this can create a distinguishing event and break the
theoretical guarantees. This is unlikely to create meaningful additional leakage
in most deployments, but it remains a formal correctness issue.

Two important examples are:

- **Zero-sized batches.** With Poisson sampling, it is possible (though
  unlikely) for a batch to have zero examples. If a training step *fails* on a
  zero-sized batch but *succeeds* on a size-one batch, the success or failure
  of the step itself leaks information about whether a particular example was
  in the dataset — violating DP.

- **Non-finite gradients.** NaN or infinity values in per-example gradients
  (e.g., from numerical instability) can propagate through aggregation and
  corrupt the clipped sum. If one example produces NaN and another does not,
  the presence or absence of NaN in the output leaks per-example information.

**How JAX Privacy handles it.** The {func}`~jax_privacy.clipping.clipped_grad`
and {func}`~jax_privacy.clipping.clipped_fun` functions support empty and padded
batches. With `nan_safe=True` (the default), contributions with non-finite
gradient norms are zeroed before aggregation.

Data-dependent failures in user-provided losses, input pipelines, runtimes, or
recovery logic remain the caller's responsibility. Keep the per-unit loss
defined on all allowed inputs and use data-independent failure handling.

See also [Handling Variable Batch Sizes](sharp_edges_variable_batch_sizes) for
strategies to handle variable batch sizes efficiently.

---

(gradient-accumulation)=
### Gradient Accumulation

**Severity: Critical.**

**The pitfall.** Gradient accumulation — processing the batch in smaller
chunks, accumulating the gradients, and applying a single optimizer update —
is a common technique for handling large batches in non-private training.
In DP training, it is not necessarily *wrong*, but it is tricky to get right.
For example, if you add IID noise to each chunk independently, you end up
with more total noise than necessary. If you add noise only after
accumulation, you need to ensure the accumulation preserves the sensitivity
bound. And if the noise across chunks is not truly independent (e.g., due to
PRNG reuse), the noise may be correlated or even identical across chunks,
which can silently break the privacy guarantee. Getting the interaction
between accumulation, clipping, and noise calibration correct is subtle and
error-prone.

**How JAX Privacy handles it.** JAX Privacy folds gradient accumulation *into*
{func}`~jax_privacy.clipping.clipped_grad` via the `microbatch_size` parameter,
and this is the key to why it is safe. Microbatching processes the batch in
sequential chunks using {func}`jax.lax.scan`, performing per-example clipping
and aggregation correctly inside a single function. Crucially,
`microbatch_size` is a **purely performance knob**: changing it changes *only*
how the computation is scheduled — how much work is done sequentially versus
vectorized, and therefore the memory/throughput tradeoff. It does **not**
change what {func}`~jax_privacy.clipping.clipped_grad` computes, either
semantically or numerically (up to floating-point issues). The sum of clipped
per-example gradients, its sensitivity, and where and at what scale noise is
added are all identical regardless of the microbatch size you pick.

This invariance is not a coincidence — it falls directly out of the
[sum-not-mean design](#division-by-batch-size). Because
{func}`~jax_privacy.clipping.clipped_grad` returns a *sum* of clipped
per-example gradients (not a mean), microbatching is just associative addition:
splitting the batch into chunks, summing each chunk, and adding the partial
sums yields the same total for any chunking, up to floating-point
non-associativity. The sensitivity of that sum — and hence the noise calibrated
against it — depends only on the per-example clip norm, not on how the examples
were grouped. Dividing by a *fixed, public* constant afterward (the
`normalize_by` option) preserves this, since it just scales the total sum. What
would break the invariance is taking a *per-chunk* mean over the actual number
of examples in each chunk: the divisor would then depend on how you grouped the
batch, so both the result and the sensitivity reasoning would too. That is
exactly the coupling JAX Privacy avoids by summing.

This is what makes it safe by construction: because the result is invariant to
`microbatch_size`, choosing it is a *performance* decision, never a *DP
correctness* decision. There is no microbatch setting that silently weakens your
guarantee — the worst you can do is pick a value that runs slowly or runs out of
memory.

```python
grad_fn = jax_privacy.clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    microbatch_size=32,  # Performance only: same result as any other value.
)
# Works correctly with arbitrarily large batches -- no manual gradient
# accumulation, no brittle code, no error-prone noise calibration.
```

To put the scale in context: in a transformer setting, a batch consists of $B$
sequences of $L$ tokens stored as `int32` values, requiring $B \times L \times
4$ bytes. With a sequence length of $L = 1024$, each sequence occupies just 4
KB — meaning you can fit over 260,000 sequences in 1 GB of memory for the input
data alone. When training across multiple machines, this scales proportionally,
and microbatching lets you trade sequential steps for peak memory without
touching the DP math.

Contrast this with a hand-rolled accumulation loop, where the noise placement
and scale are *your* responsibility — exactly the subtleties described above
(over-noising per chunk, preserving the sensitivity bound across the
accumulation, and keeping per-chunk noise independent). There, the way you
structure the loop genuinely *is* a DP-correctness decision. JAX Privacy will
not stop you from writing such a loop, but with `microbatch_size` you rarely
need to, and you sidestep that entire class of bugs.

---

(cross-example-operations-incl-batch-normalization-and-moe-routing)=
### Cross-Example Operations (incl. Batch Normalization and MoE Routing)

**Severity: Critical.** *For DP, cross-example operations would break
per-example isolation; JAX Privacy's `vmap`-based design neutralizes that by
construction. But the same design silently changes what your model computes — a
significant **utility** pitfall.*

**The pitfall.** Some computations aggregate statistics *across* the examples in
a batch. The most familiar is batch normalization, which computes means and
variances over the batch dimension, but the same issue arises in
mixture-of-experts (MoE) routing — where a load-balancing or routing loss
couples the examples in a batch — and in any other operation whose output for
one example depends on the other examples present. Such cross-example operations
are fundamentally incompatible with per-example DP, because a single example's
contribution affects the quantities used by every other example.

**How JAX Privacy handles it — and the utility cost.** Because JAX Privacy's
clipping is built on {func}`jax.vmap`, each example's forward and backward pass
runs in complete isolation. If your loss function includes batch normalization,
MoE routing, or any other cross-example operation, JAX Privacy will not raise an
error — it will run and satisfy the stated DP properties. But {func}`jax.vmap`
silently turns each cross-example operation into its *per-example* analogue,
causing "batch normalization" to become a no-op or undefined.

This means:

- **DP is not broken.** Per-example isolation is preserved by construction. You
  cannot accidentally introduce cross-example leakage through the loss function.
- **But your model probably does not do what you think.** This is the important
  part: the operation no longer computes what "batch normalization" or "MoE
  routing" normally means, and nothing warns you. Calling this merely a
  "modeling decision" undersells it — in practice, JAX Privacy does **not** work
  out of the box for models that rely on batch normalization or batch-level MoE
  routing. You get a valid DP guarantee for a *different* model than the one you
  intended to train, usually with degraded utility.
- **What to do instead.** Replace batch-dependent layers with per-example
  alternatives before training (e.g. group or layer normalization instead of
  batch normalization), and design any auxiliary losses so that they do not
  couple examples. Treat "does every layer still behave correctly under
  per-example `vmap`?" as a required modeling check, not an afterthought.
- **No special handling needed for DP.** JAX Privacy does not need to enumerate
  which operations are compatible; any JAX-traceable loss that runs under
  {func}`jax.vmap` is automatically DP-compatible. The burden is on you to
  confirm it is also *semantically* the model you want.

This is a direct consequence of the {func}`jax.vmap`-based design: the DP
guarantee holds for *any* JAX-traceable loss function, because per-example
isolation is enforced at the computation level rather than the layer level. The
flip side is that the responsibility for preserving model semantics under
per-example execution rests with you.

--------------------------------------------------------------------------------

(data-augmentation-and-the-privacy-unit)=
### Data Augmentation

**Severity: Critical.**

**The pitfall.** A faulty design is to create several augmentations of one
private record and insert them back into the dataset as separate examples, or to
mix multiple private records into one example, as in Mixup or CutMix.
Per-example clipping then need not bound each original record's total
contribution, invalidating a per-record sensitivity analysis.

**How JAX Privacy handles it.** Generate and combine the views inside the
per-record loss, then use {func}`~jax_privacy.clipping.clipped_grad` to clip
once for the original record:

```python
NUM_AUGMENTATIONS = 8

def loss_for_record(params, record, rng):
    augmentation_keys = jax.random.split(rng, NUM_AUGMENTATIONS)

    def loss_for_view(key):
        augmented_x = augment(key, record["x"])
        logits = model.apply(params, augmented_x)
        return cross_entropy(logits, record["y"])

    losses = jax.vmap(loss_for_view)(augmentation_keys)
    return jnp.mean(losses)


private_grad = jax_privacy.clipped_grad(
    loss_for_record,
    l2_clip_norm=1.0,
    batch_argnums=1,
    keep_batch_dim=False,
    prng_argnum=2,
)
```

Here the outer batch axis still ranges over original records. `prng_argnum`
supplies one key per record, and the augmentation losses are averaged before
clipping. For user-level DP, combine all examples and augmentations for one user
before clipping. Cross-record transformations such as MixUp or CutMix require
clipping each coupled group as one privacy unit or a separate sensitivity
analysis.

This is the augmentation-multiplicity pattern used in the original
[JAX Privacy research](https://arxiv.org/abs/2204.13650).

---

(auxiliary-information)=
### Auxiliary Information

**Severity: Critical.**

**The pitfall.** DP training pipelines typically need diagnostic information —
training loss, test loss, gradient norms, and other statistics — to monitor
convergence and debug issues. Returning these quantities exactly breaks DP,
because they depend on the training data. Yet many practitioners surface them
anyway, reasoning that the diagnostics will not be released publicly. The
danger is that this becomes an undocumented, unaudited exception to the
privacy guarantee.

A previous version of JAX Privacy allowed users to define custom metrics
aggregation functions. This not only complicated the API surface, but it
also obscured the fact that these aggregated metrics are not private — users
could easily overlook the need to privatize them.

**How JAX Privacy handles it.** The {func}`~jax_privacy.clipping.clipped_grad`
function returns auxiliary outputs (loss values, gradient norms, and
user-defined auxiliary data) on a *per-example* basis. It does not aggregate
them.

This design has two benefits:

1. **Power users can privatize.** Because the auxiliary outputs are
   per-example, a sophisticated user who wants end-to-end DP — including for
   the diagnostics — can apply an appropriate DP mechanism (e.g., a DP mean,
   median, or histogram) to the per-example values.

2. **Non-private aggregation is a deliberate choice.** If a user decides to
   take a non-private mean of the per-example losses for logging, that is
   their explicit decision. JAX Privacy does not do it for them, and the
   per-example nature of the output makes it clear that aggregation is the
   caller's responsibility.

```python
grad_fn = jax_privacy.clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    return_values=True,        # Per-example loss values
    return_grad_norms=True,    # Per-example gradient norms
)

clipped_grads, aux = grad_fn(params, batch)
aux.values       # shape: (batch_size,) -- per-example, not aggregated
aux.grad_norms   # shape: (batch_size,) -- per-example, not aggregated
```

```{warning}
**Limitation:** JAX Privacy does not *prevent* non-private aggregation of
auxiliary outputs. A careless user can still compute exact means or complete
histograms over per-example values without adding noise. The API makes this
a *visible, deliberate* choice rather than a hidden default, but it does
not enforce private aggregation.
```

---

(randomness-and-rng-injection)=
### Randomness and RNG Injection

**Severity: Theoretical.** *The default RNG is fine for research; a
cryptographically secure source matters only when your threat model includes an
adversary who can exploit predictable randomness.*

**The pitfall.** Privacy-critical randomness — batch selection, noise
generation — must come from a high-quality source. If the random number
generator is improperly seeded, reused across components, or not
cryptographically secure in settings that require it, the privacy guarantee
can be weakened or broken entirely.

**How JAX Privacy handles it.** Wherever privacy-critical randomness is
needed, JAX Privacy consumes the RNG directly as an explicit parameter. For
example, the `batch_iterator` method accepts an `rng` argument:

```python
# Option 1: Pass None to use NumPy's default RNG (convenient for research).
for batch in strategy.batch_iterator(num_examples, rng=None):
    ...

# Option 2: Pass a seed for reproducibility. This is fine for research, but for
# actual privacy-sensitive data a hard-coded seed is akin to inlining a password
# in source code -- use a secure source (Option 3) instead.
for batch in strategy.batch_iterator(num_examples, rng=42):
    ...

# Option 3: Inject a cryptographically secure source for production hardening.
import secrets
secure_rng = np.random.Generator(np.random.SFC64(secrets.randbits(128)))
for batch in strategy.batch_iterator(num_examples, rng=secure_rng):
    ...
```

Similarly, {meth}`~jax_privacy.execution_plan.ExecutionPlanConfig.make` and
{class}`~jax_privacy.training.DPTrainer` accept an optional `noise_seed` for
noise addition. When `noise_seed=None` (the default), a fresh random seed is
drawn from `os.urandom`, which is suitable for single-host runs. Pass an
explicit `noise_seed` when you need deterministic noise across multi-host
workers or reproducible tests; in privacy-sensitive deployments, treat any
explicit seed as secret and do not expose it in released logs or configurations.

This design provides three levels of assurance:

- **Convenient defaults** for research and experimentation (pass `None` or a
  seed).
- **Reproducibility** for debugging (pass a fixed seed).
- **Dependency injection** of cryptographically secure RNG sources for
  production deployments that require it.

---

(floating-point-robustness)=
### Floating Point Robustness

**Severity: Theoretical.** *These attacks are largely theoretical today and
irrelevant to most deployments, but under a strong adversary they can violate
the formal DP guarantee in the worst case — hence the opt-in hardened path. This
is distinct from
[finite-precision numerical error](#finite-precision-numerical-error), the
Negligible variant that only ever costs a negligible amount of privacy.*

**The pitfall.** Standard DP-SGD implementations operate in floating point
arithmetic, which introduces rounding errors. These rounding errors are
deterministic and data-dependent, meaning that in principle, an adversary
could exploit floating point non-associativity or rounding patterns to extract
information about individual training examples — even in the presence of
noise. While these attacks are largely theoretical today, they represent a
real gap between the mathematical DP guarantee (which assumes exact real
arithmetic) and what the implementation actually provides.

**How JAX Privacy handles it.** JAX Privacy's default code path uses standard
floating point arithmetic, which is sufficient for most research and
production use cases. For settings that require hardened guarantees, JAX
Privacy provides an opt-in code path that uses the **discrete Gaussian
mechanism** combined with **integer-domain clipping** via the `grid_scale`
parameter:

```python
grad_fn = jax_privacy.clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    grid_scale=10**9,  # Quantize gradients to an integer grid.
)
```

This integer-domain pipeline spans two components —
{func}`~jax_privacy.clipping.clipped_grad` handles the clipping and
quantization, and the noise-addition module handles the discrete Gaussian noise:

1. {func}`~jax_privacy.clipping.clipped_grad` clips per-example gradients and
   quantizes them to an integer grid with `grid_scale` steps per `l2_clip_norm`.
2. The quantized gradients are aggregated using exact integer arithmetic.
3. The noise-addition module adds noise from the discrete Gaussian
   distribution, which is defined over the integers and avoids floating point
   rounding in the noise itself.

The result is a DP guarantee that holds *exactly* — not up to floating point
approximation. Combined with
[cryptographically secure RNG injection](#randomness-and-rng-injection), this
provides a fully hardened implementation. See the
[secure noise example](https://github.com/google-parfait/jax_privacy/blob/main/examples/secure_noise_example.py)
for a complete working demonstration.

```{warning}
**Limitation:** The discrete Gaussian mechanism and cryptographically secure
RNG are opt-in, not the default. The default code path uses standard
floating point arithmetic and standard PRNGs (NumPy for batch selection,
JAX for noise addition), which means it is *theoretically* vulnerable to
floating-point-based attacks. For most research and production settings
this is acceptable, but users who need hardened guarantees must explicitly
opt in.
```

---

(finite-precision-numerical-error)=
### Finite-Precision Numerical Error

**Severity: Negligible.** *This is the milder sibling of the
[floating-point attack surface](#floating-point-robustness) above: same root
cause (finite-precision arithmetic), but a completely different severity. The
attack variant can violate the formal DP guarantee under a strong adversary
(Theoretical); this variant only ever costs you a negligible amount of privacy.*

**The pitfall.** Privacy-critical quantities are computed in finite-precision
arithmetic, and rounding can nudge them slightly in the *wrong* direction:

-   **Clipping.** Per-example clipping *formally* guarantees an L2 norm of at
    most the clip norm $C$, but the rescaling that enforces the clip can leave
    a gradient whose norm is a fraction of an ULP *above* $C$, so the true
    sensitivity is $C(1 + \eta)$ for a tiny $\eta$.

-   **Noise calibration.** The noise multiplier that an accountant computes for
    a target $(\epsilon, \delta)$ can round a hair *low*, so slightly less noise
    is added than the guarantee assumes.

-   **Accounting.** Modern accountants (privacy loss distributions, numerical
    composition, Monte Carlo estimators) discretize distributions and integrate
    numerically; the reported $\epsilon$ can differ slightly from the true one.

In every case the effect is the same in kind: the mechanism realizes
$\epsilon' = \epsilon + \text{tiny}$ instead of the $\epsilon$ you report.

**Why this loss is negligible.** Unlike the [Critical](#how-severe-is-each-pitfall)
pitfalls — where the released outputs no longer satisfy the claimed guarantee —
the guarantee is not voided here and no adversary gains meaningful power. You
simply have a guarantee that is a hair weaker than the number on the page,
bounded by machine precision. This is exactly why it belongs in a different
severity class from the floating-point *attack* surface: that one can void the
formal guarantee; this one cannot.

**How JAX Privacy handles it.** In the default floating-point path,
*it mostly doesn't* — and it doesn't need to. This residual error is inherent to
doing DP in finite precision, and it is negligible: bounded by machine precision
and far below the noise floor of any realistic privacy claim. JAX Privacy does
not add special mitigations (e.g., deliberate safe-direction rounding) in the
standard path, so you should treat the reported $\epsilon$ as accurate only to
within your accountant's numerical tolerance, not to arbitrary precision.

If you need this residual slack *actually eliminated* rather than merely
bounded, opt into the integer-domain path (see
[Floating point robustness](#floating-point-robustness)): clipping and
quantization happen on an integer grid and the discrete Gaussian is defined over
the integers, so there is no floating-point rounding in the sensitivity bound or
the noise, and the guarantee holds *exactly*. This is the same opt-in path that
addresses the Theoretical floating-point *attack* surface — it closes both the
negligible and the adversarial versions of the finite-precision problem at once.
See the [secure noise example](https://github.com/google-parfait/jax_privacy/blob/main/examples/secure_noise_example.py)
for a complete working demonstration.

---

(design-principles-and-scope)=
## Design Principles and Scope

The items in this section are not failure modes you can trigger by accident.
They are cross-cutting properties of how JAX Privacy is built, together with
concerns that fall outside what the library can enforce. We group them here so
that the pitfalls above stay focused on concrete, accidental failure modes.

---

(flat-auditable-design)=
### Flat, Auditable Design

**Severity: Design.** *A cross-cutting design principle, not a failure mode on
its own — it is what makes the Critical pitfalls above auditable in isolation.*

**The pitfall.** When privacy-critical logic (clipping, noise addition,
sensitivity tracking) is spread across deeply nested abstraction layers, it
becomes difficult to audit. A bug in any layer can silently break the privacy
guarantee, and the more layers there are, the harder it is to verify
correctness.

**How JAX Privacy handles it.** JAX Privacy has a completely flat design: no
component depends on any other component. The clipping module, noise addition
module, batch selection module, and accounting module each stand alone and can
be understood, tested, and audited in isolation.

The coupling between components only happens at the higher-level API layer
(e.g., {class}`~jax_privacy.execution_plan.DPExecutionPlan`), where the joint
formal guarantees are explicitly stated. This means that auditing *or
contributing to* any individual component does not require understanding the
rest of the library — and auditing the composition requires understanding only
the thin integration layer, not the internals of each component.

---

### Framework Integration

**Severity: Design.** *Guidance on where privacy-critical logic should live, not
a failure mode on its own.*

**The pitfall.** Training frameworks are typically designed without DP in
mind. When you try to retrofit DP into such a framework, you are forced to
separate components that naturally need to be coupled in order to reason about
formal guarantees — batch selection ends up in the data pipeline, gradient
clipping in the optimizer, noise addition in a callback, and accounting in a
separate module. The result is spaghetti code where the privacy-critical
logic is spread across many files and abstraction layers, making it extremely
difficult to catch bugs.

We have experienced this firsthand. In one internal codebase built on a
training framework that was never designed for DP, it took *weeks* just to
understand how the different pieces fit together — not because the individual
components were complex, but because the framework's abstractions forced them
apart in unnatural ways. This is not an isolated experience; we have seen the
same pattern repeatedly with other framework integrations.

The framework also forces shortcuts that limit what you can do. For example,
a framework may assume that noise is added independently at each step, which
immediately rules out the correlated noise mechanisms that JAX Privacy
supports (e.g., matrix factorization). You end up constantly fighting against
the framework's assumptions to make the privacy mechanism work, and the
resulting code is brittle and hard to verify.

**How JAX Privacy handles it.** JAX Privacy operates at the pure JAX level.
The core API transforms *loss functions*, not training loops:

- {func}`~jax_privacy.clipping.clipped_grad` transforms a loss function into a
  clipped-gradient function.
- {func}`~jax_privacy.clipping.clipped_fun` transforms an arbitrary function
  into a clipped-output function.

This means you can write your training loop the way you would describe it in
a paper — and anyone can come in and say "yes, this is correct" or "no, this
is not correct," because the structure matches the mathematical description
rather than being distorted by framework abstractions.

We recommend using JAX Privacy's high-level training loops (e.g., the
[Keras API](keras_api.rst) or the
{class}`~jax_privacy.execution_plan.DPExecutionPlan`-based loop shown in our
[examples](examples_guide)) for the strongest guarantees. If the benefits
of a specific framework outweigh the benefits of JAX Privacy's built-in
training loops, you can still use the lower-level building blocks — JAX Privacy
will not prevent you from doing so. But based on our experience, the
framework-agnostic approach is less error-prone.

---

### Foot-gun APIs

**Severity: Design.** *A cross-cutting API principle: refuse to expose
injection points where a plausible-looking custom implementation could silently
break DP. The injectable `vmap` below is the canonical example.*

**The pitfall.** A flexible, extensible API is usually a virtue — but in a DP
library it can be a *foot-gun*. Some extension points look harmless yet let a
user supply an implementation that type-checks and appears to work, while
quietly violating a property the privacy guarantee depends on (per-example
isolation, the sensitivity bound, noise independence). Because the code runs and
produces plausible outputs, nothing signals that DP has been broken. The safest
design is often to *not* expose such a hook at all.

**The canonical example: an injectable `vmap`.** JAX's {func}`jax.vmap` is
central to how JAX Privacy computes per-example gradients: it vectorizes the
gradient computation across the batch dimension. A natural, flexible API
design would parameterize `clipped_grad` with a user-injectable `vmap` function,
letting users plug in custom variants (e.g., `shard_map` for distributed
settings).

This is exactly the kind of foot-gun described above. A user could plug in a
function that *satisfies the signature* of `vmap` but does not actually compute
per-example results independently — for instance, one that applies the
computation to the first example and replicates the result across the batch
dimension. This would silently break the per-example clipping guarantee: one
example's gradient would stand in for every "per-example" gradient, exposing
that individual to far higher privacy risk while appearing to work correctly.

**How JAX Privacy handles it.** We deliberately chose *not* to parameterize the
`vmap` function. The {func}`~jax_privacy.clipping.clipped_grad` and
{func}`~jax_privacy.clipping.clipped_fun` functions always use {func}`jax.vmap`
internally (with an optional `spmd_axis_name` for distributed settings), so you
cannot accidentally break per-example isolation by plugging in a broken
vectorization function. More generally, this reflects a design principle we
apply throughout the library: *correctness over flexibility*. Where an
extension point would let a user silently invalidate DP, we prefer to close
it — accepting a small loss in configurability in exchange for an API that
cannot be misconfigured into a silent privacy break.

---

(opaque-code)=
### Opaque Code

**Severity: Design.** *An auditability principle, not a failure mode on its
own.*

**The pitfall.** If a DP library's implementation is opaque — closed-source,
poorly documented, or excessively complex — it is impossible for the broader
DP research community to verify that the implementation provides the claimed
guarantees. This is not a hypothetical concern: bugs in DP implementations
have been discovered in widely-used libraries, sometimes years after release.

Excessive complexity often arises from deep framework integration, where the
DP logic must conform to the framework's abstractions rather than following
the natural structure of the DP mechanism. The resulting code can be so
difficult to follow that even experienced DP researchers struggle to verify
its correctness.

**How JAX Privacy handles it.** JAX Privacy is fully open source under the
Apache 2.0 license. The codebase is designed for auditability:

- **Flat architecture.** No component depends on any other component, so each
  module can be audited independently.
- **Formal guarantees in docstrings.** Public APIs document their sensitivity
  guarantees and assumptions explicitly.
- **Exposed internals.** The `DPExecutionPlan` exposes the `DpEvent` so that
  users can independently verify the privacy accounting.
- **Minimal framework coupling.** The privacy-critical code does not depend on
  any training framework, eliminating an entire category of auditability
  concerns.

--------------------------------------------------------------------------------

(privacy-auditing-experimental-design-sharp-edges)=
### Privacy Auditing: Canary Sampling

**Severity: Critical for audit validity.** A faulty audit does not weaken the
privacy guarantee of the trained mechanism, but it can invalidate the empirical
privacy lower bound reported by the audit.

**The pitfall.** Member and non-member canaries must be IID or otherwise
exchangeable. Informally, before membership is randomized, swapping the labels
"held in" and "held out" should not change the distribution of the experiment.
It is not enough for the two groups to have the same size or similar summary
statistics.

A common failure mode is to obtain held-in canaries from one source, generator,
time period, or preprocessing pipeline and held-out canaries from another. The
audit may then distinguish the two groups because of this distribution shift,
not because training on a canary changed the released model. Since the analysis
interprets score separation as evidence of membership leakage, the resulting
privacy lower bound can be spuriously large.

Canaries need not be literally independent if the experimental design provides a
defensible exchangeability argument and the statistical bound matches that
design. But when the groups are hand-selected, generated under different
conditions, or filtered using group-specific rules, the standard membership
interpretation no longer follows automatically. Any unavoidable shift must be
modeled or corrected explicitly rather than treated as membership signal.

**How JAX Privacy handles it.** JAX Privacy accepts held-in and held-out canary
scores, but cannot enforce how the canaries were sampled. A robust workflow is
to construct one candidate pool using a common generation and preprocessing
pipeline, randomize membership assignments only afterward, and apply the same
scoring procedure to both groups. You are responsible for documenting this
assignment procedure and for ensuring that the canaries are IID or otherwise
exchangeable; the library cannot detect source, preprocessing, or selection bias
from the score arrays alone.

--------------------------------------------------------------------------------

(privacy-auditing-one-run-vs-multi-run)=
### Privacy Auditing: Multi-Run vs. One-Run Bounds

**Severity: Critical for audit validity.** Using a bound for the wrong audit
design can invalidate the empirical privacy lower bound.

**The pitfall.** Multi-run bounds assume independent executions on neighboring
datasets: each observation comes from a fresh training run with fresh mechanism
randomness. Scores from many canaries obtained from one trained model do not
satisfy that assumption. They share the same learned parameters, minibatch
history, optimizer trajectory, injected noise, and other run-level randomness,
so their errors can move together.

Treating those scores as though they came from independent retrainings can make
the audit appear to have far more independent evidence than it actually has.
This typically produces an overconfident statistical bound and may inflate the
reported empirical privacy lower bound. The number of canaries in a one-run
audit is the number of randomized membership probes within one execution, not
the number of independent executions of the mechanism.

**How JAX Privacy handles it.** JAX Privacy is concerned only with one-run
auditing. The {meth}`~jax_privacy.auditing.CanaryScoreAuditor.epsilon_one_run`
and {meth}`~jax_privacy.auditing.CanaryScoreAuditor.epsilon_one_run_fdp` methods
are designed for randomized multi-canary, single-training-run audits and account
for the shared-run design. JAX Privacy does not provide a multi-run auditing
workflow; if you perform independent retrainings, you must use a multi-run
analysis whose assumptions match that experiment.

--------------------------------------------------------------------------------

(privacy-auditing-threshold-selection)=
### Privacy Auditing: Threshold Selection

**Severity: Critical for audit validity.** A faulty threshold-selection
procedure can invalidate the empirical privacy lower bound reported by the
audit.

**The pitfall.** A threshold converts a continuous canary score into a
membership attack: scores on one side are predicted to be members and scores on
the other side are predicted to be non-members. If you try many thresholds and
report the one producing the largest lower bound on the same data, ordinary
random fluctuations are selected along with genuine attack signal. The nominal
significance level then no longer applies unless the search is corrected.

**How JAX Privacy handles it.** JAX Privacy provides four threshold-selection
strategies:

-   {class}`~jax_privacy.auditing.Explicit` evaluates a threshold fixed
    independently of the audit scores.
-   {class}`~jax_privacy.auditing.Bonferroni`, the default, corrects for
    searching all candidate thresholds. It is simple and robust, but can be
    conservative when many thresholds are considered.
-   {class}`~jax_privacy.auditing.Split` uses one random partition to select the
    threshold and a disjoint partition to evaluate the bound. This avoids
    reusing the same observations for selection and evaluation, at the cost of
    less data in each stage.
-   {class}`~jax_privacy.auditing.MultiSplit` repeats the random split,
    evaluates each split at significance level $\alpha/2$, and reports the
    median lower bound. Repetition reduces dependence on one fortunate or
    unfortunate split, but it is still a controlled sample-splitting procedure
    rather than free reuse of the data.

---

### Other Potential Issues (Out of Scope)

Some important privacy concerns lie outside what JAX Privacy can enforce for
you. The most common is **hyperparameter tuning**: selecting the learning rate,
clip norm, noise multiplier, or batch size by repeatedly evaluating on private
data carries a privacy cost of its own, which the library does not track. If you
tune on private data, either account for that cost or tune on a public proxy
dataset, and report your procedure transparently. Section 5.3.3 of
[*How to DP-fy ML*](https://arxiv.org/pdf/2303.00654) gives a practical template
for this, and JAX Privacy's accounting tools can help you quantify the extra
cost.

---

## References

Many (though not all) of the pitfalls and auditing considerations above are
discussed in more depth in the DP practitioner literature. In particular:

- N. Ponomareva, H. Hazimeh, A. Kurakin, Z. Xu, C. Denison, H. B. McMahan,
  S. Vassilvitskii, S. Chien, and A. Thakurta.
  [*How to DP-fy ML: A Practical Guide to Machine Learning with Differential
  Privacy*](https://arxiv.org/pdf/2303.00654). Journal of Artificial
  Intelligence Research, 2023. A broad practical guide covering many of the same
  sharp edges, including sensitivity calibration, the relationship between
  accounting and batch selection, hyperparameter tuning (Section 5.3.3), and
  guarantee reporting.
- S. De, L. Berrada, J. Hayes, S. L. Smith, and B. Balle.
  [*Unlocking High-Accuracy Differentially Private Image Classification
  through Scale*](https://arxiv.org/abs/2204.13650). 2022. Introduces
  augmentation multiplicity, which averages gradients across several
  augmentations of one example before clipping.
- T. Steinke, M. Nasr, and M. Jagielski.
  [*Privacy Auditing with One (1) Training
  Run*](https://arxiv.org/abs/2305.08846). 2023. Introduces the randomized
  multi-canary, single-training-run auditing procedure.
- T. Cebere, M. Even, L. Bleistein, and A. Bellet.
  [*Privacy Auditing with Zero (0) Training
  Run*](https://arxiv.org/pdf/2605.14591). 2026. Shows how distribution shift
  between member and non-member data can confound privacy audits and proposes
  corrections for this setting.
- N. Meinshausen, L. Meier, and P. Bühlmann.
  [*P-values for high-dimensional
  regression*](https://arxiv.org/abs/0811.2177). 2009. Develops the repeated
  sample-splitting aggregation underlying the `MultiSplit` threshold strategy.
- T. Cebere, D. Erb, D. Desfontaines, A. Bellet, and J. Fitzsimons.
  [*Privacy in Theory, Bugs in Practice: Grey-Box Auditing of Differential
  Privacy Libraries*](https://arxiv.org/pdf/2602.17454). 2026. A gray-box
  auditing study that uncovered 13 privacy violations across 12 open-source DP
  libraries, concretely demonstrating how implementation bugs (e.g.,
  data-dependent control flow and sensitivity miscalibration) can silently break
  the theoretical guarantee — the motivation behind several pitfalls above and
  the [Opaque Code](#opaque-code) auditability principle.

---

## Citation

If you find this guide useful, you can cite it as:

```text
@misc{mckenna2026pitfalls,
  title        = {Common Pitfalls in DP Training},
  author       = {McKenna, Ryan and McMahan, H. Brendan and Cebere, Tudor},
  year         = {2026},
  howpublished = {JAX Privacy documentation},
  url          = {https://github.com/google-parfait/jax_privacy},
}
```

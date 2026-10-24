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

# Handling Variable Batch Sizes

A major differentiator between differentially private training loops and
standard training loops is the need to handle variable batch sizes. Handled
naively, these variable batch sizes could result in a large number of
compilations, and greatly affect the overall training performance. In the
sections below, we outline a few different strategies you can employ to reduce
this cost.

**Definitions: Global Batch Size vs. Minibatch Size vs. Microbatch Size**

**(Physical) Microbatch Size** (default = None): The minibatch will be split
up into smaller microbatches of this size, which will be sequentially
fed into the loss and gradient function using {func}`jax.lax.scan`. Can reduce
memory at increased sequential computation. Can be especially useful
when the size of the batch inputs is small relative to the intermediate
model activations, as in standard language modeling tasks.
This should generally be set as large as possible.

**(Physical) Minibatch Size** (default = None): This is the number of
elements that are grouped together into a single PyTree of input arrays
before computing and clipping the per-example gradients.

**Global Batch Size**: This is the overall batch size used to compute the
gradient for a single update step. It is what matters for privacy
calculations, and should be chosen to balance utility/compute
trade-offs. The global batch may never be materialized in memory, it
will be processed in chunks of size minibatch_size and further broken
down into chunks of size microbatch_size.

All three of microbatch size, minibatch size, and global batch size, may
vary from iteration to iteration depending on the batch selection strategy.
There are several ways one might try to deal with this enumerated below.

## Approach 1: Pay for recompilation.

Our function to compute the value and clipped gradients is compiled with
respect to a fixed minibatch size and microbatch size. If different
values are encountered, we must recompile the function for the new static
shapes. Often the compilation cost is significantly longer than the
training step time, although this cost is typically amortized over many
training iterations. Recompilation by itself is not a viable approach
unless coupled with a strategy to ensure that the number of different
shapes is relatively small compared to the number of training iterations.

## Approach 2: Pay for padding

While the "physical" minibatch size must be fixed to avoid recompilation
costs, it may consist of "real" and "padding" batch elements, where the
padding batch elements will contribute 0s to the aggregate gradient.
Our function for processing minibatch gradients can ensure that the
contributions from padding elements are zeroed out.

The global batch size can be any multiple of the minibatch size without
paying recompilation costs. Thus, the "cost" of this approach is the
unnecessary compute used to process the padding batch elements, which can
be up to O(minibatch_size). This can be significant depending on how large
the global batch size is relative to the minibatch size.

## Approach 3: Pay for padding + early stopping

We can reduce this cost further from the minibatch size to the microbatch
size, by incorporating dynamic early-stopping once a microbatch with all
padding elements are encountered. Thus, even though the shapes of the
inputs to our compiled function are static, the amount of compute the
function does is dynamic and depends on which batch elements are labeled
as padding examples. This offers a strict improvement over Approach 2, at
the expense of a more complex implementation.

## Approach 4: Truncation + new privacy analysis

In all three approaches above, the global batch size can be any multiple
of the microbatch size without paying for recompilation costs. If we also
require (or desire) a fixed global batch size, then there may be some
chance that the batch size we need is larger than the fixed batch size we
are constrained by, which is problematic. This can be handled by randomly
removing batch elements to reduce to the desired global batch size. This
random dropping of elements requires careful care when doing the privacy
analysis. The cost of this approach is the slack in the privacy analysis
needed to handle truncation, the gap between expected and physical batch
sizes to ensure this occurs with low probability, and relatedly the
processing of padding examples when the sampled batch size is less than
the physical global batch size.

## Approach 5: Hybrid recompilation + padding

Approaches 1, 3, and 4 all have their merits, and the best approach in
terms of compute utilization may vary from setting to setting. These
approaches represent extremes, and a hybrid approach that operates
between them can potentially offer better compute utilization. For
instance, if we allow up to K recompilations for small K, and doing so
greatly reduces the amount of padding batch elements we have to process,
this can be a worthwhile trade-off. Choosing the value of K and the
minibatch/microbatch sizes needed to minimize this cost is the main
challenge to solve here.

(using-is-padding-example)=
## Using `is_padding_example`

The approaches above that involve padding (Approaches 2–5) all require a
mechanism to tell the clipping function which batch elements are real and
which are synthetic padding. This is accomplished via the
`is_padding_example` keyword argument, which is accepted by the
{class}`~jax_privacy.clipping.BoundedSensitivityCallable` returned by
{func}`~jax_privacy.clipping.clipped_grad` and
{func}`~jax_privacy.clipping.clipped_fun`.

### What it is

`is_padding_example` is a 1D boolean `jax.Array` of shape
`(batch_size,)`, where `True` marks a padding example and `False` marks a
real example. Examples marked as padding are zeroed out after clipping
but before summation, so they contribute exactly zero to the aggregated
output. This preserves the formal DP sensitivity guarantee: the L2
sensitivity remains the same regardless of how many padding examples are in
the batch. Floating-point leaves of per-example auxiliary outputs (`aux`,
`norms`, `values`, `grad_norms`) at padding positions are set to `NaN` so they
can be aggregated (for example, via `jnp.nanmean`) without bias from padding.

When `is_padding_example` is omitted (the default), all examples are
treated as real.

### The `index == -1` convention

The standard convention in JAX Privacy is that an index of `-1` in a
batch of indices denotes a padding position. The `is_padding_example`
mask is then derived as:

```python
is_padding_example = (indices == -1)
```

This convention is used consistently by
{func}`~jax_privacy.batch_selection.pad_to_multiple_of`,
`_data_loading.batch_iterator`, and all reference examples.

When using the indices returned by `pad_to_multiple_of` to form a batch,
padding with `-1` generally means the *last* example in the training dataset
will be used many times as a padding example (if your dataset doesn't support
negative indexing, you will need to rewrite the -1 indices returned by
`pad_to_multiple_of` to point to a valid padding example). In either case, this
padding example should ideally be set to a predefined public example, both for
clarity and as a defense-in-depth measure; however, the formal guarantees are
intended to hold even if this is not the case.

### API tiers

The different API tiers handle `is_padding_example` differently:

**Core API** ({func}`~jax_privacy.clipping.clipped_grad` /
{func}`~jax_privacy.clipping.clipped_fun`): Users must construct and
pass `is_padding_example` explicitly as a keyword argument. This is the
most flexible tier and is used by all the non-Keras example scripts.

```python
grad_fn = jax_privacy.clipped_grad(loss_fn, l2_clip_norm=1.0,
                                   batch_argnums=(1, 2))

idx = batch_selection.pad_to_multiple_of(batch_idx, PADDING_MULTIPLE)
is_padding_example = idx == -1
batch = features[idx], labels[idx]

clipped_grads = grad_fn(params, *batch,
                        is_padding_example=is_padding_example)
```

**`DPTrainer`** ({class}`~jax_privacy.training.DPTrainer` in
{mod}`jax_privacy.training`): Most users should be able to use
{meth}`~jax_privacy.training.DPTrainer.fit`, in which case padding is handled
automatically. If the {meth}`~jax_privacy.training.DPTrainer.train_step` method
is used directly, it accepts `is_padding_example` and passes it through to
{func}`~jax_privacy.clipping.clipped_grad` internally.

**Keras API** ([Keras API](keras_api.rst) /
{mod}`jax_privacy.keras_api`): Padding is handled entirely automatically.
When `poisson_sampling_in_fit=True`, the Keras integration derives
`is_padding_example` from the `sample_weight` array (positions with weight 0 are
treated as padding). Users do not need to interact with `is_padding_example`
directly.

### Interaction with microbatching

When `microbatch_size` is set, the batch is split into microbatches that
are processed sequentially. If a microbatch consists entirely of padding
examples, the library detects this via `_num_real_microbatches` and
skips it, avoiding unnecessary compute. This corresponds to Approach 3
above.

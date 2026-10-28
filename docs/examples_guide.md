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

# Examples Guide

The
[`examples/`](https://github.com/google-parfait/jax_privacy/tree/main/examples)
directory contains runnable scripts and notebooks that demonstrate how to use
the JAX Privacy building blocks. The examples fall into two broad categories:

1. **End-to-end mechanism implementations** — complete DP training pipelines
   that compose clipping, noise addition, batch selection, and accounting into
   a single program.

2. **Isolated component demonstrations** — scripts that exercise or benchmark
   an individual building block (e.g., noise generation, data loading, or
   privacy accounting) without running a full training loop.

## A note on correctness

JAX Privacy provides the *building blocks* for writing correct DP
implementations, but that does not mean every program composed from those
blocks — **including some of our own examples** — uses them in the recommended
way.

Several examples aim to be end-to-end correct reference implementations.
These examples use proper Poisson sampling for batch selection (via
{class}`~jax_privacy.batch_selection.CyclicPoissonSampling` or the
[Keras API](keras_api.rst) with `poisson_sampling_in_fit=True`)
and treat the batch size as private information (normalizing by the *expected*
batch size rather than the actual batch size). If any bugs are found in these
files, we would consider those unexpected and serious. These examples are
marked with ✅ below.

All other end-to-end examples may take shortcuts or make simplifications that
could invalidate the formal privacy guarantee. These are known limitations
and exist by design so the examples remain accessible for research and
demonstration purposes. The most common issues are:

* **Not using Poisson sampling during training.** Several examples use
  fixed-size batches via `tf.data` shuffling or do not opt into Poisson
  sampling in the Keras API, even though the privacy accounting assumes
  Poisson-sampled batches.

* **Treating the (possibly padded) batch size as public information**, for
  example by dividing gradients by the actual batch size inside the
  JIT-compiled update step, when it should be treated as private.

We intend to improve more examples — particularly the Gemma fine-tuning
notebooks — over time. Until then, treat the examples below accordingly.

---

## End-to-end mechanism implementations

These examples compose multiple JAX Privacy building blocks into a full DP
training pipeline.

### `dp_logistic_regression.py` ✅

Trains a logistic regression model on synthetic data using DP-BandMF.
Demonstrates the full pipeline:
{class}`~jax_privacy.execution_plan.BandMFConfig` for calibration
and plan construction,
{class}`~jax_privacy.batch_selection.CyclicPoissonSampling` for batch selection
with proper padding, clipped gradient computation, correlated noise addition,
and post-training privacy auditing with canary scores.

**Correctness status:** Aims to be a correct reference implementation.

### `secure_noise_example.py` ✅

Trains a logistic regression model using DP-SGD with the discrete Gaussian
mechanism backed by cryptographically secure CPU-side randomness (via
`randomgen.RDRAND`). Uses
{class}`~jax_privacy.batch_selection.CyclicPoissonSampling` for Poisson-sampled
batches, integer-grid rounding of clipped gradients, and RDP accounting.

**Correctness status:** Aims to be a correct reference implementation.

### `dp_sgd_transformer.py` ✅

Trains a character-level Transformer decoder on the Tiny Shakespeare dataset
using DP-SGD with {class}`~jax_privacy.execution_plan.BandMFConfig`.
Demonstrates next-character prediction with per-sequence privacy. Uses
{class}`~jax_privacy.batch_selection.CyclicPoissonSampling` (via the execution
plan) for batch selection and normalizes by the expected batch size.

**Correctness status:** Aims to be a correct reference implementation.

### `keras_api_example.py` ✅

Trains a CNN on MNIST using the JAX Privacy Keras integration
({class}`~jax_privacy.keras_api.DPKerasConfig` /
{func}`~jax_privacy.keras_api.make_private`) with
`poisson_sampling_in_fit=True`. Demonstrates how to enable DP-SGD with
minimal code changes on top of a standard Keras training loop.

**Correctness status:** Aims to be a correct reference implementation.

### `jax_api_example.py`

Trains a linear regression model on synthetic data using the JAX Privacy
core API ({func}`~jax_privacy.clipping.clipped_grad`,
{func}`~jax_privacy.noise_addition.gaussian_privatizer`, PLD-based noise
calibration). Shows both DP and non-DP training paths for comparison.

**Correctness status:** Research / demonstration. Uses `tf.data` shuffling with
fixed-size batches (`drop_remainder=True`) rather than Poisson sampling. Noise
is added to the clipped gradient **sum**, then the result is divided by the
fixed batch size.

### `dp_sgd_flax_linen_mnist.ipynb` (notebook)

Step-by-step Colab tutorial that trains a Flax Linen CNN on MNIST with
DP-SGD. Walks through hyper-parameter setup, PLD-based noise calibration,
per-example gradient clipping via
{func}`~jax_privacy.clipping.clipped_grad`, and comparing DP vs. non-DP
accuracy.

**Correctness status:** Research / demonstration. Uses `tf.data` shuffling
with fixed-size batches rather than Poisson sampling.

### `dp_sgd_keras_gemma3_lora_finetuning_samsum.ipynb` (notebook)

Colab tutorial for DP-SGD LoRA fine-tuning of Gemma 3 (4B) on the SAMSum
summarization dataset using the Keras API. Covers data preprocessing,
mixed-precision training, enabling DP via
{class}`~jax_privacy.keras_api.DPKerasConfig`, and ROUGE evaluation.

**Correctness status:** Research / demonstration. Does not enable
`poisson_sampling_in_fit` in the Keras API configuration.

### `dp_sgd_keras_gemma3_synthetic_data.ipynb` (notebook)

Colab tutorial for generating differentially private synthetic data by
DP fine-tuning Gemma 3 (12B) with LoRA on IMDb reviews and then sampling
from the tuned model. Includes MAUVE-based evaluation of synthetic data
quality.

**Correctness status:** Research / demonstration. Does not enable
`poisson_sampling_in_fit` in the Keras API configuration.

---

## Isolated component demonstrations

These examples focus on a single building block and do not implement a full
DP training loop.

### `saliency_probe.py`

Runs `jax_privacy.experimental.saliency.topk_vote_probe` on a tiny MLP
with synthetic data. Prints the selected parameter leaves and noisy vote
scores, constructs the full-batch probe's Gaussian event, and composes it with
an illustrative Poisson-sampled DP-SGD training event using
{func}`~jax_privacy.accounting.dpsgd_event`.

Run from the repository root after installing the library:

```bash
python -m pip install -e .
JAX_PLATFORMS=cpu python examples/saliency_probe.py
```

No model or dataset downloads are needed. Noise uses fresh private randomness,
so scores and selected leaves can vary between runs. The fixed data seed and
model initialization are public.

**Accounting scope:** The probe uses the entire synthetic batch, with no
sampling amplification. Under add/remove-one adjacency, the printed combined
budget covers this probe followed by the stated DP-SGD plan on the same
population: 20 steps, independent Poisson sampling with probability 0.125,
and noise multiplier 2.0. Training is not executed; an actual fine-tune must
match this plan and use independent noise. This illustrates API usage and
accounting, not a comparison of fine-tuning quality.

### `data_loading.py`

Demonstrates how to integrate
{class}`~jax_privacy.batch_selection.BatchSelectionStrategy` with
[PyGrain](https://github.com/google/grain) for efficient data loading from
on-disk datasets. Includes a reusable `CustomBatchIterator` class with
checkpointing support, and benchmarks throughput comparing the custom
iterator against a standard PyGrain pipeline.

### `distributed_noise_generation.py`

Standalone tool for benchmarking distributed correlated noise generation
with DP-BandMF. Visualizes array sharding across a TPU mesh and reports
compilation time, per-step runtime, and memory usage. Useful for
determining feasibility of a given model on a given TPU topology.

### `dpmf_strategy_optimization.py`

Numerically optimizes DP Matrix Factorization (DP-MF) strategy matrices
and evaluates the expected error of different parameterizations (dense,
banded, Toeplitz, BLT, etc.). The reported errors correspond to
*unamplified* DP-MF and can be used to compare strategy quality across
different configurations.

### `balls_in_bins_accounting.py`

Uses Monte Carlo accounting to calibrate the noise multiplier for DP-SGD
under the "balls-in-bins" batch selection strategy. Demonstrates importance
sampled privacy loss generation, noise multiplier sweeping, and
verification-based calibration from the
{mod}`jax_privacy.experimental.monte_carlo` module.

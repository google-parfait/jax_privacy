# coding=utf-8
# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Demonstrates cryptographically secure DP-SGD with discrete Gaussian noise.

This example implements DP-SGD using the discrete Gaussian mechanism with
cryptographically secure CPU-side randomness. The key steps are:

  1. Compute per-example gradients.
  2. Clip, scale, and round each per-example gradient to an integer grid.
  3. Sum the rounded integer gradients across examples.
  4. Add discrete Gaussian noise (integer-valued, generated on CPU).
  5. Scale back to float and normalize by expected batch size.

For cryptographic security, the PRNG is backed by hardware-level entropy
(RDRAND) via ``randomgen.RDRAND()``. If ``randomgen`` is not installed, the
example falls back to a standard NumPy PRNG with a warning.

Privacy accounting is out of scope for this example, as Poisson-sampled discrete
Gaussian noise is not supported by the accounting libraries we use.
"""

import time
from typing import Mapping
import warnings

from absl import app
import jax
import jax.numpy as jnp
import jax_privacy
from jax_privacy import batch_selection
from jax_privacy.experimental import discrete_gaussian
import numpy as np
import optax


# Enable 64-bit mode for int64 support. With GRID_SCALE=10^9, the sum of
# per-example integer gradients and the discrete Gaussian noise can exceed
# int32 range (~2.1e9).
jax.config.update('jax_enable_x64', True)


USERS = 10_000
FEATURES = 100
ITERATIONS = 500
EXPECTED_BATCH_SIZE = 1000
PADDING_MULTIPLE = 32
NOISE_MULTIPLIER = 1.0
LEARNING_RATE = 0.2
L2_CLIP_NORM = 1.0
GRID_SCALE = 10**9  # Number of integer grid steps per L2_CLIP_NORM.


def _create_secure_prng() -> np.random.Generator:
  """Creates a cryptographically secure PRNG, falling back to standard NumPy."""
  try:
    # Make sure this example still works in environments where randomgen is
    # not installed.
    # pylint: disable-next=g-import-not-at-top,import-outside-toplevel
    import randomgen  # pyrefly: ignore[missing-import]

    rng = randomgen.RDRAND()
    # randomgen.RDRAND is a numpy BitGenerator at runtime, but its stubs do
    # not declare that, so pyrefly wrongly rejects the argument.
    return np.random.Generator(rng)
  except ImportError:
    warnings.warn(
        'randomgen is not installed. Falling back to a standard NumPy PRNG. '
        'This is NOT cryptographically secure. Install randomgen for '
        'production use: pip install randomgen',
        stacklevel=2,
    )
    return np.random.default_rng(seed=0)


def elementwise_loss(
    params: Mapping[str, jax.Array],
    feature_matrix: jax.Array,
    labels: jax.Array,
) -> jax.Array:
  """Computes element-wise loss for auditing."""
  logits = jnp.dot(feature_matrix, params['weights']) + params['bias']
  signed_logits = jnp.where(labels, logits, -logits)
  return -jax.nn.log_sigmoid(signed_logits)


def logistic_loss(
    params: Mapping[str, jax.Array],
    feature_matrix: jax.Array,
    labels: jax.Array,
) -> jax.Array:
  """Computes logistic loss."""
  return jnp.mean(elementwise_loss(params, feature_matrix, labels))


def create_benchmark(
    samples: int,
    features: int,
    seed: int = 0,
) -> tuple[Mapping[str, jax.Array], jax.Array, jax.Array]:
  """Creates a simple logistic regression model and training data."""
  key = jax.random.key(seed)
  data_key, params_key = jax.random.split(key)

  params = {
      'weights': jax.random.normal(params_key, (features,)),
      'bias': jnp.array(0.0),
  }

  feature_matrix = jax.random.normal(data_key, (samples, features))

  logits = jnp.dot(feature_matrix, params['weights']) + params['bias']
  labels = np.random.rand(samples) < jax.nn.sigmoid(logits)

  return params, feature_matrix, labels  # pyrefly: ignore[bad-return]


def main(_):
  """Main function."""
  true_params, all_features, all_labels = create_benchmark(USERS, FEATURES)
  init_params = jax.tree.map(jnp.zeros_like, true_params)

  optimizer = optax.sgd(LEARNING_RATE)
  opt_state = optimizer.init(init_params)

  prng = _create_secure_prng()

  grad_fn = jax_privacy.clipped_grad(
      logistic_loss,
      l2_clip_norm=L2_CLIP_NORM,
      batch_argnums=(1, 2),
      grid_scale=GRID_SCALE,
  )

  # The noise multiplier is relative to the L2 sensitivity. Since each rounded
  # per-example gradient has integer L2 norm at most GRID_SCALE, the L2
  # sensitivity of the sum (under add/remove) is GRID_SCALE.
  discrete_sigma = NOISE_MULTIPLIER * grad_fn.sensitivity()

  @jax.jit
  def train_step(
      params,
      opt_state,
      batch_features,
      batch_labels,
      is_padding_example,
      noise,
  ):
    """Executes a single training step with discrete Gaussian noise.

    Args:
      params: Current model parameters.
      opt_state: Current optimizer state.
      batch_features: Feature matrix for the batch, shape [batch, features].
      batch_labels: Labels for the batch, shape [batch].
      is_padding_example: Boolean array indicating padded examples, shape
        [batch].
      noise: Pre-generated discrete Gaussian noise (integer pytree).

    Returns:
      A tuple of (updated_params, updated_opt_state).
    """
    # Compute clipped, rounded integer gradients and sum across the batch.
    grads = grad_fn(
        params,
        batch_features,
        batch_labels,
        is_padding_example=is_padding_example,
    )
    # Add noise and rescale to float.
    noisy_grads = jax.tree.map(jnp.add, grads, noise)
    scale_down = L2_CLIP_NORM / (GRID_SCALE * EXPECTED_BATCH_SIZE)
    float_grads = jax.tree.map(
        lambda g: g.astype(jnp.float32) * scale_down, noisy_grads
    )

    updates, new_opt_state = optimizer.update(float_grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

  params = init_params
  # WARNING: Batch selection and noise generation happen on CPU
  # for cryptographic security.
  strategy = batch_selection.CyclicPoissonSampling(
      sampling_prob=EXPECTED_BATCH_SIZE / USERS, iterations=ITERATIONS
  )
  start_time = time.perf_counter()
  for step, batch_idx in enumerate(
      strategy.batch_iterator(num_examples=USERS, rng=prng)
  ):
    idx = batch_selection.pad_to_multiple_of(batch_idx, PADDING_MULTIPLE)
    # Mark padding positions so clipped_grad zeros them out before
    # aggregation, preserving the DP sensitivity guarantee. See
    # "Using is_padding_example" in the Variable Batch Sizes documentation.
    # TODO: b/543134538 - Add public padding example as last dataset entry.
    is_padding_example = idx == -1
    batch_features = all_features[idx]
    batch_labels = all_labels[idx]

    noise_pytree = jax.tree.map(
        jnp.asarray,
        discrete_gaussian.sample_discrete_gaussian_pytree(
            prng,
            sigma=discrete_sigma,
            pytree=params,
            dtype=np.int64,
        ),
    )

    params, opt_state = train_step(
        params,
        opt_state,
        batch_features,
        batch_labels,
        is_padding_example,
        noise_pytree,
    )
    if step > 0 and step % 20 == 0:
      loss = logistic_loss(params, all_features, all_labels)
      elapsed = time.perf_counter() - start_time
      print(f'Step {step}, Loss: {loss:.4f}, Elapsed Time (s): {elapsed:.4f}')
      start_time = time.perf_counter()


if __name__ == '__main__':
  app.run(main)

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

"""Trains a logistic regression model with DP-BandMF."""

import time
from typing import Any, Mapping, Tuple

from absl import app
import jax
import jax.numpy as jnp
from jax_privacy import auditing
from jax_privacy import batch_selection
from jax_privacy import execution_plan
import numpy as np
import optax

USERS = 10_000
FEATURES = 100
EPSILON = 1.0
DELTA = 1e-5
BANDS = 4
EXPECTED_BATCH_SIZE = 1000
ITERATIONS = 500
LEARNING_RATE = 0.2
L2_CLIP_NORM = 1.0
AUDIT_HOLDOUT_PERCENT = 0.1
# Trades-off recompilation with padding for variable batch sizes.
# PADDING_MULTIPLE = 1 recompiles update_fn for every batch size encountered.
# PADDING_MULTIPLE > 1 pads batches in range (kP, (k+1)P ] to size (k+1)P
PADDING_MULTIPLE = 32


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
  # JAX privacy will call this function with size 1 batches, so it doesn't
  # matter how we aggregate across the batch dimension here.
  # (jnp.mean and jnp.sum will produce the same result)
  return jnp.mean(elementwise_loss(params, feature_matrix, labels))


def create_benchmark(
    samples: int,
    features: int,
    seed: int = 0,
) -> Tuple[Mapping[str, jax.Array], jax.Array, jax.Array]:
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
  # Split data into training set (in-canaries) and audit set (out-canaries)
  audit_users = int(USERS * AUDIT_HOLDOUT_PERCENT)
  train_users = USERS - audit_users
  true_params, all_features, all_labels = create_benchmark(USERS, FEATURES)

  # Split dataset
  train_features, train_labels = (
      all_features[:train_users],
      all_labels[:train_users],
  )
  audit_held_out_features, audit_held_out_labels = (
      all_features[train_users:],
      all_labels[train_users:],
  )

  init_params = jax.tree.map(jnp.zeros_like, true_params)
  init_loss = logistic_loss(init_params, train_features, train_labels)
  print(f'Initial training loss: {init_loss:.3f}')

  config = execution_plan.BandMFConfig.default(
      iterations=ITERATIONS,
      num_bands=BANDS,
      l2_clip_norm=L2_CLIP_NORM,
      normalize_by=EXPECTED_BATCH_SIZE,
      expected_participations=EXPECTED_BATCH_SIZE * ITERATIONS / train_users,
  ).calibrate(epsilon=EPSILON, delta=DELTA)
  print('Initialized BandMFConfig')
  plan = config.make()
  grad_fn = plan.clipped_grad(logistic_loss, batch_argnums=(1, 2))

  optimizer = optax.sgd(LEARNING_RATE)

  @jax.jit
  def update_fn(
      params: Mapping[str, jax.Array],
      batch: Tuple[jax.Array, jax.Array],
      is_padding_example: jax.Array,
      noise_state: Any,
      opt_state: optax.OptState,
  ) -> Tuple[Mapping[str, jax.Array], Any, optax.OptState]:
    print('Compiling update_fn for batch size ', batch[0].shape[0])
    x, y = batch
    clipped_grad_sum = grad_fn(
        params, x, y, is_padding_example=is_padding_example
    )

    noisy_grad, noise_state = plan.noise_addition_transform.update(
        clipped_grad_sum, noise_state
    )
    updates, opt_state = optimizer.update(noisy_grad, opt_state)
    # pyrefly: ignore[bad-assignment]
    params = optax.apply_updates(params, updates)
    return params, noise_state, opt_state

  params = init_params
  noise_state = plan.noise_addition_transform.init(params)
  opt_state = optimizer.init(params)

  start_time = time.perf_counter()
  for step, batch_idx in enumerate(
      plan.batch_selection_strategy.batch_iterator(train_users)
  ):

    # Padding reduces the required number of compilations of update_fn.
    idx = batch_selection.pad_to_multiple_of(batch_idx, PADDING_MULTIPLE)
    # Mark padding positions so clipped_grad zeros them out before
    # aggregation, preserving the DP sensitivity guarantee. See
    # "Using is_padding_example" in the Variable Batch Sizes documentation.
    # TODO: b/543134538 - Add public padding example as last dataset entry.
    is_padding_example = idx == -1
    batch = train_features[idx], train_labels[idx]

    params, noise_state, opt_state = update_fn(
        params, batch, is_padding_example, noise_state, opt_state
    )

    if step > 0 and step % 20 == 0:
      loss = logistic_loss(params, train_features, train_labels)
      elapsed = time.perf_counter() - start_time
      print(f'Step {step}, Loss: {loss:.4f}, Elapsed Time (s): {elapsed:.4f}')
      start_time = time.perf_counter()

  final_loss = logistic_loss(params, train_features, train_labels)
  print(f'Final training loss:   {final_loss:.3f}')
  optimal_loss = logistic_loss(true_params, train_features, train_labels)
  print(f'Optimal training loss: {optimal_loss:.3f}')

  assert final_loss < init_loss, (
      f'Final loss {final_loss:.3f} is not less than initial loss'
      f' {init_loss:.3f}'
  )
  assert final_loss < optimal_loss + 0.2, (
      f'Final loss {final_loss:.3f} is too far from optimal loss'
      f' {optimal_loss:.3f}'
  )

  # Calculate scores (Initial Loss - Final Loss)
  audit_held_in_features = train_features[:audit_users]
  audit_held_in_labels = train_labels[:audit_users]
  in_init_loss = elementwise_loss(
      init_params, audit_held_in_features, audit_held_in_labels
  )
  in_final_loss = elementwise_loss(
      params, audit_held_in_features, audit_held_in_labels
  )
  in_scores = in_init_loss - in_final_loss

  out_init_loss = elementwise_loss(
      init_params, audit_held_out_features, audit_held_out_labels
  )
  out_final_loss = elementwise_loss(
      params, audit_held_out_features, audit_held_out_labels
  )
  out_scores = out_init_loss - out_final_loss

  auditor = auditing.CanaryScoreAuditor(in_scores, out_scores)

  estimated_epsilon = auditor.epsilon_raw_counts(min_count=10, delta=DELTA)

  # 99% upper confidence interval
  quantile_99 = auditing.BootstrapParams(quantiles=(0.99,))
  estimated_epsilon_99 = auditor.epsilon_raw_counts(
      min_count=10, delta=DELTA, bootstrap_params=quantile_99
  )[0]
  auroc_99 = auditor.attack_auroc(bootstrap_params=quantile_99)[0]

  print(f'Audited with {audit_users} held-in/held-out canaries.')
  print(f'Theoretical Epsilon: {EPSILON:.2f}')
  print(f'Estimated Epsilon:   {estimated_epsilon:.2f}')
  print(f'Estimated Epsilon (99% CI): {estimated_epsilon_99:.2f}')
  print(f'AUROC (99% CI):      {auroc_99:.4f}')

  assert (
      estimated_epsilon < EPSILON + 5.0
  ), f'Estimated epsilon {estimated_epsilon:.2f} is too high!'
  assert (
      estimated_epsilon_99 < EPSILON + 10.0
  ), f'Estimated epsilon (99% CI) {estimated_epsilon_99:.2f} is too high!'


if __name__ == '__main__':
  app.run(main)

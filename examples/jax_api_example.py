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

"""Example of using JAX Privacy API in raw JAX.

The examples shows how to train a simple linear regression model using
JAX Privacy API. The model is a simple linear regression model with two
parameters (w and b). The training is done using DP-SGD with a synthetic dataset
that is generated using a known w and b. The goal is to learn w and b from the
synthetic dataset and compare the learned parameters with the known w and b.

The expected final loss should be very close to zero, ~0.0005 and the learned
w and b should be very close to the true w and b (max absolute error should be
smaller than 0.3).
"""

from typing import Any, Mapping, Tuple

from absl import app
import dp_accounting
import jax
from jax import random
import jax.numpy as jnp
import jax_privacy
from jax_privacy import accounting
from jax_privacy import noise_addition
import tensorflow as tf


def init_model_params() -> Mapping[str, jax.Array]:
  """Initializes the model's weight (w) and bias (b)."""
  key = random.key(12)
  w_key, b_key = random.split(key)
  w = random.normal(w_key, ())
  b = random.normal(b_key, ())
  return {"w": w, "b": b}


def model(params: Mapping[str, jax.Array], x: jax.Array) -> jax.Array:
  """Defines the linear regression model."""
  return params["w"] * x + params["b"]


def loss_fn(
    params: Mapping[str, jax.Array], x: jax.Array, y: jax.Array
) -> jax.Array:
  """Calculates the mean squared error loss."""
  predictions = model(params, x)
  return jnp.mean((predictions - y) ** 2)


def load_data(
    num_samples: Any,
    true_w: float = 2,
    true_b: float = 1,
    noise_std: float = 0.1,
) -> Tuple[jax.Array, jax.Array]:
  """Generates synthetic linear regression data."""
  key = random.key(3)
  x = random.uniform(key, (num_samples,), minval=0.0, maxval=10.0)
  noise = noise_std * random.normal(key, (num_samples,))
  y = true_w * x + true_b + noise
  return x, y


def batch_dataset(
    x: jax.Array,
    y: jax.Array,
    batch_size: int,
) -> tf.data.Dataset:
  """Batches the data into batches of the given size."""
  return (
      tf.data.Dataset.from_tensor_slices((x, y))
      .shuffle(buffer_size=1024)
      .batch(batch_size, drop_remainder=True)
      .prefetch(tf.data.AUTOTUNE)
  )


def updated_model_params(
    model_params: Mapping[str, jax.Array],
    noisy_grads: Mapping[str, jax.Array],
) -> Mapping[str, jax.Array]:
  """Updates the model parameters using the given gradients."""
  lr = 0.001
  return jax.tree.map(lambda p, g: p - lr * g, model_params, noisy_grads)


def main(_):

  true_w = 2.0
  true_b = 1.0
  num_epochs = 100
  batch_size = 256
  clipping_norm = 1.0
  train_size = 10000
  use_dp = True

  # Marker to insert the main part of the example into ReadTheDocs.
  # [START example]
  x_train_full, y_train_full = load_data(train_size, true_w, true_b)
  model_params = init_model_params()

  # DP only begin.
  # Calculate noise_multiplier (stddev) given the privacy budget.
  num_updates = num_epochs * train_size // batch_size
  make_event = lambda sigma: accounting.dpsgd_event(
      sigma, num_updates, sampling_prob=batch_size / train_size
  )
  noise_multiplier = dp_accounting.calibrate_dp_mechanism(
      dp_accounting.pld.PLDAccountant,
      make_event,
      target_epsilon=1.0,
      target_delta=1e-5,
  )
  grad_and_value_fn = jax_privacy.clipped_grad(
      loss_fn,
      l2_clip_norm=clipping_norm,
      batch_argnums=(1, 2),
      has_aux=False,
      return_values=True,
  )
  sensitivity = grad_and_value_fn.sensitivity(
      dp_accounting.NeighboringRelation.REPLACE_ONE
  )
  privatizer = noise_addition.gaussian_privatizer(
      stddev=noise_multiplier * sensitivity, rng=42
  )
  noise_state = privatizer.init(model_params)

  @jax.jit
  def dp_update_step(
      model_params: Mapping[str, jax.Array],
      batch_x: jax.Array,
      batch_y: jax.Array,
      noise_state: Any,
  ) -> Tuple[Mapping[str, jax.Array], jax.Array, Any]:
    """Updates the model parameters using DP-SGD."""
    grads, aux_outputs = grad_and_value_fn(model_params, batch_x, batch_y)
    loss = aux_outputs.values.mean()
    # clipped_grad returns a sum; noise is calibrated to that sensitivity.
    noisy_sum, noise_state = privatizer.update(grads, noise_state)
    mean_grads = jax.tree.map(lambda x: x / batch_size, noisy_sum)
    # pyrefly: ignore[bad-argument-type]
    updated_params = updated_model_params(model_params, mean_grads)
    return updated_params, loss, noise_state

  # DP only end.

  @jax.jit
  def update_step(
      model_params: Mapping[str, jax.Array],
      batch_x: jax.Array,
      batch_y: jax.Array,
  ) -> Tuple[Mapping[str, jax.Array], jax.Array]:
    """Updates the model parameters without DP."""
    loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(
        model_params, batch_x, batch_y
    )
    model_params = updated_model_params(model_params, grads)
    return model_params, loss

  if use_dp:
    print(f"Using DP with noise multiplier {noise_multiplier}.")
  else:
    print("Not using DP.")

  print("\nStarting training...")
  for epoch in range(num_epochs):
    batched_dataset = batch_dataset(x_train_full, y_train_full, batch_size)

    epoch_loss = 0.0
    for batch_x_tf, batch_y_tf in batched_dataset:
      batch_x = jnp.asarray(batch_x_tf)
      batch_y = jnp.asarray(batch_y_tf)

      if use_dp:
        model_params, loss, noise_state = dp_update_step(
            model_params,
            batch_x,
            batch_y,
            noise_state,
        )
      else:
        model_params, loss = update_step(model_params, batch_x, batch_y)
      # [END example]

      epoch_loss += loss * batch_size

    avg_epoch_loss = epoch_loss / train_size

    if (epoch + 1) % 20 == 0:
      print(f"Epoch {epoch:4d}, Avg. Loss: {avg_epoch_loss:.4f}")

  print("\nTraining complete!")
  learned_w = model_params["w"]
  learned_b = model_params["b"]
  print(f"Learned parameters: w={learned_w:.4f}, b={learned_b:.4f}")
  print(f"True parameters: w={true_w:.4f}, b={true_b:.4f}")

  if use_dp:
    assert abs(model_params["w"] - true_w) < 0.6, "w is too far from true_w!"
    assert abs(model_params["b"] - true_b) < 0.6, "b is too far from true_b!"
  else:
    assert abs(model_params["w"] - true_w) < 0.1, "w is too far from true_w!"
    assert abs(model_params["b"] - true_b) < 0.1, "b is too far from true_b!"


if __name__ == "__main__":
  app.run(main)

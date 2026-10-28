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

import dataclasses
import os
import types
from unittest import mock

os.environ["KERAS_BACKEND"] = "jax"
# pylint: disable=g-import-not-at-top, wrong-import-position
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from jax_privacy import keras_api
import keras
import numpy as np
from scipy import stats

# pylint: enable=g-import-not-at-top, wrong-import-position


class KerasApiTest(parameterized.TestCase):

  def _get_params(self):
    return keras_api.DPKerasConfig(
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=10,
        gradient_accumulation_steps=1,
        train_steps=100,
        train_size=1000,
    )

  def test_validate_params(self):
    # Valid parameters, does not raise an error.
    valid_params = self._get_params()

    # Invalid epsilon
    with self.assertRaisesRegex(ValueError, "Expected epsilon=0.0 > 0"):
      dataclasses.replace(valid_params, epsilon=0.0)

    # Invalid delta
    with self.assertRaisesRegex(ValueError, r"Expected delta=0.0 in \(0, 1\]"):
      dataclasses.replace(valid_params, delta=0.0)
    with self.assertRaisesRegex(ValueError, r"Expected delta=2.0 in \(0, 1\]"):
      dataclasses.replace(valid_params, delta=2.0)
    with self.assertRaisesRegex(ValueError, r"Expected delta=nan in \(0, 1\]"):
      dataclasses.replace(valid_params, delta=float("nan"))

    # Invalid batch size
    with self.assertRaisesRegex(ValueError, "Expected batch_size=0 > 0"):
      dataclasses.replace(valid_params, batch_size=0)

    # Invalid clipping norm
    with self.assertRaisesRegex(ValueError, "Expected clipping_norm=0.0 > 0"):
      dataclasses.replace(valid_params, clipping_norm=0.0)

    # Invalid train steps
    with self.assertRaisesRegex(ValueError, "Expected train_steps=0 > 0"):
      dataclasses.replace(valid_params, train_steps=0)

    # Invalid train size
    with self.assertRaisesRegex(ValueError, "Expected train_size=0 > 0"):
      dataclasses.replace(valid_params, train_size=0)

    # Batch size must not exceed train size.
    with self.assertRaisesRegex(
        ValueError,
        "Batch size 1001 must be less than or equal to train size 1000",
    ):
      dataclasses.replace(valid_params, batch_size=1001)

    # Invalid noise multiplier
    with self.assertRaisesRegex(
        ValueError, "Expected noise_multiplier=0.0 > 0"
    ):
      dataclasses.replace(valid_params, noise_multiplier=0.0)

    # Noise multiplier is too small
    with self.assertRaisesRegex(
        ValueError,
        "Value error occurred while calculating epsilon",
    ):
      dataclasses.replace(valid_params, noise_multiplier=1e-10)

    # Noise multiplier exceeds privacy budget
    with self.assertRaisesRegex(
        ValueError,
        "Provided self.noise_multiplier=0.1 will lead to privacy budget exceed",
    ):
      dataclasses.replace(valid_params, noise_multiplier=0.1)

    # Gradient accumulation steps must be positive
    with self.assertRaisesRegex(
        ValueError,
        "Expected gradient_accumulation_steps=0 > 0",
    ):
      dataclasses.replace(valid_params, gradient_accumulation_steps=0)

    # Invalid microbatch size
    with self.assertRaisesRegex(
        ValueError,
        "Expected microbatch_size=0 > 0",
    ):
      dataclasses.replace(valid_params, microbatch_size=0)

    with self.assertRaisesRegex(
        ValueError,
        "Microbatch size 11 must be less than or equal to batch size 10",
    ):
      dataclasses.replace(valid_params, microbatch_size=11)

  def test_validate_noise_multiplier_uses_effective_batch_size(self):
    # Validation must account at the same sampling rate as
    # update_with_calibrated_noise_multiplier, i.e. effective_batch_size =
    # batch_size * gradient_accumulation_steps; the old code dropped the
    # gradient_accumulation_steps factor. With gas > 1 that under-accounts:
    # noise_multiplier=1.0 gives epsilon ~0.72 at the physical batch size (which
    # the old check wrongly accepted against the target epsilon of 1.1) but ~5.6
    # at the effective batch size, so it must be rejected.
    params = dataclasses.replace(
        self._get_params(), gradient_accumulation_steps=8
    )
    with self.assertRaisesRegex(
        ValueError, "will lead to privacy budget exceed"
    ):
      dataclasses.replace(params, noise_multiplier=1.0)

  def test_poisson_sampling_in_fit_defaults_to_disabled(self):
    self.assertFalse(self._get_params().poisson_sampling_in_fit)

  def test_effective_batch_size(self):
    params1 = dataclasses.replace(self._get_params(), batch_size=5)
    self.assertEqual(params1.effective_batch_size, 5)

    params2 = dataclasses.replace(params1, gradient_accumulation_steps=10)
    self.assertEqual(params2.effective_batch_size, 50)

  def test_dp_params_calculates_noise_multiplier(self):
    params = keras_api.DPKerasConfig(
        noise_multiplier=None,
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=10,
        gradient_accumulation_steps=1,
        train_steps=100,
        train_size=1000,
    )

    updated_params = params.update_with_calibrated_noise_multiplier()

    self.assertGreater(updated_params.noise_multiplier, 0.0)

  def test_add_dp_sgd_attributes(self):
    model = keras.Sequential([keras.layers.Dense(10, input_shape=(784,))])
    params = keras_api.DPKerasConfig(
        epsilon=1.1,
        delta=1e-5,
        batch_size=10,
        gradient_accumulation_steps=1,
        clipping_norm=1.0,
        train_steps=20,
        train_size=500,
        noise_multiplier=10.0,
        microbatch_size=1,
    )

    keras_api._add_dp_sgd_attributes(model, params)

    self.assertTrue(hasattr(model, "_dp_params"))
    self.assertEqual(model._dp_params, params)
    self.assertEqual(model._dp_noise_multiplier, params.noise_multiplier)

  def test_get_noise_multiplier_uses_config_value(self):
    model = keras.Sequential([keras.layers.Dense(10, input_shape=(784,))])
    params = dataclasses.replace(self._get_params(), noise_multiplier=2.5)
    private_model = keras_api.make_private(model, params)

    self.assertEqual(private_model.get_noise_multiplier(), 2.5)

  def test_get_noise_multiplier_calibrates_once(self):
    model = keras.Sequential([keras.layers.Dense(10, input_shape=(784,))])
    params = self._get_params()
    private_model = keras_api.make_private(model, params)

    noise_multiplier = private_model.get_noise_multiplier()
    self.assertIsNotNone(noise_multiplier)
    self.assertGreater(noise_multiplier, 0.0)
    self.assertEqual(private_model._dp_noise_multiplier, noise_multiplier)
    self.assertEqual(private_model.get_noise_multiplier(), noise_multiplier)

  @parameterized.named_parameters(
      ("no_rescale_no_clip", 100.0, 1, False, [-10.0, -20.0]),
      ("no_rescale_clip", 1.0, 1, False, [-0.44721362, -0.89442724]),
      # rescale=true: clipped_grads = grads / max(clipping_norm, norm(grads))
      ("rescale_big_clipping_norm", 100.0, 1, True, [-0.1, -0.2]),
      ("rescale_clipping_norm_1", 1.0, 1, True, [-0.44721362, -0.89442724]),
      ("rescale_clipping_norm_2", 2.0, 1, True, [-0.44721362, -0.89442724]),
      ("batch_size_5", 100.0, 5, False, [-10.0, -20.0]),
  )
  def test_clipped_grads(
      self,
      clipping_norm: float,
      batch_size: int,
      rescale_to_unit_norm: bool,
      expected_grads: list[float],
  ):
    dp_params = keras_api.DPKerasConfig(
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=clipping_norm,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        train_steps=20,
        train_size=500,
        rescale_to_unit_norm=rescale_to_unit_norm,
    )
    # turn off noise for testing
    object.__setattr__(dp_params, "noise_multiplier", 0.0)

    # The function is (a0*x0+a1*x1-4)^2, where a0, a1 = 3, -2, x0, x1 = 1, 2.
    # We compute gradient respect a0, a1, so
    # grad(f) = (2*x0*(a0*x0+a1*x1-4), 2*x1*(a0*x0+a1*x1-4)) =
    # (2*(3-4-4), 4*(3-4-4)) = (-10, -20).
    trainable_variables = [jnp.array([3.0, -2.0])]
    x = jnp.array([[1.0, 2.0] for _ in range(batch_size)])
    y = jnp.array([4.0] * batch_size)

    noise_rng = jax.random.PRNGKey(0)
    non_trainable_variables = [noise_rng]
    state = (trainable_variables, non_trainable_variables, [], [])
    data = (x, y, None)

    (loss, _), grads = keras_api._noised_clipped_grads(
        _compute_mse_loss_and_updates_fn,
        dp_params,
        state,
        data,
    )

    self.assertEqual(loss, jnp.array([25.0]))
    chex.assert_trees_all_close(grads[0], jnp.array(expected_grads))

  @parameterized.parameters((1.1, 200, 1), (10, 50, 10), (5, 100, 20))
  def test_noise_distribution(
      self, epsilon: float, clipping_norm: float, batch_size: int
  ):
    dp_params = keras_api.DPKerasConfig(
        epsilon=epsilon,
        delta=1e-5,
        clipping_norm=clipping_norm,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        train_steps=500,
        train_size=500,
        rescale_to_unit_norm=False,
    ).update_with_calibrated_noise_multiplier()

    # The function is (a0*x0+a1*x1-4)^2, where a0, a1 = 3, -2, x0, x1 = 1, 2.
    # We compute gradient respect a0, a1, so
    # grad(f) = (2*x0*(a0*x0+a1*x1-4), 2*x1*(a0*x0+a1*x1-4)) =
    # (2*(3-4-4), 4*(3-4-4)) = (-10, -20).
    trainable_variables = [jnp.array([3.0, -2.0])]
    x = jnp.array([[1.0, 2.0]] * batch_size)
    y = jnp.array([4.0] * batch_size)

    # Generate sample.
    sample = []
    for _ in range(100):  # ~5 seconds
      noise_rng = jax.random.PRNGKey(keras_api._get_random_int64())
      non_trainable_variables = [noise_rng]
      state = (trainable_variables, non_trainable_variables, [], [])
      data = (x, y, None)

      _, grads = keras_api._noised_clipped_grads(
          _compute_mse_loss_and_updates_fn,
          dp_params,
          state,
          data,
      )
      sample.append(np.array(grads[0]))

    # Check that noised gradients aligned with expected distribution. The
    # gradient is 2 element vector, and the first element ~N(-10, stddev**2),
    # the second ~N(-20, stddev**2).
    sample = np.stack(sample)
    stddev = dp_params.noise_multiplier * clipping_norm / batch_size
    self._check_distribution(sample[:, 0], -10, stddev)
    self._check_distribution(sample[:, 1], -20, stddev)

  def test_validate_optimizer_mismatched_gradient_accumulation_steps(self):
    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(1)])
    dp_params = dataclasses.replace(
        self._get_params(), gradient_accumulation_steps=10
    )
    model = keras_api.make_private(model, dp_params)
    optimizer = keras.optimizers.Adam(gradient_accumulation_steps=5)
    model.compile(optimizer=optimizer)
    with self.assertRaisesRegex(
        ValueError, "optimizer.gradient_accumulation_steps = 5 must be equal to"
    ):
      keras_api._validate_optimizer(model, dp_params)

  def test_fit_with_weighted_metrics(self):
    """Verifies that fit with weighted_metrics works.

    There was once a failure related to weighted_metrics introducing shape
    mismatches in auxiliary outputs, therefore we've added this test case.
    """
    batch_size = 2
    epochs = 1
    train_size = 4

    input_dim = 3
    features = 2
    classes = 4

    inputs = keras.Input(shape=(input_dim, features), dtype="float32")
    dense = keras.layers.Dense(classes)(inputs)

    model = keras.Model(
        inputs=inputs,
        outputs=dense,
    )

    # Inputs (x): (train_size, input_dim, features)
    x = np.zeros((train_size, input_dim, features), dtype=np.float32)
    # Targets (y): (train_size, input_dim) - per-step class indices
    y = np.zeros((train_size, input_dim), dtype=np.int32)
    # Sample weights: (train_size, input_dim) - per-step weights
    sample_weight = np.ones((train_size, input_dim), dtype=np.float32)

    dp_params = keras_api.DPKerasConfig(
        epsilon=100.0,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        train_steps=epochs * (train_size // batch_size),
        train_size=train_size,
        poisson_sampling_in_fit=True,
        seed=0,
    )

    model = keras_api.make_private(model, dp_params)

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    history = model.fit(
        x, y, batch_size=batch_size, epochs=epochs, sample_weight=sample_weight
    )

    accuracy_key = "sparse_categorical_accuracy"
    self.assertIn(accuracy_key, history.history)
    self.assertGreaterEqual(history.history[accuracy_key][-1], 0.0)

  def test_poisson_sampled_training_dataset_batches_and_masks_padding(self):
    x = np.arange(24).reshape(12, 2)
    y = np.arange(12)
    sample_weight = np.linspace(1.0, 2.1, 12)
    dp_params = keras_api.DPKerasConfig(
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=4,
        gradient_accumulation_steps=1,
        train_steps=20,
        train_size=len(x),
        noise_multiplier=10.0,
        seed=0,
    )
    dataset = keras_api._PoissonSampledTrainingDataset(
        x,
        y,
        sample_weight,
        dp_params=dp_params,
        steps_per_epoch=8,
    )

    realized_batch_sizes = []
    for i, padded_indices in enumerate(dataset._epoch_batches):
      batch_x, batch_y, batch_sample_weight = (
          keras.utils.unpack_x_y_sample_weight(dataset[i])
      )
      is_padding_example = keras_api._padding_mask_from_sample_weight(
          batch_sample_weight
      )
      valid_positions = padded_indices >= 0

      self.assertEqual(batch_x.shape[0] % dp_params.batch_size, 0)
      np.testing.assert_array_equal(is_padding_example, ~valid_positions)
      np.testing.assert_array_equal(
          batch_x[valid_positions], x[padded_indices[valid_positions]]
      )
      np.testing.assert_array_equal(
          batch_y[valid_positions], y[padded_indices[valid_positions]]
      )
      np.testing.assert_array_equal(
          batch_sample_weight[valid_positions],
          sample_weight[padded_indices[valid_positions]],
      )
      np.testing.assert_array_equal(
          batch_x[is_padding_example],
          np.zeros_like(batch_x[is_padding_example]),
      )
      np.testing.assert_array_equal(
          batch_sample_weight[is_padding_example],
          np.zeros_like(batch_sample_weight[is_padding_example]),
      )
      realized_batch_sizes.append(int(valid_positions.sum()))

    self.assertGreater(len(set(realized_batch_sizes)), 1)

  def test_poisson_sampled_training_dataset_generates_mask_sample_weights(self):
    x = np.arange(12).reshape(6, 2)
    dp_params = keras_api.DPKerasConfig(
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=3,
        gradient_accumulation_steps=1,
        train_steps=20,
        train_size=len(x),
        noise_multiplier=10.0,
        seed=0,
    )
    dataset = keras_api._PoissonSampledTrainingDataset(
        x,
        None,
        None,
        dp_params=dp_params,
        steps_per_epoch=4,
    )

    _, _, batch_sample_weight = keras.utils.unpack_x_y_sample_weight(dataset[0])
    is_padding_example = keras_api._padding_mask_from_sample_weight(
        batch_sample_weight
    )

    np.testing.assert_array_equal(
        batch_sample_weight,
        (~np.asarray(is_padding_example)).astype(np.float32),
    )

  def test_poisson_sampled_training_dataset_with_gradient_accumulation(self):
    train_size = 60
    batch_size = 4
    gas = 3  # gradient_accumulation_steps
    x = np.arange(train_size * 2).reshape(train_size, 2)
    dp_params = keras_api.DPKerasConfig(
        epsilon=100.0,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=batch_size,
        gradient_accumulation_steps=gas,
        train_steps=10,
        train_size=train_size,
        noise_multiplier=10.0,
        seed=42,
    )
    steps_per_epoch = 6  # 2 macro-steps * 3 microbatches
    with mock.patch.object(
        keras_api.batch_selection,
        "CyclicPoissonSampling",
        wraps=keras_api.batch_selection.CyclicPoissonSampling,
    ) as mock_sampling:
      dataset = keras_api._PoissonSampledTrainingDataset(
          x,
          None,
          None,
          dp_params=dp_params,
          steps_per_epoch=steps_per_epoch,
      )
      mock_sampling.assert_called_once_with(
          sampling_prob=mock.ANY,
          iterations=2,
      )
    self.assertLen(dataset, steps_per_epoch)

    # For each macro-step (accumulation cycle of gas microbatches), verify that
    # all valid (non-padding) indices are disjoint across the gas microbatches.
    for step in range(0, steps_per_epoch, gas):
      macro_indices = []
      for offset in range(gas):
        padded_indices = dataset._epoch_batches[step + offset]
        self.assertEqual(padded_indices.shape[0] % dataset._padding_multiple, 0)
        valid_indices = padded_indices[padded_indices >= 0]
        self.assertTrue(np.all(valid_indices < train_size))
        self.assertTrue(np.all(valid_indices >= 0))
        # Ensure no duplicates within a single microbatch slice.
        self.assertEqual(len(valid_indices), len(np.unique(valid_indices)))
        macro_indices.extend(valid_indices.tolist())

      # Ensure no duplicates across all microbatches in this accumulation step!
      self.assertEqual(len(macro_indices), len(set(macro_indices)))

    # When steps_per_epoch is not divisible by gas, ceil(7 / 3) = 3 macro steps.
    with mock.patch.object(
        keras_api.batch_selection,
        "CyclicPoissonSampling",
        wraps=keras_api.batch_selection.CyclicPoissonSampling,
    ) as mock_sampling:
      dataset_undivided = keras_api._PoissonSampledTrainingDataset(
          x,
          None,
          None,
          dp_params=dp_params,
          steps_per_epoch=7,
      )
      mock_sampling.assert_called_once_with(
          sampling_prob=mock.ANY,
          iterations=3,
      )
    self.assertLen(dataset_undivided, 7)

  def test_poisson_sampled_training_dataset_macro_steps_rng_progression(self):
    train_size = 60
    batch_size = 4
    gas = 3
    x = np.arange(train_size * 2).reshape(train_size, 2)
    seed = 42
    dp_params = keras_api.DPKerasConfig(
        epsilon=100.0,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=batch_size,
        gradient_accumulation_steps=gas,
        train_steps=10,
        train_size=train_size,
        noise_multiplier=10.0,
        seed=seed,
    )
    steps_per_epoch = 6  # 2 macro steps * 3 microbatches
    dataset = keras_api._PoissonSampledTrainingDataset(
        x,
        None,
        None,
        dp_params=dp_params,
        steps_per_epoch=steps_per_epoch,
    )
    # Epoch 0 was precomputed during __init__ (consuming 2 macro-steps from
    # RNG). Trigger epoch 1 (should consume the next 2 macro-steps from RNG).
    dataset.on_epoch_end()

    # Reconstruct expected epoch 1 microbatches from an independent reference
    # RNG.
    ref_rng = np.random.default_rng(seed)
    ref_strategy = keras_api.batch_selection.CyclicPoissonSampling(
        sampling_prob=dataset._sampling_prob,
        iterations=2,
    )
    # Drain epoch 0 (2 macro-steps):
    list(ref_strategy.batch_iterator(train_size, rng=ref_rng))
    # Epoch 1 (next 2 macro-steps):
    ref_macro_batches = list(
        ref_strategy.batch_iterator(train_size, rng=ref_rng)
    )
    expected_epoch_1_microbatches = []
    for macro_indices in ref_macro_batches:
      for slice_indices in np.array_split(np.asarray(macro_indices), gas):
        expected_epoch_1_microbatches.append(
            keras_api._pad_batch_indices(
                slice_indices, dataset._padding_multiple
            )
        )

    for i in range(steps_per_epoch):
      np.testing.assert_array_equal(
          dataset._epoch_batches[i],
          expected_epoch_1_microbatches[i],
      )

  def test_calculate_optimizer_steps_to_perform_in_fit_with_gas(self):
    train_size = 100
    batch_size = 10
    gas = 4
    # With steps_per_epoch = None: effective_batch_size = 40.
    # default_steps_per_epoch for optimizer steps = 100 // 40 = 2.
    steps = keras_api._calculate_optimizer_steps_to_perform_in_fit(
        train_size=train_size,
        batch_size=batch_size,
        epochs=3,
        initial_epoch=0,
        steps_per_epoch=None,
        gradient_accumulation_steps=gas,
    )
    self.assertEqual(steps, 2 * 3)

    # With explicit steps_per_epoch (e.g. 12 microbatches -> 3 optimizer steps)
    steps_explicit = keras_api._calculate_optimizer_steps_to_perform_in_fit(
        train_size=train_size,
        batch_size=batch_size,
        epochs=2,
        initial_epoch=0,
        steps_per_epoch=12,
        gradient_accumulation_steps=gas,
    )
    self.assertEqual(steps_explicit, 3 * 2)

  def test_fit_with_poisson_sampling_and_gradient_accumulation(self):
    train_size = 40
    batch_size = 4
    gas = 2
    epochs = 2
    # effective_batch_size = 8, steps per epoch = 40 // 8 = 5 optimizer steps.
    train_steps = epochs * (train_size // (batch_size * gas))
    x = np.random.uniform(0, 1, (train_size, 4)).astype(np.float32)
    y = np.random.uniform(0, 1, (train_size, 1)).astype(np.float32)

    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(1)])
    dp_params = keras_api.DPKerasConfig(
        epsilon=100.0,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=batch_size,
        gradient_accumulation_steps=gas,
        train_steps=train_steps,
        train_size=train_size,
        noise_multiplier=1.0,
        poisson_sampling_in_fit=True,
        seed=123,
    )
    model = keras_api.make_private(model, dp_params)
    optimizer = keras.optimizers.Adam(
        learning_rate=0.01, gradient_accumulation_steps=gas
    )
    model.compile(loss="mse", optimizer=optimizer)

    with mock.patch.object(
        keras_api,
        "_PoissonSampledTrainingDataset",
        wraps=keras_api._PoissonSampledTrainingDataset,
    ) as mock_dataset:
      history = model.fit(  # pylint: disable=not-callable
          x, y, batch_size=batch_size, epochs=epochs
      )
      mock_dataset.assert_called_once_with(
          mock.ANY,
          mock.ANY,
          mock.ANY,
          dp_params=dp_params,
          steps_per_epoch=10,
      )
    self.assertIn("loss", history.history)
    self.assertLen(history.history["loss"], epochs)
    self.assertEqual(
        keras_api._get_non_trainable_weight("_optimizer_steps", model)
        .numpy()
        .item(),
        train_steps * gas,
    )

    # Test with explicit steps_per_epoch passed to fit()
    model2 = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(1)])
    dp_params2 = dataclasses.replace(dp_params, train_steps=6)
    model2 = keras_api.make_private(model2, dp_params2)
    optimizer2 = keras.optimizers.Adam(
        learning_rate=0.01, gradient_accumulation_steps=gas
    )
    model2.compile(loss="mse", optimizer=optimizer2)
    with mock.patch.object(
        keras_api,
        "_PoissonSampledTrainingDataset",
        wraps=keras_api._PoissonSampledTrainingDataset,
    ) as mock_dataset2:
      model2.fit(  # pylint: disable=not-callable
          x, y, batch_size=batch_size, epochs=1, steps_per_epoch=4
      )
      mock_dataset2.assert_called_once_with(
          mock.ANY,
          mock.ANY,
          mock.ANY,
          dp_params=dp_params2,
          steps_per_epoch=4,
      )
    self.assertEqual(
        keras_api._get_non_trainable_weight("_optimizer_steps", model2)
        .numpy()
        .item(),
        4,
    )

  def test_pad_batch_indices_reifies_empty_poisson_draw(self):
    padded_indices = keras_api._pad_batch_indices(
        np.array([], dtype=np.int32), multiple=4
    )

    np.testing.assert_array_equal(
        padded_indices, np.array([-1, -1, -1, -1], dtype=np.int32)
    )

  def test_padding_mask_from_2d_sample_weight(self):
    sample_weight = np.array(
        [[1.0, 0.5], [0.0, 0.0], [2.0, 3.0]], dtype=np.float32
    )
    padding_mask = keras_api._padding_mask_from_sample_weight(sample_weight)

    np.testing.assert_array_equal(padding_mask, np.array([False, True, False]))

  def test_padding_mask_from_1d_sample_weight(self):
    sample_weight = np.array([1.0, 0.0, 2.0], dtype=np.float32)
    padding_mask = keras_api._padding_mask_from_sample_weight(sample_weight)

    np.testing.assert_array_equal(padding_mask, np.array([False, True, False]))

  def test_prepare_fit_kwargs_for_poisson_dataset(self):
    fit_kwargs = {
        "x": np.arange(6).reshape(3, 2),
        "y": np.arange(3),
        "sample_weight": np.ones(3),
        "batch_size": 3,
        "shuffle": True,
        "validation_split": 0.0,
        "steps_per_epoch": None,
        "epochs": 5,
    }
    poisson_dataset = object()

    rewritten_kwargs = keras_api._prepare_fit_kwargs_for_poisson_dataset(
        fit_kwargs,
        poisson_dataset=poisson_dataset,
    )

    self.assertEqual(rewritten_kwargs["x"], poisson_dataset)
    self.assertEqual(rewritten_kwargs["epochs"], 5)
    self.assertNotIn("y", rewritten_kwargs)
    self.assertNotIn("sample_weight", rewritten_kwargs)
    self.assertNotIn("batch_size", rewritten_kwargs)
    self.assertNotIn("shuffle", rewritten_kwargs)
    self.assertNotIn("validation_split", rewritten_kwargs)
    self.assertNotIn("steps_per_epoch", rewritten_kwargs)
    self.assertEqual(fit_kwargs["x"].shape, (3, 2))

  def test_masked_mean_ignores_padding_examples(self):
    values = jnp.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    is_padding_example = jnp.array([False, True, False])

    mean = keras_api._masked_mean(values, is_padding_example)

    np.testing.assert_allclose(mean, np.array([3.0, 30.0]))

  def test_masked_mean_returns_zero_for_all_padding(self):
    values = jnp.array([[1.0, 10.0], [3.0, 30.0]])
    is_padding_example = jnp.array([True, True])

    mean = keras_api._masked_mean(values, is_padding_example)

    np.testing.assert_allclose(mean, np.array([0.0, 0.0]))

  def test_dp_training_e2e_work(self):
    keras.utils.set_random_seed(42)
    train_size = 200
    batch_size = 100
    epochs = 5
    train_steps = 10  # 5 * (200 / 100)
    x, y = np.random.uniform(0, 1, (train_size, 4)), np.random.uniform(
        0, 1, train_size
    )
    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(1)])
    dp_params = keras_api.DPKerasConfig(
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        train_steps=train_steps,
        train_size=train_size,
        poisson_sampling_in_fit=True,
        seed=0,
    )
    model = keras_api.make_private(model, dp_params)
    model.compile(loss="mse", optimizer="adam")
    model.fit(x, y, epochs=epochs, batch_size=batch_size)  # pylint: disable=not-callable
    self.assertAlmostEqual(model.evaluate(x, y), 2, delta=2)

  def test_dp_training_exceeds_privacy_budget_raises_error(self):
    train_size = 200
    batch_size = 100
    train_steps = 28
    x, y = np.random.uniform(0, 1, (train_size, 4)), np.random.uniform(
        0, 1, train_size
    )
    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(1)])
    dp_params = keras_api.DPKerasConfig(
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        train_steps=train_steps,
        train_size=train_size,
        seed=0,
    )
    model = keras_api.make_private(model, dp_params)
    model.compile(loss="mse", optimizer="adam")

    # fit is wrapped and does arguments parsing therefore below we try setting
    # different arguments in different ways (in args and in kwargs).
    # optimizer steps per epoch = 200 / 100 = 2
    # batch_size is in args
    model.fit(x, y, batch_size, epochs=2)  # optimizer steps = 2 * 2 = 4  # pylint: disable=not-callable
    model.fit(  # pylint: disable=not-callable
        x, y, epochs=2, batch_size=batch_size
    )  # optimizer steps = 4 + 2 * 2 = 8
    # initial_epoch is set
    model.fit(  # pylint: disable=not-callable
        x, y, batch_size, initial_epoch=4, epochs=5
    )  # optimizer steps = 8 + 1 * 2 = 10
    # steps_per_epoch is set
    model.fit(  # pylint: disable=not-callable
        x, y, batch_size, epochs=3, steps_per_epoch=2
    )  # optimizer steps = 10 + 3 * 2 = 16
    with self.assertRaisesRegex(
        RuntimeError,
        "you will run out of privacy budget",
    ):
      # cannot be performed because 16 + 7 * 2 = 30 > 28
      model.fit(x, y, epochs=7, batch_size=batch_size)  # pylint: disable=not-callable

  def test_fit_with_missing_args(self):
    # Arrange.
    train_size = 64
    train_steps = 15
    dp_params = keras_api.DPKerasConfig(
        batch_size=32,
        gradient_accumulation_steps=1,
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        train_steps=train_steps,
        train_size=train_size,
        seed=0,
    )
    x, y = np.random.uniform(0, 1, (train_size, 4)), np.random.uniform(
        0, 1, train_size
    )
    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(1)])
    original_fit_fn = model.fit

    # Signature of fit function of Gemma model in Keras.
    # For example it doesn't have epochs argument.
    def _fit_fn_with_missing_args(  # pylint: disable=too-many-positional-arguments
        _,  # unused self
        x=None,
        y=None,
        batch_size=None,
        sample_weight=None,
        validation_data=None,
        validation_split=None,
        **kwargs
    ):
      return original_fit_fn(  # pylint: disable=not-callable
          x=x,
          y=y,
          batch_size=batch_size,
          sample_weight=sample_weight,
          validation_data=validation_data,
          validation_split=validation_split,
          **kwargs
      )

    model.fit = types.MethodType(_fit_fn_with_missing_args, model)
    model = keras_api.make_private(model, dp_params)
    model.compile()

    # Act & assert.
    # train_steps = epochs (=1) * train_size (=64) / batch_size (=32)= 2
    model.fit(x, y, batch_size=32)  # optimizer steps = 1 * 2 = 2  # pylint: disable=not-callable
    with self.assertRaisesRegex(
        RuntimeError,
        "you will run out of privacy budget",
    ):
      # cannot be performed because 2 + 7 * 2 = 16 > 15
      model.fit(x, y, epochs=7)  # pylint: disable=not-callable

  def test_fit_raises_error_if_dp_params_not_aligned_with_fit_args(self):
    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(1)])
    dp_params = keras_api.DPKerasConfig(
        batch_size=100,
        gradient_accumulation_steps=1,
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        train_steps=28,
        train_size=200,
        seed=0,
    )
    model = keras_api.make_private(model, dp_params)
    model.compile()
    with self.assertRaisesRegex(
        ValueError,
        "The batch size in the DP parameters is not equal to the batch size"
        " passed to fit()",
    ):
      model.fit(batch_size=101)  # pylint: disable=not-callable

  def test_fit_rejects_non_random_access_training_data(self):
    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(1)])
    dp_params = keras_api.DPKerasConfig(
        batch_size=100,
        gradient_accumulation_steps=1,
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        train_steps=28,
        train_size=200,
        poisson_sampling_in_fit=True,
        seed=0,
    )
    model = keras_api.make_private(model, dp_params)

    def data_generator():
      while True:
        yield np.zeros((100, 4)), np.zeros((100,))

    model.compile()
    with self.assertRaisesRegex(
        ValueError,
        "random-access array-like inputs",
    ):
      model.fit(data_generator())  # pylint: disable=not-callable

  def test_fit_allows_generator_when_poisson_sampling_in_fit_disabled(self):
    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(1)])
    dp_params = keras_api.DPKerasConfig(
        batch_size=100,
        gradient_accumulation_steps=1,
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        train_steps=2,
        train_size=200,
        seed=0,
    )
    model = keras_api.make_private(model, dp_params)

    def data_generator():
      while True:
        yield np.zeros((100, 4)), np.zeros((100,))

    model.compile(loss="mse", optimizer="adam")
    history = model.fit(  # pylint: disable=not-callable
        data_generator(),
        steps_per_epoch=1,
        epochs=1,
    )

    self.assertIn("loss", history.history)

  def test_fit_rejects_train_size_mismatch(self):
    train_size = 200
    x, y = np.random.uniform(0, 1, (train_size, 4)), np.random.uniform(
        0, 1, train_size
    )
    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(1)])
    dp_params = keras_api.DPKerasConfig(
        batch_size=100,
        gradient_accumulation_steps=1,
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        train_steps=28,
        train_size=train_size - 1,
        poisson_sampling_in_fit=True,
        seed=0,
    )
    model = keras_api.make_private(model, dp_params)
    model.compile(loss="mse", optimizer="adam")

    with self.assertRaisesRegex(
        ValueError,
        "The train size in the DP parameters is not equal to the size of the"
        " training data passed to fit",
    ):
      model.fit(x, y, batch_size=100)  # pylint: disable=not-callable

  def test_fit_rejects_validation_split(self):
    train_size = 200
    x, y = np.random.uniform(0, 1, (train_size, 4)), np.random.uniform(
        0, 1, train_size
    )
    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(1)])
    dp_params = keras_api.DPKerasConfig(
        batch_size=100,
        gradient_accumulation_steps=1,
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        train_steps=28,
        train_size=train_size,
        poisson_sampling_in_fit=True,
        seed=0,
    )
    model = keras_api.make_private(model, dp_params)
    model.compile(loss="mse", optimizer="adam")

    with self.assertRaisesRegex(
        ValueError,
        "validation_split is not supported for DP Keras training",
    ):
      model.fit(  # pylint: disable=not-callable
          x, y, batch_size=100, validation_split=0.1
      )

  def test_train_step_call_noised_clipped_grads(self):
    train_size = 200
    batch_size = 100
    epochs = 1
    train_steps = 2  # 1 * (200 / 100)
    x, y = np.random.uniform(0, 1, (train_size, 4)), np.random.uniform(
        0, 1, train_size
    )
    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(1)])
    dp_params = keras_api.DPKerasConfig(
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=100,
        gradient_accumulation_steps=1,
        train_steps=train_steps,
        train_size=train_size,
        poisson_sampling_in_fit=True,
        seed=0,
    )
    model = keras_api.make_private(model, dp_params)
    model.compile(loss="mse", optimizer="adam")

    with mock.patch.object(
        keras_api,
        "_noised_clipped_grads",
        wraps=keras_api._noised_clipped_grads,
    ) as mock_noised_clipped_grads:
      model.fit(x, y, epochs=epochs, batch_size=batch_size)  # pylint: disable=not-callable
      mock_noised_clipped_grads.assert_called_once()

  def _check_distribution(
      self, sample: np.ndarray, mean: float, stddev: float
  ) -> None:
    """Verifies the hypothesis that the sample comes from N(mean, stddev**2)."""
    size = len(sample)
    sample_expected_distribution = np.random.normal(mean, stddev, size=size)
    pvalue = stats.ks_2samp(sample_expected_distribution, sample).pvalue
    self.assertGreater(pvalue, 1e-5)  # expected flakiness 1e-5


def _compute_mse_loss_and_updates_fn(  # pylint: disable=too-many-positional-arguments
    params,
    non_trainable_variables,
    metrics_variables,
    x,
    y,
    sample_weight,
    training,
    optimizer_variables,
):
  """Compute mse loss and updates for a simple model.

  This function has the same signature as JaxTrainer.compute_loss_and_updates_fn
  in order to replace in testing of gradient computation.

  Args:
    params: The parameters of the model.
    non_trainable_variables: The non-trainable variables of the model. Not used.
    metrics_variables: The metrics variables of the model. Not used.
    x: The input data.
    y: The target data.
    sample_weight: The sample weight.
    training: Whether the model is in training mode. Not used.
    optimizer_variables: The optimizer variables of the model. Not used.

  Returns:
    (loss, aux), where
      loss is the loss value.
      aux is a tuple (y_pred, non_trainable_variables, metrics_variables).
  """
  del (
      non_trainable_variables,
      metrics_variables,
      training,
      optimizer_variables,
  )
  y_pred = jnp.sum(x * params[0], axis=1)
  if sample_weight is not None:
    loss_value = jnp.mean(sample_weight * (y_pred - y) ** 2)
  else:
    loss_value = jnp.mean((y_pred - y) ** 2)
  aux = (loss_value, y_pred, [], [])
  return loss_value, aux


if __name__ == "__main__":
  absltest.main()

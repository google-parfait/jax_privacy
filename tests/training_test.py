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
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import grain.python as grain
import jax
import jax.numpy as jnp
from jax_privacy import _compilation
from jax_privacy import _data_loading
from jax_privacy import _validate
from jax_privacy import batch_selection
from jax_privacy import execution_plan
from jax_privacy import noise_addition
from jax_privacy import training
import numpy as np
import optax


def _quadratic_loss(params, batch, prng):
  """Per-example quadratic loss with has_aux=True convention."""
  del prng
  loss = jnp.mean((params - batch) ** 2)
  return loss, {'loss': loss}


def _make_config(
    iterations,
    noise_multiplier=1.0,
    expected_participations=None,
):
  """Creates a simple BandMF config for testing."""
  if expected_participations is None:
    expected_participations = iterations
  return execution_plan.BandMFConfig.default(
      num_bands=1,
      iterations=iterations,
      noise_multiplier=noise_multiplier,
      expected_participations=expected_participations,
  )


@dataclasses.dataclass(frozen=True)
class _FixedPlanConfig:
  """An ``ExecutionPlanConfig`` that returns a pre-built plan (for tests)."""

  plan: execution_plan.DPExecutionPlan

  def make(self, performance_flags=None, noise_seed=None):
    del performance_flags, noise_seed  # The plan is already built.
    return self.plan


def _as_map_dataset(dataset):
  n = jax.tree.leaves(dataset)[0].shape[0]
  elements = [jax.tree.map(lambda x, i=i: x[i], dataset) for i in range(n)]
  return grain.MapDataset.source(elements)


_DATASET_TYPES = (
    ('pytree', lambda x: x),
    ('grain', _as_map_dataset),
)


class BatchIteratorTest(parameterized.TestCase):
  """Tests for _data_loading.batch_iterator."""

  @parameterized.named_parameters(*_DATASET_TYPES)
  def test_padding_zeros_out_padded_examples(self, to_dataset):
    raw = np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32
    )
    dataset = to_dataset(raw)
    strategy = batch_selection.FixedBatchSampling(batch_size=3, iterations=2)
    batches = list(
        _data_loading.batch_iterator(
            dataset, strategy, rng=0, pad_to_multiple_of=4
        )
    )
    self.assertLen(batches, 2)
    expected_indices = list(strategy.batch_iterator(4, rng=0))
    for (batch, is_padding), idx in zip(batches, expected_indices):
      self.assertEqual(batch.shape, (4, 2))
      np.testing.assert_array_equal(is_padding, [False, False, False, True])
      np.testing.assert_allclose(batch[:3], raw[idx])
      np.testing.assert_allclose(batch[3], [0.0, 0.0])

  @parameterized.named_parameters(*_DATASET_TYPES)
  def test_empty_batch(self, to_dataset):
    raw = np.arange(10, dtype=np.float32).reshape((5, 2))
    dataset = to_dataset(raw)
    strategy = batch_selection.CyclicPoissonSampling(
        sampling_prob=0.0, iterations=1
    )

    [(batch, is_padding)] = list(
        _data_loading.batch_iterator(
            dataset, strategy, rng=0, pad_to_multiple_of=4
        )
    )
    self.assertEqual(batch.shape, (0, 2))
    self.assertEqual(is_padding.shape, (0,))

  @parameterized.named_parameters(*_DATASET_TYPES)
  def test_pytree_dataset(self, to_dataset):
    raw = {
        'x': np.arange(12, dtype=np.float32).reshape((4, 3)),
        'y': np.arange(4, dtype=np.int32) + 10,
    }
    dataset = to_dataset(raw)
    strategy = batch_selection.FixedBatchSampling(batch_size=3, iterations=1)
    [(batch, is_padding)] = list(
        _data_loading.batch_iterator(
            dataset, strategy, rng=0, pad_to_multiple_of=4
        )
    )
    self.assertEqual(batch['x'].shape, (4, 3))
    self.assertEqual(batch['y'].shape, (4,))
    np.testing.assert_array_equal(is_padding, [False, False, False, True])
    np.testing.assert_allclose(batch['x'][3], np.zeros(3))
    self.assertEqual(int(batch['y'][3]), 0)

  @parameterized.named_parameters(*_DATASET_TYPES)
  def test_initial_step_fast_forwards(self, to_dataset):
    raw = np.arange(20, dtype=np.float32).reshape((10, 2))
    dataset = to_dataset(raw)
    strategy = batch_selection.FixedBatchSampling(batch_size=3, iterations=5)
    all_batches = list(
        _data_loading.batch_iterator(
            dataset, strategy, rng=42, pad_to_multiple_of=4
        )
    )
    resumed_batches = list(
        _data_loading.batch_iterator(
            dataset, strategy, rng=42, pad_to_multiple_of=4, initial_step=2
        )
    )
    self.assertLen(resumed_batches, 3)
    for (b1, p1), (b2, p2) in zip(all_batches[2:], resumed_batches):
      np.testing.assert_allclose(b1, b2)
      np.testing.assert_array_equal(p1, p2)

  @parameterized.named_parameters(*_DATASET_TYPES)
  def test_microbatch_size_reorders_padding(self, to_dataset):
    raw = np.arange(1, 9, dtype=np.float32).reshape((4, 2))
    dataset = to_dataset(raw)
    strategy = batch_selection.FixedBatchSampling(batch_size=2, iterations=1)
    [(batch, is_padding)] = list(
        _data_loading.batch_iterator(
            dataset,
            strategy,
            rng=0,
            pad_to_multiple_of=4,
            microbatch_size=2,
        )
    )
    self.assertEqual(batch.shape, (4, 2))
    np.testing.assert_array_equal(is_padding, [False, True, False, True])
    np.testing.assert_allclose(batch[1], [0.0, 0.0])
    np.testing.assert_allclose(batch[3], [0.0, 0.0])

  @parameterized.named_parameters(*_DATASET_TYPES)
  def test_multihost_checks_seed_and_dataset(self, to_dataset):
    raw = np.arange(8, dtype=np.int32).reshape((4, 2))
    dataset = to_dataset(raw)
    strategy = batch_selection.FixedBatchSampling(batch_size=2, iterations=1)
    with (
        mock.patch.object(jax, 'process_count', return_value=2),
        mock.patch.object(
            _validate.multihost_utils, 'assert_equal'
        ) as mock_assert_equal,
    ):
      [(batch, _)] = list(
          _data_loading.batch_iterator(dataset, strategy, rng=42)
      )
    mock_assert_equal.assert_called_once()
    _validate.multihost_utils.assert_equal(*mock_assert_equal.call_args[0])
    checked_seed, checked_crc = mock_assert_equal.call_args[0][0]
    self.assertEqual(checked_seed, np.random.default_rng(42).integers(2**31))
    expected_crc = _validate.zlib.crc32(raw[:1].tobytes()) % 2**31
    self.assertEqual(checked_crc, expected_crc)
    expected_idx = next(strategy.batch_iterator(4, rng=42))
    np.testing.assert_allclose(batch, raw[expected_idx])

  def test_grain_padding_example_overrides_element_zero(self):
    class _MismatchedZeroSource(grain.RandomAccessDataSource):

      def __len__(self):
        return 3

      def __getitem__(self, idx):
        if idx == 0:
          return {'x': np.zeros((99,), dtype=np.float32)}
        return {'x': np.array([float(idx), float(idx)], dtype=np.float32)}

    dataset = grain.MapDataset.source(_MismatchedZeroSource())
    strategy = mock.Mock(spec=batch_selection.BatchSelectionStrategy)
    strategy.batch_iterator.return_value = iter([np.array([1, 2])])
    spec = {'x': jax.ShapeDtypeStruct((2,), np.float32)}
    [(batch, is_padding)] = list(
        _data_loading.batch_iterator(
            dataset,
            strategy,
            rng=0,
            pad_to_multiple_of=4,
            padding_example=spec,
        )
    )
    self.assertEqual(batch['x'].shape, (4, 2))
    np.testing.assert_allclose(
        batch['x'], [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0]]
    )
    np.testing.assert_array_equal(is_padding, [False, False, True, True])

  def test_pytree_padding_example_validation(self):
    dataset = {'x': np.zeros((4, 2), dtype=np.float32)}
    strategy = batch_selection.FixedBatchSampling(batch_size=2, iterations=1)
    valid_spec = {'x': jax.ShapeDtypeStruct((2,), np.float32)}
    self.assertLen(
        list(
            _data_loading.batch_iterator(
                dataset, strategy, rng=0, padding_example=valid_spec
            )
        ),
        1,
    )
    with self.assertRaises(ValueError):
      list(
          _data_loading.batch_iterator(
              dataset,
              strategy,
              rng=0,
              padding_example={'x': jax.ShapeDtypeStruct((3,), np.float32)},
          )
      )
    with self.assertRaises(ValueError):
      list(
          _data_loading.batch_iterator(
              dataset,
              strategy,
              rng=0,
              padding_example={'y': jax.ShapeDtypeStruct((2,), np.float32)},
          )
      )


class DPTrainerTest(parameterized.TestCase):
  """Tests for the DPTrainer class."""

  def test_map_dataset_with_padding_example(self):
    """DPTrainer(padding_example=...) uses padding_example over dataset[0]."""

    class _MismatchedZeroSource(grain.RandomAccessDataSource):

      def __len__(self):
        return 3

      def __getitem__(self, idx):
        if idx == 0:
          return np.zeros((99,), dtype=np.float32)
        return np.array([float(idx), float(idx)], dtype=np.float32)

    dataset = grain.MapDataset.source(_MismatchedZeroSource())
    strategy = mock.Mock(spec=batch_selection.BatchSelectionStrategy)
    strategy.batch_iterator.side_effect = lambda n, rng=None: iter(
        [np.array([1, 2])]
    )
    plan = dataclasses.replace(
        _make_config(iterations=1).make(),
        batch_selection_strategy=strategy,
    )
    trainer = training.DPTrainer(
        config=_FixedPlanConfig(plan),
        loss_fn=_quadratic_loss,
        optimizer=optax.sgd(0.01),
        padding_example=jax.ShapeDtypeStruct((2,), np.float32),
    )
    state = trainer.fit(
        dataset, trainer.init(jnp.array([5.0, 5.0])), rng_or_seed=0
    )
    self.assertEqual(int(state.step), 1)

  @parameterized.named_parameters(*_DATASET_TYPES)
  def test_basic_training_runs(self, to_dataset):
    """Train loop completes and returns a valid TrainingState."""
    params = jnp.array([5.0, 5.0])
    dataset = to_dataset(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    config = _make_config(iterations=3)
    optimizer = optax.sgd(0.01)

    trainer = training.DPTrainer(
        config=config,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )
    state = trainer.fit(dataset, trainer.init(params), rng_or_seed=0)

    self.assertIsInstance(state, training.TrainingState)
    self.assertEqual(int(state.step), 3)

  @parameterized.product(
      to_dataset=[fn for _, fn in _DATASET_TYPES],
      compress=[False, True],
  )
  def test_resume_from_state_yields_identical_results(
      self, to_dataset, compress
  ):
    """Test that resuming from intermediate steps works as intended."""
    params = jnp.array([5.0, 5.0], dtype=jnp.bfloat16)
    dataset = to_dataset(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    )
    config = execution_plan.BandMFConfig.default(
        num_bands=3,
        iterations=5,
        noise_multiplier=0.1,
        expected_participations=1,
    )

    def make_trainer():
      return training.DPTrainer(
          config=config,
          loss_fn=_quadratic_loss,
          optimizer=optax.sgd(0.01),
          noise_seed=7 if compress else None,
      )

    trainer = make_trainer()
    intermediate_states = []

    def callback(step, state, _):
      del step
      intermediate_states.append(jax.tree.map(jax.numpy.copy, state))

    expected_final_state = trainer.fit(
        dataset, trainer.init(params), rng_or_seed=42, callback=callback
    )

    for state in intermediate_states:
      resumed_trainer = make_trainer() if compress else trainer
      if compress:
        compressed = state.compress()
        self.assertIsNone(compressed.noise_state)
        state = compressed.restore(resumed_trainer)
      final_state_resumed = resumed_trainer.fit(dataset, state, rng_or_seed=42)
      np.testing.assert_allclose(
          final_state_resumed.params, expected_final_state.params
      )
      self.assertEqual(final_state_resumed.step, expected_final_state.step)

  def test_compress_and_restore_validation(self):
    """Tests noise_seed validation, unrestored fit error, and idempotence."""
    params = jnp.array([5.0, 5.0])
    dataset = np.array([[1.0, 2.0], [3.0, 4.0]])
    trainer = training.DPTrainer(
        config=_make_config(3),
        loss_fn=_quadratic_loss,
        optimizer=optax.sgd(1.0),
    )
    state0 = trainer.init(params)
    compressed = state0.compress()
    self.assertIsNone(compressed.noise_state)
    with self.assertRaisesRegex(ValueError, 'noise_seed'):
      compressed.restore(trainer)
    with self.assertRaisesRegex(ValueError, 'Compressed state'):
      trainer.fit(dataset, compressed, rng_or_seed=0)
    self.assertIs(state0.restore(trainer), state0)

  def test_map_dataset_pytree_structure(self):
    """Train loop works with a Grain MapDataset yielding PyTrees."""
    params = jnp.array([1.0, 2.0])
    data = [
        {'x': np.array([1.0, 2.0]), 'y': np.array([0.5])},
        {'x': np.array([3.0, 4.0]), 'y': np.array([1.5])},
        {'x': np.array([5.0, 6.0]), 'y': np.array([2.5])},
    ]
    dataset = grain.MapDataset.source(data)

    def pytree_loss(params, batch, prng):
      del prng
      loss = jnp.mean((params - batch['x']) ** 2) + jnp.mean(batch['y'])
      return loss, {'loss': loss}

    trainer = training.DPTrainer(
        config=_make_config(iterations=3),
        loss_fn=pytree_loss,
        optimizer=optax.sgd(0.01),
    )
    state = trainer.fit(dataset, trainer.init(params), rng_or_seed=0)
    self.assertEqual(int(state.step), 3)

  def test_params_change_after_training(self):
    """Parameters should change from initial values after training."""
    params = jnp.array([10.0, 10.0])
    dataset = np.array([[0.0, 0.0], [0.0, 0.0]])
    config = _make_config(iterations=5)
    optimizer = optax.sgd(0.1)

    trainer = training.DPTrainer(
        config=config,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )
    state = trainer.fit(dataset, trainer.init(params), rng_or_seed=42)

    self.assertFalse(jnp.allclose(state.params, params))

  def test_non_private_training_runs(self):
    """Verifies that DPTrainer runs successfully with NonPrivateConfig."""
    params = jnp.array([5.0, 5.0])
    dataset = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    config = execution_plan.NonPrivateConfig(iterations=3, batch_size=2)

    trainer = training.DPTrainer(
        config=config,
        loss_fn=_quadratic_loss,
        optimizer=optax.sgd(0.01),
    )
    state = trainer.fit(dataset, trainer.init(params), rng_or_seed=0)

    self.assertIsInstance(state, training.TrainingState)
    self.assertEqual(int(state.step), 3)

  def test_non_private_params_change_after_training(self):
    """Parameters should change from initial values after training."""
    params = jnp.array([10.0, 10.0])
    dataset = np.array([[0.0, 0.0], [0.0, 0.0]])
    config = execution_plan.NonPrivateConfig(iterations=5, batch_size=2)

    trainer = training.DPTrainer(
        config=config,
        loss_fn=_quadratic_loss,
        optimizer=optax.sgd(0.1),
    )
    state = trainer.fit(dataset, trainer.init(params), rng_or_seed=42)

    self.assertFalse(jnp.allclose(state.params, params))

  def test_callback_invoked(self):
    """Callback should be invoked once per training step."""
    params = jnp.array([1.0])
    dataset = np.array([[0.0], [1.0]])
    iterations = 3
    config = _make_config(iterations=iterations)
    optimizer = optax.sgd(0.01)
    callback_log = []

    def callback(step, state, aux):
      callback_log.append((int(step), float(state.params[0])))
      self.assertIsInstance(aux.grad_norms, jax.Array)
      self.assertIsInstance(aux.values, jax.Array)
      self.assertIsNotNone(aux.aux)

    trainer = training.DPTrainer(
        config=config,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )
    trainer.fit(
        dataset,
        trainer.init(params),
        callback=callback,
        rng_or_seed=0,
    )

    self.assertLen(callback_log, iterations)
    self.assertEqual([s for s, _ in callback_log], [1, 2, 3])

  def test_padding_multiple(self):
    """Training should work with padding_multiple set."""
    params = jnp.array([1.0])
    dataset = np.array([[0.0], [1.0], [2.0]])
    config = _make_config(iterations=2)
    optimizer = optax.sgd(0.01)

    trainer = training.DPTrainer(
        config=config,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
        compilation_strategy=training.PadToMultiple(multiple=4),
    )
    state = trainer.fit(dataset, trainer.init(params), rng_or_seed=0)

    self.assertEqual(int(state.step), 2)

  def test_zero_iterations_config_raises(self):
    """BandMFConfig requires iterations >= 1."""
    with self.assertRaises(Exception):
      _make_config(iterations=0)

  def test_single_iteration(self):
    """Training with a single iteration should work correctly."""
    params = jnp.array([5.0, 5.0])
    dataset = np.array([[1.0, 0.0]])
    config = _make_config(iterations=1)
    optimizer = optax.sgd(0.01)

    trainer = training.DPTrainer(
        config=config,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )
    state = trainer.fit(dataset, trainer.init(params), rng_or_seed=0)

    self.assertEqual(int(state.step), 1)

  def test_loss_fn_traced_once_with_equal_batches(self):
    """loss_fn should only be traced once with constant batches."""
    trace_count = [0]

    def counting_loss(params, batch, prng):
      del prng
      trace_count[0] += 1
      loss = jnp.mean((params - batch) ** 2)
      return loss, {}

    params = jnp.array([1.0])
    dataset = np.array([[0.0], [1.0]])
    config = _make_config(iterations=3)
    optimizer = optax.sgd(0.01)

    jax.clear_caches()

    trainer = training.DPTrainer(
        config=config,
        loss_fn=counting_loss,
        optimizer=optimizer,
    )
    state = trainer.fit(dataset, trainer.init(params), rng_or_seed=0)

    self.assertEqual(int(state.step), 3)
    self.assertLess(trace_count[0], 3 * 2)

  def test_train_step_callable_directly(self):
    """train_step should be directly callable outside of fit()."""
    params = jnp.array([5.0, 5.0])
    config = _make_config(iterations=2, noise_multiplier=0.0)
    optimizer = optax.sgd(0.01)

    trainer = training.DPTrainer(
        config=config,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )

    state = training.TrainingState(
        step=0,
        params=jnp.copy(params),
        opt_state=optimizer.init(params),
        noise_state=trainer.plan.noise_addition_transform.init(params),
    )

    batch = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    is_padding = jnp.array([False, False])
    prng_key = jax.random.key(0)

    new_state, _ = trainer.train_step(state, batch, is_padding, prng_key)

    self.assertEqual(int(new_state.step), 1)
    self.assertFalse(jnp.allclose(new_state.params, params))

  def test_train_step_jit_compilable(self):
    """train_step should be JIT-compilable."""
    params = jnp.array([5.0])
    config = _make_config(iterations=1, noise_multiplier=0.0)
    optimizer = optax.sgd(0.01)

    trainer = training.DPTrainer(
        config=config,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )

    state = training.TrainingState(
        step=0,
        params=params,
        opt_state=optimizer.init(params),
        noise_state=trainer.plan.noise_addition_transform.init(params),
    )

    batch = jnp.array([[1.0], [0.0]])
    is_padding = jnp.array([False, False])
    prng_key = jax.random.key(0)

    # train_step is already @jax.jit decorated; call it directly.
    new_state, _ = trainer.train_step(state, batch, is_padding, prng_key)

    self.assertEqual(int(new_state.step), 1)

  def test_fit_with_param_specs(self):
    """fit() works on an Auto mesh with ZERO strategy and param_specs."""
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mesh = jax.sharding.Mesh(devices, ('replica', 'model'))
    param_spec = jax.sharding.PartitionSpec('model')
    param_sharding = jax.sharding.NamedSharding(mesh, param_spec)

    with jax.set_mesh(mesh):
      base_w = jax.device_put(jnp.ones((4,), dtype=jnp.float32), param_sharding)
      params = jax.device_put(
          jnp.full((4,), 5.0, dtype=jnp.float32), param_sharding
      )
      dataset = np.zeros((8, 4), dtype=np.float32)

      def loss_with_closed_over_base(p, batch, prng):
        del prng
        loss = jnp.mean((p + base_w - batch) ** 2)
        return loss, {'loss': loss}

      config = execution_plan.BandMFConfig.default(
          num_bands=2,
          iterations=2,
          noise_multiplier=1.0,
          expected_participations=1,
      )
      flags = execution_plan.PerformanceFlags(
          intermediate_strategy=noise_addition.SupportedStrategies.ZERO,
          microbatch_size=2,
          param_specs=param_spec,
      )
      trainer = training.DPTrainer(
          config=config,
          performance_flags=flags,
          loss_fn=loss_with_closed_over_base,
          optimizer=optax.sgd(0.01),
          compilation_strategy=training.PadToMultiple(multiple=4),
      )
      state = trainer.fit(dataset, trainer.init(params), rng_or_seed=0)

    self.assertEqual(int(state.step), 2)
    self.assertEqual(state.params.sharding.mesh, mesh)
    self.assertTrue(jnp.all(jnp.isfinite(state.params)))

  @parameterized.named_parameters(
      ('array', lambda x: x),
      ('grain', _as_map_dataset),
  )
  def test_fit_with_batch_axis_name(self, to_dataset):
    """fit() shards batches along batch_axis_name on an active mesh."""
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mesh = jax.sharding.Mesh(devices, ('replica', 'model'))
    param_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('model')
    )

    with jax.set_mesh(mesh):
      base_w = jax.device_put(jnp.ones((4,), dtype=jnp.float32), param_sharding)
      params = jax.device_put(
          jnp.full((4,), 5.0, dtype=jnp.float32), param_sharding
      )
      dataset = to_dataset(np.zeros((8, 4), dtype=np.float32))

      def loss_with_closed_over_base(p, batch, prng):
        del prng
        loss = jnp.mean((p + base_w - batch) ** 2)
        return loss, {'loss': loss}

      config = execution_plan.BandMFConfig.default(
          num_bands=2,
          iterations=2,
          noise_multiplier=1.0,
          expected_participations=1,
      )
      flags = execution_plan.PerformanceFlags(
          batch_axis_name='replica',
          microbatch_size=2,
      )
      trainer = training.DPTrainer(
          config=config,
          performance_flags=flags,
          loss_fn=loss_with_closed_over_base,
          optimizer=optax.sgd(0.01),
          compilation_strategy=training.PadToMultiple(multiple=4),
      )
      state = trainer.fit(dataset, trainer.init(params), rng_or_seed=0)

    self.assertEqual(int(state.step), 2)
    self.assertEqual(state.params.sharding.mesh, mesh)
    self.assertTrue(jnp.all(jnp.isfinite(state.params)))


class DPTrainerEdgeCasesTest(parameterized.TestCase):
  """Edge case tests for the DPTrainer class."""

  def test_epsilon_zero_high_noise(self):
    """Near-zero epsilon (very high noise) should run without error."""
    params = jnp.array([1.0, 2.0])
    dataset = np.array([[0.0, 0.0], [1.0, 1.0]])
    config = _make_config(iterations=2, noise_multiplier=1e6)
    optimizer = optax.sgd(0.01)

    trainer = training.DPTrainer(
        config=config,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )
    state = trainer.fit(dataset, trainer.init(params), rng_or_seed=0)

    self.assertEqual(int(state.step), 2)
    self.assertTrue(jnp.all(jnp.isfinite(state.params)))

  def test_epsilon_inf_no_noise(self):
    """noise_multiplier=0 should behave like non-private SGD."""
    params = jnp.array([5.0])
    dataset = np.array([[0.0], [0.0]])
    config = _make_config(iterations=3, noise_multiplier=0.0)
    optimizer = optax.sgd(0.1)

    trainer = training.DPTrainer(
        config=config,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )
    state = trainer.fit(dataset, trainer.init(params), rng_or_seed=0)

    self.assertEqual(int(state.step), 3)
    self.assertLess(
        float(jnp.abs(state.params[0])),
        float(jnp.abs(params[0])),
    )

  def test_single_example_dataset(self):
    """Training on a single example should work correctly."""
    params = jnp.array([5.0])
    dataset = np.array([[0.0]])
    config = _make_config(iterations=2, noise_multiplier=0.0)
    optimizer = optax.sgd(0.1)

    trainer = training.DPTrainer(
        config=config,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )
    state = trainer.fit(dataset, trainer.init(params), rng_or_seed=0)

    self.assertEqual(int(state.step), 2)
    self.assertTrue(jnp.all(jnp.isfinite(state.params)))

  def test_dict_dataset(self):
    """Training should work with dict-structured datasets."""
    params = jnp.array([1.0])
    dataset = {'x': np.array([[0.0], [1.0], [2.0]])}

    def dict_loss(params, batch, prng):
      del prng
      loss = jnp.mean((params - batch['x']) ** 2)
      return loss, {}

    config = _make_config(iterations=2)
    optimizer = optax.sgd(0.01)

    trainer = training.DPTrainer(
        config=config,
        loss_fn=dict_loss,
        optimizer=optimizer,
    )
    state = trainer.fit(dataset, trainer.init(params), rng_or_seed=0)

    self.assertEqual(int(state.step), 2)

  def test_bfloat16_params_preserved(self):
    """bfloat16 params with float32 aggregation should return bfloat16."""
    params = jnp.array([5.0, 5.0], dtype=jnp.bfloat16)
    dataset = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    flags = execution_plan.PerformanceFlags(dtype=np.float32)
    config = _make_config(iterations=2, noise_multiplier=0.0)
    optimizer = optax.sgd(0.01)

    trainer = training.DPTrainer(
        config=config,
        performance_flags=flags,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )
    state = trainer.fit(dataset, trainer.init(params), rng_or_seed=0)

    self.assertEqual(state.params.dtype, jnp.bfloat16)


class DPTrainerInitTest(parameterized.TestCase):
  """Tests for DPTrainer.init."""

  def test_init_returns_training_state(self):
    """init() should return a TrainingState at step 0."""
    params = jnp.array([1.0, 2.0])
    config = _make_config(iterations=3)
    optimizer = optax.sgd(0.01)

    trainer = training.DPTrainer(
        config=config,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )
    state = trainer.init(params)

    self.assertIsInstance(state, training.TrainingState)
    self.assertEqual(int(state.step), 0)
    np.testing.assert_array_equal(state.params, params)

  def test_noise_seed_passed_to_plan(self):
    """DPTrainer(noise_seed=...) configures deterministic noise addition."""
    params = jnp.array([1.0, 2.0])
    dataset = np.array([[0.0, 0.0], [1.0, 1.0]])
    config = _make_config(iterations=2, noise_multiplier=1.0)
    trainer1 = training.DPTrainer(
        config=config,
        loss_fn=_quadratic_loss,
        optimizer=optax.sgd(0.1),
        noise_seed=7,
    )
    trainer2 = training.DPTrainer(
        config=config,
        loss_fn=_quadratic_loss,
        optimizer=optax.sgd(0.1),
        noise_seed=7,
    )
    state1 = trainer1.fit(dataset, trainer1.init(params), rng_or_seed=0)
    state2 = trainer2.fit(dataset, trainer2.init(params), rng_or_seed=0)
    np.testing.assert_allclose(state1.params, state2.params)


class DPTrainerPrecompileTest(parameterized.TestCase):
  """Tests for DPTrainer.precompile."""

  def test_precompile_returns_futures(self):
    """precompile() should return a dict of batch_size -> Future."""
    params = jnp.array([1.0, 2.0])
    dataset = np.array([[0.0, 0.0]] * 10)  # 10 examples.
    config = _make_config(iterations=5)
    optimizer = optax.sgd(0.01)

    trainer = training.DPTrainer(
        config=config,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )
    _, futures = trainer._precompile(
        dataset, trainer.init(params), rng_or_seed=42
    )

    self.assertIsInstance(futures, dict)
    self.assertNotEmpty(futures)
    for size, future in futures.items():
      self.assertIsInstance(size, int)
      self.assertGreater(size, 0)
      # Compilation should complete without error.
      future.result()

  def test_precompile_sizes_are_padded(self):
    """All precompiled sizes should be multiples of padding_multiple."""
    params = jnp.array([1.0])
    dataset = np.array([[0.0]] * 20)  # 20 examples.
    config = _make_config(iterations=10)
    optimizer = optax.sgd(0.01)
    padding_multiple = 8

    trainer = training.DPTrainer(
        config=config,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
        compilation_strategy=training.PadToMultiple(multiple=padding_multiple),
    )
    _, futures = trainer._precompile(
        dataset, trainer.init(params), rng_or_seed=0
    )

    for size in futures:
      self.assertEqual(size % padding_multiple, 0)

    # Wait for all compilations.
    for future in futures.values():
      future.result()

  def test_precompile_rng_not_consumed(self):
    """precompile should deep-copy the RNG, not consume the caller's."""
    params = jnp.array([1.0])
    dataset = np.array([[0.0]] * 5)  # 5 examples.
    config = _make_config(iterations=3)
    optimizer = optax.sgd(0.01)

    trainer = training.DPTrainer(
        config=config,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )

    rng = np.random.default_rng(42)
    state_before = rng.__getstate__()
    _, futures = trainer._precompile(
        dataset, trainer.init(params), rng_or_seed=rng
    )
    state_after = rng.__getstate__()

    # RNG should not have been consumed.
    np.testing.assert_equal(state_before, state_after)

    for future in futures.values():
      future.result()

  def test_precompile_with_shape_dtype_struct(self):
    """precompile() should work with abstract ShapeDtypeStruct inputs."""
    params = jax.ShapeDtypeStruct((3,), jnp.float32)
    dataset = jax.ShapeDtypeStruct((5, 3), jnp.float32)
    config = _make_config(iterations=3)
    optimizer = optax.sgd(0.01)

    trainer = training.DPTrainer(
        config=config,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )
    _, futures = trainer._precompile(
        dataset, jax.eval_shape(trainer.init, params), rng_or_seed=0
    )

    self.assertNotEmpty(futures)
    for future in futures.values():
      future.result()

  def test_fit_precompile_aot_compiles_all_sizes(self):
    """precompile=True should AOT-compile once per unique batch size."""
    trace_count = [0]

    def loss_fn(params, batch, _):
      trace_count[0] += 1
      return jnp.mean((params - batch) ** 2), {}

    params = jnp.array([1.0])
    dataset = np.array([[i] for i in range(50)])

    # Use a stochastic batch strategy so precompile sees several batch sizes.
    # ``_FixedPlanConfig`` lets us inject a custom plan through the config API.
    plan = dataclasses.replace(
        _make_config(iterations=5).make(),
        batch_selection_strategy=batch_selection.CyclicPoissonSampling(0.5, 5),
    )
    trainer = training.DPTrainer(
        config=_FixedPlanConfig(plan),
        loss_fn=loss_fn,
        optimizer=optax.sgd(1),
        compilation_strategy=training.PadToMultiple(multiple=1),
    )

    with self.assertLogs(level='INFO') as logs:
      trainer.fit(dataset, trainer.init(params), rng_or_seed=0, precompile=True)
      for log in logs.output:
        self.assertIn('AOT-compiling train_step for batch size', log)
        self.assertNotIn('JIT-compiling train_step for batch size', log)
      self.assertEqual(trace_count[0], 5)
      self.assertLen(logs.output, 5)

  def test_precompile_dedupes_shared_padded_sizes(self):
    """Steps sharing a padded size are lowered and compiled only once."""
    trace_count = [0]

    def loss_fn(params, batch, _):
      trace_count[0] += 1
      return jnp.mean((params - batch) ** 2), {}

    params = jnp.array([1.0])
    dataset = np.array([[i] for i in range(50)])

    # A padding multiple larger than any possible batch collapses every step to
    # the same padded size, so precompile must lower/compile exactly once even
    # though the run takes several (differently sized) steps.
    plan = dataclasses.replace(
        _make_config(iterations=5).make(),
        batch_selection_strategy=batch_selection.CyclicPoissonSampling(0.5, 5),
    )
    trainer = training.DPTrainer(
        config=_FixedPlanConfig(plan),
        loss_fn=loss_fn,
        optimizer=optax.sgd(1),
        compilation_strategy=training.PadToMultiple(multiple=64),
    )

    with self.assertLogs(level='INFO') as logs:
      trainer.fit(dataset, trainer.init(params), rng_or_seed=0, precompile=True)

    aot = [l for l in logs.output if 'AOT-compiling train_step' in l]
    self.assertLen(aot, 1)
    self.assertEqual(trace_count[0], 1)

  @parameterized.parameters(jnp.bfloat16, jnp.float16, jnp.float32)
  def test_fit_precompile_low_precision_params(self, param_dtype):
    """precompile=True works when optax promotes low-precision moments."""
    params = jnp.ones((3,), dtype=param_dtype)
    dataset = np.zeros((10, 3), dtype=np.float32)
    trainer = training.DPTrainer(
        config=_make_config(iterations=5),
        loss_fn=_quadratic_loss,
        optimizer=optax.adamw(1e-3),
    )

    with self.assertLogs(level='INFO') as logs:
      state = trainer.fit(
          dataset, trainer.init(params), rng_or_seed=0, precompile=True
      )

    self.assertEqual(int(state.step), 5)
    self.assertEqual(state.params.dtype, param_dtype)
    # AOT precompilation should be effective (no JIT cache misses).
    for log in logs.output:
      self.assertNotIn('Cache Miss', log)

  def test_default_pad_to_max_with_microbatch_size(self):
    """Default PadToMultiple() pads to max batch size rounded to microbatch."""
    params = jnp.ones((3,), dtype=jnp.float32)
    dataset = np.zeros((20, 3), dtype=np.float32)
    trainer = training.DPTrainer(
        config=_make_config(iterations=3),
        performance_flags=execution_plan.PerformanceFlags(microbatch_size=6),
        loss_fn=_quadratic_loss,
        optimizer=optax.sgd(0.01),
    )

    resolved, futures = trainer._precompile(
        dataset, trainer.init(params), rng_or_seed=0
    )
    self.assertEqual(resolved.compilation_strategy.multiple % 6, 0)
    self.assertLen(futures, 1)

    with self.assertLogs(level='INFO') as logs:
      state = trainer.fit(
          dataset, trainer.init(params), rng_or_seed=0, precompile=True
      )

    self.assertEqual(int(state.step), 3)
    for log in logs.output:
      self.assertNotIn('Cache Miss', log)

  def test_incompatible_multiple_and_microbatch_size_raises(self):
    """PadToMultiple(multiple=4) with microbatch_size=6 raises ValueError."""
    params = jnp.ones((3,), dtype=jnp.float32)
    dataset = np.zeros((20, 3), dtype=np.float32)
    trainer = training.DPTrainer(
        config=_make_config(iterations=3),
        performance_flags=execution_plan.PerformanceFlags(microbatch_size=6),
        loss_fn=_quadratic_loss,
        optimizer=optax.sgd(0.01),
        compilation_strategy=training.PadToMultiple(multiple=4),
    )
    with self.assertRaisesRegex(ValueError, 'must be divisible by'):
      trainer.fit(dataset, trainer.init(params), rng_or_seed=0)


class DPTrainerAutotuneTest(parameterized.TestCase):
  """CPU smoke test for microbatch_size autotuning."""

  def test_autotune_fit_runs_on_cpu(self):
    """fit() with autotuning selects a microbatch size and completes."""
    # On CPU ``memory_stats`` is unavailable, so autotuning falls back to
    # compile-success and picks the largest candidate that compiles.
    params = jnp.array([1.0, 2.0])
    dataset = np.array([[0.0, 0.0]] * 10)  # 10 examples.
    trainer = training.DPTrainer(
        config=_make_config(iterations=3),
        loss_fn=_quadratic_loss,
        optimizer=optax.sgd(0.01),
        compilation_strategy=training.AutotuneMicrobatch(),
    )

    with self.assertLogs(level='INFO') as logs:
      state = trainer.fit(dataset, trainer.init(params), rng_or_seed=0)

    self.assertEqual(int(state.step), 3)
    self.assertTrue(any('fits; pad=' in log for log in logs.output))
    # Autotuning compiles the one fixed padded size ahead of time; the training
    # loop must reuse that step for every batch (no recompilation).
    for log in logs.output:
      self.assertNotIn('Cache Miss', log)


class ExtrapolateSeedTest(parameterized.TestCase):
  """Unit tests for the affine microbatch seed used by autotuning."""

  @parameterized.named_parameters(
      ('affine_exact', 1000.0, 2000.0, 8000.0, 8),
      ('zero_slope_all_fit', 0.0, 0.0, 5.0, 1024),
      ('seed_one_when_min_over_budget', 10.0, 20.0, 5.0, 1),
      ('decreasing_all_fit', 200.0, 100.0, 500.0, 1024),
  )
  def test_extrapolate_seed_cases(self, peak1, peak2, budget, expected):
    powers = [2**i for i in range(11)]
    seed = _compilation._extrapolate_seed(peak1, peak2, budget, powers)
    self.assertEqual(seed, expected)

  def test_extrapolate_seed_matches_brute_force(self):
    """Seed equals the brute-force largest power of two under the affine fit."""
    rng = np.random.default_rng(0)
    powers = [2**i for i in range(14)]
    for _ in range(1000):
      c0, c1 = rng.uniform(0, 1e6), rng.uniform(1, 1e4)
      budget = rng.uniform(0, c0 + c1 * powers[-1])
      expected = powers[0]
      for b in powers:
        if c0 + c1 * b <= budget:
          expected = b
      seed = _compilation._extrapolate_seed(
          c0 + c1, c0 + 2 * c1, budget, powers
      )
      self.assertEqual(seed, expected)


class _FakeDevice:
  """Stand-in JAX device with configurable memory_stats and topology attr."""

  def __init__(self, *, stats=None, stats_error=None, total=None):
    self._stats = stats
    self._stats_error = stats_error
    if total is not None:
      self.device_memory_bytes_limit = total

  def memory_stats(self):
    if self._stats_error is not None:
      raise self._stats_error
    return self._stats


class DeviceHbmLimitTest(parameterized.TestCase):
  """Unit tests for sourcing the per-chip HBM budget."""

  def _patch_devices(self, devices):
    self.enterContext(
        mock.patch.object(jax, 'local_devices', return_value=devices)
    )

  def test_prefers_bytes_limit_over_attribute(self):
    self._patch_devices([_FakeDevice(stats={'bytes_limit': 100}, total=999)])
    self.assertEqual(_compilation._device_hbm_limit(), 100)

  def test_falls_back_to_attribute_when_memory_stats_raises(self):
    self._patch_devices(
        [_FakeDevice(stats_error=RuntimeError('unsupported'), total=42)]
    )
    self.assertEqual(_compilation._device_hbm_limit(), 42)

  def test_falls_back_to_attribute_when_bytes_limit_missing(self):
    self._patch_devices([_FakeDevice(stats={}, total=42)])
    self.assertEqual(_compilation._device_hbm_limit(), 42)

  def test_none_when_neither_available(self):
    self._patch_devices([_FakeDevice(stats_error=RuntimeError('x'))])
    self.assertIsNone(_compilation._device_hbm_limit())

  def test_none_when_no_local_devices(self):
    self._patch_devices([])
    self.assertIsNone(_compilation._device_hbm_limit())


class DPTrainerCanaryAuditTest(parameterized.TestCase):
  """Tests for num_examples validation and empirical privacy auditing."""

  @parameterized.named_parameters(*_DATASET_TYPES)
  def test_num_examples_validation(self, to_dataset):
    params = jnp.array([1.0, 2.0])
    config = execution_plan.BandMFConfig.default(
        num_bands=1,
        iterations=2,
        noise_multiplier=1.0,
        expected_participations=2,
        num_examples=10,
    )
    trainer = training.DPTrainer(
        config=config,
        loss_fn=_quadratic_loss,
        optimizer=optax.sgd(0.1),
    )

    short_ds = to_dataset(np.zeros((8, 2), dtype=np.float32))
    with self.assertRaisesRegex(ValueError, 'num_examples=8.*expected 10'):
      trainer.fit(short_ds, trainer.init(params), rng_or_seed=0)

    exact_ds = to_dataset(np.zeros((10, 2), dtype=np.float32))
    state = trainer.fit(exact_ds, trainer.init(params), rng_or_seed=0)
    self.assertEqual(int(state.step), 2)

    long_ds = to_dataset(np.zeros((12, 2), dtype=np.float32))
    with self.assertRaisesRegex(ValueError, 'num_examples=12.*expected 10'):
      trainer.fit(long_ds, trainer.init(params), rng_or_seed=0)

  @parameterized.parameters(True, False)
  def test_score_canaries(self, pass_untrained):
    init_params = jnp.array([0.0, 0.0])
    trained = jnp.array([2.0, 2.0])
    canaries = np.arange(10, dtype=np.float32).reshape(5, 2)
    untrained = init_params if pass_untrained else None
    kw = dict(microbatch_size=2, prng_key=jax.random.key(0))

    scores = training._score_canaries(
        _quadratic_loss, trained, canaries, untrained, **kw
    )
    l0 = np.mean((init_params - canaries) ** 2, axis=1) if pass_untrained else 0
    expected = l0 - np.mean((trained - canaries) ** 2, axis=1)
    np.testing.assert_allclose(scores, expected, rtol=1e-5, atol=1e-5)

  @parameterized.product(
      pass_untrained=[True, False],
      identical_canaries=[False, True],
  )
  def test_audit(self, pass_untrained, identical_canaries):
    init_params = jnp.array([0.0, 0.0])
    held_in = np.full((5, 2), 4.0, dtype=np.float32)
    held_out = held_in if identical_canaries else -held_in
    dataset = np.concatenate([np.zeros((4, 2), dtype=np.float32), held_in])

    trainer = training.DPTrainer(
        config=_make_config(iterations=5, noise_multiplier=0.0),
        performance_flags=execution_plan.PerformanceFlags(microbatch_size=2),
        loss_fn=_quadratic_loss,
        optimizer=optax.sgd(0.1),
    )

    state = trainer.fit(dataset, trainer.init(init_params), rng_or_seed=0)
    untrained = init_params if pass_untrained else None
    auditor = training.audit(
        _quadratic_loss,
        state.params,
        held_in,
        held_out,
        untrained,
        microbatch_size=2,
    )
    expected_auroc = 0.5 if identical_canaries else 1.0
    self.assertAlmostEqual(float(auditor.attack_auroc()), expected_auroc)


if __name__ == '__main__':
  absltest.main()

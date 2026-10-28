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
import math

from absl.testing import absltest
from absl.testing import parameterized
import dp_accounting
from jax_privacy import batch_selection
from jax_privacy import execution_plan
import numpy as np
import optax

BandMFConfig = execution_plan.BandMFConfig


# pylint: disable=g-bad-todo
# TODO: Improve test coverage, including correctness of the
# privacy guarantees.
class ExecutionPlanTest(parameterized.TestCase):

  @parameterized.parameters(
      {"strategy": np.array([])},
      {"truncated_batch_size": 5, "num_examples": None},
      {"normalize_by": 0},
      {"normalize_by": float("nan")},
      {"normalize_by": float("inf")},
  )
  def test_bandmf_validation(self, **kwargs):
    default_kwargs = {
        "strategy": np.linspace(1, 0, 10),
        "iterations": 20,
        "expected_participations": 2,
        "noise_multiplier": 1.0,
    }
    default_kwargs.update(kwargs)
    with self.assertRaises(ValueError):
      BandMFConfig(**default_kwargs)

  def test_bandmf_rejects_nan_normalize_by(self):
    with self.assertRaisesRegex(
        ValueError, r"Expected normalize_by=nan > 0"
    ):
      BandMFConfig(
          strategy=np.linspace(1, 0, 10),
          iterations=20,
          expected_participations=2,
          noise_multiplier=1.0,
          normalize_by=float("nan"),
      )

  @parameterized.parameters(
      {
          "noise_multiplier": 1.0,
      },
      {
          "noise_multiplier": 1.0,
          "truncated_batch_size": 5,
          "num_examples": 10,
      },
  )
  def test_bandmf_execution_plan_creation(self, **privacy_kwargs):
    iterations = 20
    config = BandMFConfig.default(
        num_bands=10,
        iterations=iterations,
        expected_participations=iterations / 10,
        **privacy_kwargs,
    )

    plan = config.make()

    self.assertIsInstance(plan, execution_plan.DPExecutionPlan)
    self.assertIsInstance(
        plan.batch_selection_strategy, batch_selection.CyclicPoissonSampling
    )
    self.assertEqual(plan.batch_selection_strategy.sampling_prob, 1.0)
    self.assertIsInstance(
        plan.noise_addition_transform,
        optax.GradientTransformation,
    )
    self.assertLen(
        list(plan.batch_selection_strategy.batch_iterator(100)), iterations
    )

    self.assertIsInstance(plan.dp_event, dp_accounting.DpEvent)
    batch_gen = plan.batch_selection_strategy.batch_iterator(100, rng=0)
    self.assertIsInstance(next(batch_gen), np.ndarray)

  def test_bandmf_calibrate(self):
    config = BandMFConfig.default(
        num_bands=10,
        iterations=20,
        expected_participations=2,
    ).calibrate(epsilon=1.0, delta=1e-06)

    self.assertIsNotNone(config.noise_multiplier)
    self.assertGreater(config.noise_multiplier, 0)
    plan = config.make()
    self.assertIsInstance(plan, execution_plan.DPExecutionPlan)
    self.assertIsInstance(plan.dp_event, dp_accounting.DpEvent)

  def test_uncalibrated_make_raises_error(self):
    config = BandMFConfig.default(
        num_bands=10,
        iterations=20,
        expected_participations=2,
    )
    with self.assertRaises(ValueError):
      config.make()

  def test_make_with_default_performance_flags(self):
    config = BandMFConfig.default(
        num_bands=10,
        iterations=20,
        expected_participations=2,
        noise_multiplier=1.0,
    )
    plan = config.make()
    self.assertIsInstance(plan, execution_plan.DPExecutionPlan)

  def test_make_with_custom_performance_flags(self):
    config = BandMFConfig.default(
        num_bands=10,
        iterations=20,
        expected_participations=2,
        noise_multiplier=1.0,
    )
    flags = execution_plan.PerformanceFlags(
        dtype=np.float64,
        microbatch_size=4,
    )
    plan = config.make(flags, noise_seed=42)
    self.assertIsInstance(plan, execution_plan.DPExecutionPlan)

  def test_make_noise_seed_takes_precedence(self):
    config = BandMFConfig.default(
        num_bands=2,
        iterations=4,
        expected_participations=1,
        noise_multiplier=1.0,
    )
    grad = np.zeros(4, dtype=np.float32)

    def first_step_noise(plan):
      state = plan.noise_addition_transform.init(grad)
      noisy, _ = plan.noise_addition_transform.update(grad, state)
      return noisy

    with self.assertLogs(level="WARNING"):
      flags = execution_plan.PerformanceFlags(noise_seed=7)
    np.testing.assert_allclose(
        first_step_noise(config.make(flags)),
        first_step_noise(config.make(noise_seed=7)),
    )
    np.testing.assert_allclose(
        first_step_noise(config.make(flags, noise_seed=42)),
        first_step_noise(config.make(noise_seed=42)),
    )
    self.assertFalse(
        np.allclose(
            first_step_noise(config.make(flags, noise_seed=42)),
            first_step_noise(config.make(noise_seed=7)),
        )
    )

  def test_rmse_invariant_to_strategy_scale(self):
    # Scaling the strategy scales the noise stddev up and the noising matrix
    # down, leaving the mechanism (and hence its rmse) unchanged.
    config = BandMFConfig(
        iterations=8,
        expected_participations=2,
        strategy=np.array([1.0, 0.5, 0.2]),
        noise_multiplier=1.0,
        column_normalize=False,
    )
    scaled = dataclasses.replace(
        config, strategy=2 * np.asarray(config.strategy)
    )
    self.assertAlmostEqual(scaled._max_column_norm, 2 * config._max_column_norm)
    self.assertAlmostEqual(config.rmse, scaled.rmse, places=6)

  def test_rmse_raises_on_column_normalize(self):
    config = BandMFConfig(
        iterations=8,
        expected_participations=2,
        strategy=np.array([1.0, 0.5, 0.2]),
        noise_multiplier=1.0,
        column_normalize=True,
    )
    self.assertEqual(config._max_column_norm, 1.0)
    with self.assertRaises(NotImplementedError):
      _ = config.rmse

  def test_rmse_requires_calibration(self):
    config = BandMFConfig.default(
        num_bands=1,
        iterations=10,
        expected_participations=10,
    )
    with self.assertRaises(ValueError):
      _ = config.rmse

  def test_rmse_decreases_with_participations(self):
    config1 = BandMFConfig.default(
        num_bands=2,
        iterations=16,
        expected_participations=4,
    ).calibrate(epsilon=1.0, delta=1e-06)
    config2 = BandMFConfig.default(
        num_bands=2,
        iterations=16,
        expected_participations=2,
    ).calibrate(epsilon=1.0, delta=1e-06)
    self.assertLess(config1.rmse, config2.rmse)

  def test_non_private_config(self):
    """Tests that NonPrivateConfig creates a valid non-private plan."""
    iterations = 20
    batch_size = 5
    config = execution_plan.NonPrivateConfig(
        iterations=iterations,
        batch_size=batch_size,
    )
    plan = config.make()

    self.assertIsInstance(plan, execution_plan.DPExecutionPlan)
    self.assertIsInstance(
        plan.batch_selection_strategy, batch_selection.FixedBatchSampling
    )
    self.assertEqual(plan.batch_selection_strategy.batch_size, batch_size)
    self.assertEqual(plan.batch_selection_strategy.iterations, iterations)
    self.assertIsInstance(
        plan.noise_addition_transform, optax.GradientTransformation
    )
    self.assertIsInstance(plan.dp_event, dp_accounting.DpEvent)

    # Verify that the privatizer acts as identity (no-op)
    dummy_grads = {"w": np.ones((2, 2))}
    opt_state = plan.noise_addition_transform.init(dummy_grads)
    updates, _ = plan.noise_addition_transform.update(dummy_grads, opt_state)
    np.testing.assert_equal(updates, dummy_grads)

  @parameterized.parameters(
      # expected_participations must be an integer.
      {"expected_participations": 2.5},
      # k > iterations // num_bands.
      {"expected_participations": 11},
      # truncation is Poisson-only.
      {"truncated_batch_size": 5, "num_examples": 100},
      # num_examples (zero-out adjacency) is Poisson-only.
      {"num_examples": 100},
  )
  def test_random_allocation_validation(self, **kwargs):
    default_kwargs = {
        "strategy": np.linspace(1, 0, 2),
        "iterations": 20,
        "expected_participations": 2,
        "noise_multiplier": 1.0,
        "sub_strategy": execution_plan.CyclicInnerStrategy.RANDOM_ALLOCATION,
    }
    default_kwargs.update(kwargs)
    with self.assertRaises(ValueError):
      BandMFConfig(**default_kwargs)

  @parameterized.parameters(
      {"num_bands": 1, "iterations": 20, "expected_participations": 4},
      {"num_bands": 4, "iterations": 20, "expected_participations": 5},
      # iterations is not a multiple of num_bands.
      {"num_bands": 3, "iterations": 20, "expected_participations": 7},
  )
  def test_random_allocation_execution_plan_creation(
      self, num_bands, iterations, expected_participations
  ):
    config = BandMFConfig.default(
        num_bands=num_bands,
        iterations=iterations,
        expected_participations=expected_participations,
        noise_multiplier=1.5,
        sub_strategy=execution_plan.CyclicInnerStrategy.RANDOM_ALLOCATION,
    )
    plan = config.make()

    self.assertIsInstance(plan, execution_plan.DPExecutionPlan)
    strategy = plan.batch_selection_strategy
    self.assertIsInstance(strategy, batch_selection.RandomAllocationSampling)
    self.assertEqual(strategy.total_participations, expected_participations)
    self.assertEqual(strategy.iterations, iterations)
    self.assertEqual(strategy.cycle_length, num_bands)
    self.assertLen(list(strategy.batch_iterator(100)), iterations)

    self.assertIsInstance(
        plan.dp_event, dp_accounting.dp_event.RandomAllocationDpEvent
    )
    self.assertEqual(plan.dp_event.num_selected, expected_participations)
    self.assertEqual(plan.dp_event.num_steps, math.ceil(iterations / num_bands))
    self.assertEqual(
        plan.neighboring_relation,
        dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE,
    )

  def test_random_allocation_uncalibrated_raises(self):
    config = BandMFConfig.default(
        num_bands=1,
        iterations=10,
        expected_participations=2,
        sub_strategy=execution_plan.CyclicInnerStrategy.RANDOM_ALLOCATION,
    )
    with self.assertRaises(ValueError):
      config.make()

  def test_random_allocation_calibrate(self):
    accountant_fn = lambda rel: dp_accounting.pld.PLDAccountant(
        neighboring_relation=rel, value_discretization_interval=1e-2
    )
    config = BandMFConfig.default(
        num_bands=1,
        iterations=5,
        expected_participations=1,
        sub_strategy=execution_plan.CyclicInnerStrategy.RANDOM_ALLOCATION,
    ).calibrate(
        epsilon=1.0,
        delta=1e-03,
        tol=1e-2,
        accountant_fn=accountant_fn,
    )

    self.assertIsNotNone(config.noise_multiplier)
    self.assertGreater(config.noise_multiplier, 0)
    plan = config.make()
    self.assertIsInstance(plan, execution_plan.DPExecutionPlan)
    self.assertIsInstance(
        plan.dp_event, dp_accounting.dp_event.RandomAllocationDpEvent
    )

  def test_poisson_is_the_default_sub_strategy(self):
    config = BandMFConfig.default(
        num_bands=2,
        iterations=20,
        expected_participations=2,
        noise_multiplier=1.0,
    )
    self.assertEqual(
        config.sub_strategy, execution_plan.CyclicInnerStrategy.POISSON
    )
    self.assertIsInstance(
        config.make().batch_selection_strategy,
        batch_selection.CyclicPoissonSampling,
    )

  def test_spmd_axis_name_alias(self):
    flags = execution_plan.PerformanceFlags(spmd_axis_name="data")
    self.assertEqual(flags.batch_axis_name, "data")
    with self.assertRaises(ValueError):
      execution_plan.PerformanceFlags(
          batch_axis_name="data", spmd_axis_name="data"
      )


if __name__ == "__main__":
  absltest.main()

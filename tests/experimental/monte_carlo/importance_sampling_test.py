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

import math

from absl.testing import absltest
from absl.testing import parameterized
from jax_privacy import batch_selection
from jax_privacy.experimental.monte_carlo import delta_calculation
from jax_privacy.experimental.monte_carlo import importance_sampling
from jax_privacy.experimental.monte_carlo import sample_generation
import numpy as np

_STRATEGY = batch_selection.BallsInBinsSampling(cycle_length=2, iterations=4)
_C_COL = np.array([1.0, 0.5])
_NOISE_MULTIPLIER = 1.0
# Modes are [1, 0.5, 1, 0.5] and [0, 1, 0.5, 1] (padded first mode only).
_KAPPA = 2.5


def _hockey_stick_estimands(epsilon, privacy_loss):
  return -np.expm1(np.minimum(epsilon - privacy_loss, 0.0))


class LogSupportBoundTest(parameterized.TestCase):

  def test_is_zero_for_unit_stretch(self):
    self.assertEqual(
        importance_sampling.log_support_bound(1.0, epsilon=2.0, kappa=3.0), 0.0
    )

  def test_matches_closed_form(self):
    # At alpha = 2, log B = ln(1 / 4) + kappa - epsilon.
    epsilon, kappa = 1.5, 0.7
    self.assertAlmostEqual(
        importance_sampling.log_support_bound(2.0, epsilon, kappa),
        -np.log(4.0) + kappa - epsilon,
    )

  def test_is_decreasing_in_epsilon(self):
    bounds = [
        importance_sampling.log_support_bound(1.5, epsilon, kappa=1.0)
        for epsilon in [0.0, 0.5, 1.0, 2.0]
    ]
    self.assertTrue(np.all(np.diff(bounds) < 0))

  @parameterized.named_parameters(
      ("alpha", dict(alpha=0.5, epsilon=1.0, kappa=1.0)),
      ("epsilon", dict(alpha=1.5, epsilon=-1.0, kappa=1.0)),
      ("kappa", dict(alpha=1.5, epsilon=1.0, kappa=-1.0)),
  )
  def test_rejects_invalid_arguments(self, kwargs):
    with self.assertRaises(ValueError):
      importance_sampling.log_support_bound(**kwargs)


class OptimalStretchTest(parameterized.TestCase):

  @parameterized.product(
      alpha=[1.05, 1.5, 3.0, 20.0], kappa_multiplier=[1.01, 2.0, 10.0]
  )
  def test_recovers_minimizer(self, alpha, kappa_multiplier):
    # alpha minimizes the log support bound exactly when the derivative is 0,
    # i.e. for epsilon = ln(1 - 1 / alpha) + kappa (2 alpha - 1) / 2, which is
    # non-negative iff kappa >= kappa_min. The root finder must recover alpha,
    # and independently of that formula, alpha must beat its neighbors.
    kappa_min = -2 * math.log1p(-1 / alpha) / (2 * alpha - 1)
    kappa = kappa_multiplier * kappa_min
    epsilon = math.log1p(-1 / alpha) + kappa * (2 * alpha - 1) / 2
    self.assertAlmostEqual(
        importance_sampling.optimal_stretch(epsilon, kappa), alpha, places=6
    )
    optimum = importance_sampling.log_support_bound(alpha, epsilon, kappa)
    for other in [max(1.0, alpha / 1.05), alpha * 1.05]:
      self.assertLess(
          optimum, importance_sampling.log_support_bound(other, epsilon, kappa)
      )

  def test_returns_one_for_huge_kappa(self):
    self.assertEqual(importance_sampling.optimal_stretch(1.0, 1e20), 1.0)

  @parameterized.named_parameters(
      ("negative_epsilon", -1.0, 1.0), ("zero_kappa", 1.0, 0.0)
  )
  def test_rejects_invalid_arguments(self, epsilon, kappa):
    with self.assertRaises(ValueError):
      importance_sampling.optimal_stretch(epsilon, kappa)

  def test_bound_improves_with_smaller_kappa(self):
    epsilon = 1.0
    bounds = [
        importance_sampling.log_support_bound(
            importance_sampling.optimal_stretch(epsilon, kappa), epsilon, kappa
        )
        for kappa in [0.01, 0.1, 1.0, 10.0]
    ]
    self.assertTrue(np.all(np.diff(bounds) > 0))
    self.assertLess(bounds[0], np.log(0.05))


class KappaTest(absltest.TestCase):

  def test_matches_hand_computation(self):
    self.assertAlmostEqual(
        importance_sampling.compute_kappa(_STRATEGY, _NOISE_MULTIPLIER, _C_COL),
        _KAPPA,
    )
    self.assertAlmostEqual(
        importance_sampling.compute_kappa(
            _STRATEGY, 2.0 * _NOISE_MULTIPLIER, _C_COL
        ),
        _KAPPA / 4,
    )

  def test_rejects_other_strategies(self):
    strategy = batch_selection.CyclicPoissonSampling(
        sampling_prob=0.5, iterations=4, cycle_length=2
    )
    with self.assertRaisesRegex(ValueError, "only supported for balls-in-bins"):
      importance_sampling.compute_kappa(strategy, _NOISE_MULTIPLIER, _C_COL)


class GetImportanceSampledPrivacyLossTest(parameterized.TestCase):

  @parameterized.product(positive_sample=[True, False], epsilon=[0.5, 1.0, 3.0])
  def test_weighted_estimands_respect_support_bound(
      self, positive_sample, epsilon
  ):
    sample = importance_sampling.get_importance_sampled_privacy_loss(
        _STRATEGY,
        _NOISE_MULTIPLIER,
        _C_COL,
        epsilon,
        seed=0xBAD5EED,
        positive_sample=positive_sample,
        num_samples=100_000,
    )
    self.assertAlmostEqual(
        np.log(sample.support_bound),
        importance_sampling.log_support_bound(
            importance_sampling.optimal_stretch(epsilon, _KAPPA),
            epsilon,
            _KAPPA,
        ),
    )
    for eval_epsilon in [epsilon, epsilon + 1.0]:
      weighted = _hockey_stick_estimands(
          eval_epsilon, sample.privacy_loss
      ) * np.exp(sample.log_weights)
      self.assertGreaterEqual(weighted.min(), 0.0)
      self.assertLessEqual(weighted.max(), sample.support_bound * (1 + 1e-9))

  def test_rejects_alpha_worse_than_plain_monte_carlo(self):
    with self.assertRaisesRegex(ValueError, "worse than plain Monte Carlo"):
      importance_sampling.get_importance_sampled_privacy_loss(
          _STRATEGY, _NOISE_MULTIPLIER, _C_COL, epsilon=1.0, alpha=10.0
      )

  @parameterized.named_parameters(("forward", True), ("reverse", False))
  def test_unit_stretch_is_plain_monte_carlo(self, positive_sample):
    kwargs = dict(positive_sample=positive_sample, num_samples=100_000)
    sample = importance_sampling.get_importance_sampled_privacy_loss(
        _STRATEGY,
        _NOISE_MULTIPLIER,
        _C_COL,
        epsilon=1.0,
        alpha=1.0,
        seed=0xDECAF,
        **kwargs,
    )
    plain_privacy_loss, _ = sample_generation.get_privacy_loss_sample(
        _STRATEGY, _NOISE_MULTIPLIER, _C_COL, seed=0xDECAF, **kwargs
    )
    np.testing.assert_allclose(sample.log_weights, 0.0, atol=1e-12)
    self.assertEqual(sample.support_bound, 1.0)
    if positive_sample:
      np.testing.assert_allclose(sample.privacy_loss, plain_privacy_loss)
    else:
      # Sampling from Q as a mixture with zero modes consumes the rng
      # differently from sampling Q directly, so only the distributions agree.
      self.assertAlmostEqual(
          sample.privacy_loss.mean(), plain_privacy_loss.mean(), delta=0.1
      )
      self.assertAlmostEqual(
          sample.privacy_loss.var(), plain_privacy_loss.var(), delta=0.1
      )

  @parameterized.named_parameters(("forward", True), ("reverse", False))
  def test_is_unbiased_and_reduces_variance(self, positive_sample):
    epsilon, num_samples = 1.0, 200_000
    sample = importance_sampling.get_importance_sampled_privacy_loss(
        _STRATEGY,
        _NOISE_MULTIPLIER,
        _C_COL,
        epsilon,
        seed=0xC0FFEE,
        positive_sample=positive_sample,
        num_samples=num_samples,
    )
    plain_privacy_loss, _ = sample_generation.get_privacy_loss_sample(
        _STRATEGY,
        _NOISE_MULTIPLIER,
        _C_COL,
        seed=0xC0FFEE,
        positive_sample=positive_sample,
        num_samples=num_samples,
    )
    weighted = _hockey_stick_estimands(epsilon, sample.privacy_loss) * np.exp(
        sample.log_weights
    )
    plain = _hockey_stick_estimands(epsilon, plain_privacy_loss)

    self.assertGreater(plain.mean(), 1e-3)
    standard_error = np.sqrt((plain.var() + weighted.var()) / num_samples)
    self.assertLess(abs(plain.mean() - weighted.mean()), 6 * standard_error)
    self.assertLess(weighted.var(), plain.var())

    # The weighted estimate is what delta_from_epsilon_and_samples computes.
    self.assertAlmostEqual(
        delta_calculation.delta_from_epsilon_and_samples(
            epsilon, sample.privacy_loss, log_weights=sample.log_weights
        ),
        weighted.mean(),
    )


if __name__ == "__main__":
  absltest.main()

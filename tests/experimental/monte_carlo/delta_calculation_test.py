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
import dp_accounting
from jax_privacy.experimental.monte_carlo import delta_calculation
import numpy as np

_PrivacyLossSamples = delta_calculation.PrivacyLossSamples


def _samples(
    privacy_loss, counts=None
) -> list[delta_calculation.PrivacyLossSamples]:
  """Builds one PrivacyLossSamples per hyperparameter value."""
  if counts is None:
    counts = [None] * len(privacy_loss)
  return [_PrivacyLossSamples(p, c) for p, c in zip(privacy_loss, counts)]


def _slightly_fewer(num_samples):
  """One sample fewer, or 0.001% fewer for large counts."""
  # minimum_samples_to_calibrate is minimal up to a margin of that size.
  return np.minimum(num_samples - 1, np.floor(num_samples * (1 - 1e-5)))


def _point_mass(
    privacy_loss, count=1000, log_weight=None, support_bound=1.0
) -> delta_calculation.PrivacyLossSamples:
  """Samples all equal to a single privacy loss value."""
  if log_weight is None:
    return _PrivacyLossSamples(
        [privacy_loss], [count], support_bound=support_bound
    )
  # Counts cannot be combined with log weights, so repeat the sample instead.
  return _PrivacyLossSamples(
      [privacy_loss] * count,
      log_weights=[log_weight] * count,
      support_bound=support_bound,
  )


class PrivacyLossSamplesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('not_1d', dict(privacy_loss=[[1.0, 2.0]])),
      ('wrong_counts_shape', dict(privacy_loss=[1.0, 2.0], counts=[1, 2, 3])),
      (
          'wrong_log_weights_shape',
          dict(privacy_loss=[1.0, 2.0], log_weights=[0.0]),
      ),
      (
          'counts_with_log_weights',
          dict(privacy_loss=[1.0], counts=[2], log_weights=[0.0]),
      ),
  )
  def test_rejects_invalid_arguments(self, kwargs):
    with self.assertRaises(ValueError):
      delta_calculation.PrivacyLossSamples(**kwargs)

  @parameterized.parameters(([1.0, 2.0, 3.0], None, 3), ([1.0, 2.0], [2, 5], 7))
  def test_num_samples(self, privacy_loss, counts, expected):
    self.assertEqual(
        delta_calculation.PrivacyLossSamples(privacy_loss, counts).num_samples,
        expected,
    )


class DeltaCalculationTest(parameterized.TestCase):

  @parameterized.parameters(
      (100, 1, 1 / 2, 1.0),
      (1000, 1, 1 / 2, 1.0),
      (3, 2, 1 / 3, 1 / 2),
      (6, 2, 1 / 3, 1 / 4),
      (2, 3, 1 / 4, 1 / 3),
      (4, 3, 1 / 4, 1 / 9),
  )
  def test_hoeffding_bound(self, num_samples, tau, delta, expected_bound):
    # Check that Hoeffding bound is correct for hand-calculable cases.
    self.assertAlmostEqual(
        delta_calculation._hoeffding_bound(num_samples, tau, delta),
        expected_bound,
        places=5,
    )

  @parameterized.parameters([10**-i for i in range(1, 17)])
  def test_overall_delta(self, base_delta):
    # Check that overall delta is decreasing in num_samples, and always in
    # (base_delta, 1].
    overall_delta_10000 = delta_calculation.get_overall_delta(10000, base_delta)
    overall_delta_10001 = delta_calculation.get_overall_delta(10001, base_delta)
    self.assertLessEqual(overall_delta_10000, 1.0)
    self.assertLess(overall_delta_10001, overall_delta_10000)
    self.assertLess(base_delta, overall_delta_10001)

  @parameterized.parameters(1e-8, 1e-12, 1e-15)
  def test_kl_is_precise_for_tiny_arguments(self, q):
    # KL(q || tau q) = q (tau - 1 - ln tau) + O(q^2); the two terms of the KL
    # nearly cancel, so a naive implementation loses most digits here.
    tau = 2.0
    kl = delta_calculation._kl(q, tau * q)
    self.assertAlmostEqual(kl / (q * (tau - 1 - math.log(tau))), 1.0, places=6)

  @parameterized.parameters((0.3, 0.3), (1e-12, 1e-12), (1.0, 1.0))
  def test_kl_is_zero_for_equal_arguments(self, q, p):
    self.assertEqual(delta_calculation._kl(q, p), 0.0)

  def test_kl_is_infinite_when_p_is_one(self):
    self.assertEqual(delta_calculation._kl(0.5, 1.0), math.inf)

  @parameterized.parameters([(10 ** (i + 2), 10**-i) for i in range(1, 17)])
  def test_base_delta(self, num_samples, target_delta):
    base_delta = delta_calculation.get_base_delta(num_samples, target_delta)
    overall_delta = delta_calculation.get_overall_delta(num_samples, base_delta)
    self.assertLessEqual(overall_delta, target_delta)
    self.assertGreaterEqual(overall_delta, target_delta * (1 - 1e-5))

  @parameterized.parameters([(10**i, 10**-i) for i in range(1, 17)])
  def test_num_samples_too_small(self, num_samples, target_delta):
    with self.assertRaisesRegex(
        ValueError, 'Failed to find a valid base_delta'
    ):
      delta_calculation.get_base_delta(num_samples, target_delta)

  @parameterized.product(
      base_delta_multiplier=[0.5, 0.8, 0.9],
      target_delta=[10**-i for i in range(1, 15)],
  )
  def test_minimum_samples_to_calibrate(
      self, base_delta_multiplier, target_delta
  ):
    base_delta = base_delta_multiplier * target_delta
    num_samples = delta_calculation.minimum_samples_to_calibrate(
        base_delta, target_delta
    )
    self.assertLessEqual(
        delta_calculation.get_overall_delta(num_samples, base_delta),
        target_delta,
    )
    self.assertGreaterEqual(
        delta_calculation.get_base_delta(num_samples, target_delta),
        base_delta / 1.001,
    )
    self.assertGreater(
        delta_calculation.get_overall_delta(
            _slightly_fewer(num_samples), base_delta
        ),
        target_delta,
    )

  @parameterized.parameters(
      # Rescaling delta and support_bound together leaves the bound unchanged.
      (3, 2, 1 / 6, 0.5, 1 / 2),
      (4, 3, 1 / 8, 0.5, 1 / 9),
      (4, 3, 1 / 2, 2.0, 1 / 9),
      # tau * delta exceeds support_bound, so the true mean is impossible. This
      # is only reachable through rounding in callers.
      (3, 4, 1 / 6, 0.5, 0.0),
  )
  def test_hoeffding_bound_with_support_bound(
      self, num_samples, tau, delta, support_bound, expected_bound
  ):
    self.assertAlmostEqual(
        delta_calculation._hoeffding_bound(
            num_samples, tau, delta, support_bound
        ),
        expected_bound,
        places=5,
    )

  def test_hoeffding_bound_raises_for_tau_less_than_one(self):
    with self.assertRaises(ValueError):
      delta_calculation._hoeffding_bound(10, 0.5, 0.1)

  @parameterized.parameters([10**-i for i in range(1, 17)])
  def test_overall_delta_decreasing_in_support_bound(self, base_delta):
    overall_deltas = [
        delta_calculation.get_overall_delta(10000, base_delta, support_bound)
        for support_bound in [1.0, 0.5, 0.1, 0.01]
        if support_bound > base_delta
    ]
    self.assertEqual(
        overall_deltas[0],
        delta_calculation.get_overall_delta(10000, base_delta),
    )
    for larger, smaller in zip(overall_deltas, overall_deltas[1:]):
      self.assertLess(base_delta, smaller)
      self.assertLess(smaller, larger)

  @parameterized.named_parameters(
      ('at_most_base_delta', 0.1), ('greater_than_one', 1.5)
  )
  def test_overall_delta_raises_for_invalid_support_bound(self, support_bound):
    with self.assertRaises(ValueError):
      delta_calculation.get_overall_delta(100, 0.1, support_bound)

  @parameterized.product(
      delta=[1e-3, 1e-5, 1e-8], support_bound=[0.5, 0.05, 0.005]
  )
  def test_support_bound_is_equivalent_to_rescaling_deltas(
      self, delta, support_bound
  ):
    # Weighted samples in [0, support_bound] divided by support_bound are plain
    # samples in [0, 1] with every delta divided by support_bound.
    base_delta = delta / 2
    num_samples = delta_calculation.minimum_samples_to_calibrate(
        base_delta, delta, support_bound
    )
    self.assertAlmostEqual(
        num_samples
        / delta_calculation.minimum_samples_to_calibrate(
            base_delta / support_bound, delta / support_bound
        ),
        1.0,
        delta=1e-5,
    )
    self.assertAlmostEqual(
        delta_calculation.get_overall_delta(
            num_samples, base_delta, support_bound
        ),
        support_bound
        * delta_calculation.get_overall_delta(
            num_samples, base_delta / support_bound
        ),
        delta=1e-9 * delta,
    )
    self.assertAlmostEqual(
        delta_calculation.get_base_delta(num_samples, delta, support_bound),
        support_bound
        * delta_calculation.get_base_delta(num_samples, delta / support_bound),
        delta=1e-5 * delta,
    )

  @parameterized.product(
      target_delta=[10**-i for i in range(3, 9)],
      support_bound=[0.5, 0.05, 0.005],
  )
  def test_base_delta_with_support_bound(self, target_delta, support_bound):
    num_samples = int(100 / target_delta)
    base_delta = delta_calculation.get_base_delta(
        num_samples, target_delta, support_bound
    )
    overall_delta = delta_calculation.get_overall_delta(
        num_samples, base_delta, support_bound
    )
    self.assertLessEqual(overall_delta, target_delta)
    self.assertAlmostEqual(overall_delta, target_delta, places=5)
    # A smaller support bound is less conservative.
    self.assertGreater(
        base_delta,
        delta_calculation.get_base_delta(num_samples, target_delta),
    )

  @parameterized.product(
      base_delta_multiplier=[0.5, 0.9],
      target_delta=[10**-i for i in range(3, 9)],
      support_bound=[0.5, 0.05, 0.005],
  )
  def test_minimum_samples_to_calibrate_with_support_bound(
      self, base_delta_multiplier, target_delta, support_bound
  ):
    base_delta = base_delta_multiplier * target_delta
    num_samples = delta_calculation.minimum_samples_to_calibrate(
        base_delta, target_delta, support_bound
    )
    self.assertLessEqual(
        delta_calculation.get_overall_delta(
            num_samples, base_delta, support_bound
        ),
        target_delta,
    )
    self.assertGreater(
        delta_calculation.get_overall_delta(
            _slightly_fewer(num_samples), base_delta, support_bound
        ),
        target_delta,
    )
    self.assertLess(
        num_samples,
        delta_calculation.minimum_samples_to_calibrate(
            base_delta, target_delta
        ),
    )

  def test_overall_delta_per_verifier_matches_scalar_for_equal_entries(self):
    self.assertEqual(
        delta_calculation.get_overall_delta([1000, 1000], 1e-3, [0.5, 0.5]),
        delta_calculation.get_overall_delta(1000, 1e-3, 0.5),
    )

  def test_overall_delta_per_verifier_is_between_best_and_worst(self):
    # Per-verifier accounting is at least as good as treating every verifier as
    # the worst one, and no better than the worst verifier's own bound, since
    # the failure probability is dominated by the verifier with fewest samples
    # relative to its support bound.
    base_delta = 1e-3
    num_samples, support_bounds = [100_000, 2_000], [1.0, 0.02]
    per_verifier = delta_calculation.get_overall_delta(
        num_samples, base_delta, support_bounds
    )
    worst_case = delta_calculation.get_overall_delta(
        min(num_samples), base_delta, max(support_bounds)
    )
    self.assertLess(per_verifier, worst_case)
    for n, b in zip(num_samples, support_bounds):
      self.assertGreaterEqual(
          per_verifier, delta_calculation.get_overall_delta(n, base_delta, b)
      )

  @parameterized.product(
      target_delta=[1e-3, 1e-6, 1e-9],
      support_bounds=[[1.0, 0.5], [1.0, 0.1, 0.01], [0.3, 0.03]],
  )
  def test_minimum_samples_to_calibrate_per_verifier(
      self, target_delta, support_bounds
  ):
    base_delta = target_delta / 2
    num_samples = delta_calculation.minimum_samples_per_verification(
        base_delta, target_delta, support_bounds
    )
    self.assertEqual(num_samples.shape, (len(support_bounds),))
    self.assertLessEqual(
        delta_calculation.get_overall_delta(
            num_samples, base_delta, support_bounds
        ),
        target_delta,
    )
    # Sample counts scale with the support bound, up to the shared log factor.
    ratios = num_samples / np.asarray(support_bounds)
    self.assertLess(ratios.max() / ratios.min(), 1.5)
    # Slightly fewer samples for every verifier breaks the target.
    self.assertGreater(
        delta_calculation.get_overall_delta(
            _slightly_fewer(num_samples), base_delta, support_bounds
        ),
        target_delta,
    )
    # No verifier needs (materially) more samples than if all had the worst
    # support bound: tau is shared, so the worst verifier may be a few samples
    # off its own optimum. The ones with small support bounds need far fewer.
    worst_case = delta_calculation.minimum_samples_to_calibrate(
        base_delta, target_delta, max(support_bounds)
    )
    self.assertLessEqual(num_samples.max(), worst_case * (1 + 1e-5))
    self.assertLess(num_samples.min(), 0.75 * worst_case)

  @parameterized.named_parameters(
      ('all_at_most_epsilon', 3, [1, 2, 3], None, 0.0),
      ('all_greater_than_epsilon', 3, [3 + math.log(2), 1e9], None, 3 / 4),
      ('some_greater_than_epsilon', 3, [2, 3, 3 + math.log(2)], None, 1 / 6),
      (
          'all_greater_than_epsilon_with_counts',
          3,
          [3 + math.log(2), 1e9],
          [2, 1],
          2 / 3,
      ),
      (
          'some_greater_than_epsilon_with_counts',
          3,
          [2, 3, 3 + math.log(2)],
          [1, 1, 2],
          1 / 4,
      ),
      ('large_epsilon', 1000, [999, 1000, 1000 + math.log(2)], None, 1 / 6),
      ('epsilon_zero', 0, [-1, 0, math.log(2)], None, 1 / 6),
      (
          'large_epsilon_with_counts',
          1000,
          [999, 1000, 1000 + math.log(2)],
          [1, 1, 2],
          1 / 4,
      ),
      (
          'epsilon_zero_with_counts',
          0,
          [-1, 0, math.log(2)],
          [1, 1, 2],
          1 / 4,
      ),
  )
  def test_delta_from_epsilon_and_samples(
      self, epsilon, samples, counts, expected_delta
  ):
    delta = delta_calculation.delta_from_epsilon_and_samples(
        epsilon, _PrivacyLossSamples(samples, counts)
    )
    self.assertAlmostEqual(delta, expected_delta, places=5)

  @parameterized.named_parameters(
      ('zero_log_weights', [0.0, 0.0], 3 / 4),
      ('log_weights', [math.log(0.25), math.log(0.5)], 5 / 16),
  )
  def test_delta_from_epsilon_and_samples_with_log_weights(
      self, log_weights, expected_delta
  ):
    delta = delta_calculation.delta_from_epsilon_and_samples(
        3, _PrivacyLossSamples([3 + math.log(2), 1e9], log_weights=log_weights)
    )
    self.assertAlmostEqual(delta, expected_delta, places=5)

  def test_delta_from_epsilon_and_samples_with_huge_log_weight(self):
    # A sample below epsilon contributes 0 regardless of its weight, even when
    # exp(log_weight) overflows.
    delta = delta_calculation.delta_from_epsilon_and_samples(
        3,
        _PrivacyLossSamples([1.0, 3 + math.log(2)], log_weights=[1000.0, 0.0]),
    )
    self.assertAlmostEqual(delta, 1 / 4)

  def test_delta_from_epsilon_and_samples_rejects_weights_with_other_event(
      self,
  ):
    with self.assertRaises(ValueError):
      delta_calculation.delta_from_epsilon_and_samples(
          3,
          _PrivacyLossSamples([1.0, 2.0], log_weights=[0.0, 0.0]),
          other_event=dp_accounting.NoOpDpEvent(),
      )

  @parameterized.parameters(
      ([1, 2, 3], None),
      ([4, 5], None),
      ([2, 3, 4], None),
      ([2, 3, 4], [2, 1, 1]),
      ([2, 3, 4], [1, 1, 2]),
  )
  def test_composition_with_no_op_event(self, samples, counts):
    """No-op DP event should have same result as no event."""
    samples = _PrivacyLossSamples(samples, counts)
    delta_1 = delta_calculation.delta_from_epsilon_and_samples(
        3, samples, dp_accounting.NoOpDpEvent()
    )
    delta_2 = delta_calculation.delta_from_epsilon_and_samples(3, samples)
    self.assertBetween(delta_1, delta_2 * (1 - 1e-7), delta_2 * (1 + 1e-7))

  @parameterized.parameters(
      ([1, 2, 3], None),
      ([4, 5], None),
      ([2, 3, 4], None),
      ([2, 3, 4], [2, 1, 1]),
      ([2, 3, 4], [1, 1, 2]),
  )
  def test_composition_with_nonprivate_dp_event(self, samples, counts):
    """Non-private DP event should force delta = 1 always."""
    delta = delta_calculation.delta_from_epsilon_and_samples(
        3,
        _PrivacyLossSamples(samples, counts),
        dp_accounting.NonPrivateDpEvent(),
    )
    self.assertEqual(delta, 1.0)

  def test_gaussian_mc_with_gaussian_pld(self):
    """Test that composing MC and PLD matches PLD alone."""
    rng = np.random.default_rng(0)
    # Samples from PLD for Gaussian mechanism with noise multiplier 1.0.
    samples = rng.normal(loc=0.5, size=100_000)
    other_event = dp_accounting.pld.PLDAccountant(
        value_discretization_interval=1e-2
    ).compose(dp_accounting.GaussianDpEvent(1.0))
    accountant = dp_accounting.pld.PLDAccountant(
        value_discretization_interval=1e-2
    )
    accountant.compose(dp_accounting.GaussianDpEvent(1 / (2**0.5)))
    delta_1 = delta_calculation.delta_from_epsilon_and_samples(
        1.0, _PrivacyLossSamples(samples), other_event=other_event
    )
    delta_2 = accountant.get_delta(1.0)
    self.assertAlmostEqual(delta_1, delta_2, places=3)

  _FAILURE_DELTA = delta_calculation.get_base_delta(1000, 0.1)

  @parameterized.named_parameters(
      ('0_is_best', [[1] * 1000, [10] * 1000], None, None, None, (True, 0)),
      (
          'support_size_two_0_is_best',
          [[1] * 1000, [1] * 500 + [10] * 500],
          None,
          None,
          None,
          (True, 0),
      ),
      ('1_is_best', [[1] * 1000, [1] * 1000], None, None, None, (True, 1)),
      (
          'simple_0_fails',
          [[10] * 1000, [10] * 1000],
          None,
          None,
          None,
          (False, _FAILURE_DELTA),
      ),
      (
          '1_fails_2_passes',
          [[1] * 1000, [10] * 1000, [1] * 1000],
          None,
          None,
          None,
          (True, 0),
      ),
      (
          'min_samples_used_for_base_delta',
          [[10] * 2000, [10] * 1000],
          None,
          None,
          None,
          (False, _FAILURE_DELTA),
      ),
      (
          'simple_0_is_best_with_counts',
          [[1], [10]],
          [[1000], [1000]],
          None,
          None,
          (True, 0),
      ),
      (
          '0_passes_positive_but_fails_negative',
          [[1] * 1000, [10] * 1000],
          None,
          [[10] * 1000, [10] * 1000],
          None,
          (False, _FAILURE_DELTA),
      ),
      (
          '0_passes_1_fails_positive',
          [[1] * 1000, [10] * 1000],
          None,
          [[1] * 1000, [1] * 1000],
          None,
          (True, 0),
      ),
      (
          '0_passes_1_fails_negative',
          [[1] * 1000, [1] * 1000],
          None,
          [[1] * 1000, [10] * 1000],
          None,
          (True, 0),
      ),
      (
          '0_passes_positive_but_fails_negative_with_counts',
          [[1], [10]],
          [[1000], [1000]],
          [[10], [10]],
          [[1000], [1000]],
          (False, _FAILURE_DELTA),
      ),
      (
          '0_fails_negative_1_fails_positive_with_counts',
          [[1], [10]],
          [[1000], [1000]],
          [[10], [1]],
          [[1000], [1000]],
          (False, _FAILURE_DELTA),
      ),
      (
          '1_fails_negative_with_counts',
          [[1], [10]],
          [[1000], [1000]],
          [[1], [1]],
          [[1000], [1000]],
          (True, 0),
      ),
  )
  def test_perform_calibration_from_samples(
      self,
      positive_samples,
      positive_counts,
      negative_samples,
      negative_counts,
      expected_result,
  ):
    result = delta_calculation.perform_calibration_from_samples(
        1.0,
        0.1,
        positive_samples=_samples(positive_samples, positive_counts),
        negative_samples=(
            None
            if negative_samples is None
            else _samples(negative_samples, negative_counts)
        ),
    )
    self.assertEqual(result, expected_result)

  @parameterized.named_parameters(
      (
          'no_op_dp_event_no_effect_first_passes',
          [[1], [1000]],
          dp_accounting.NoOpDpEvent(),
          (True, 0),
      ),
      (
          'empty_accountant_no_effect_first_passes',
          [[1], [1000]],
          dp_accounting.pld.PLDAccountant(),
          (True, 0),
      ),
      (
          'no_op_dp_event_no_effect_all_passes',
          [[1], [1]],
          dp_accounting.NoOpDpEvent(),
          (True, 1),
      ),
      (
          'no_op_dp_event_no_effect_fails',
          [[1000], [1000]],
          dp_accounting.NoOpDpEvent(),
          (False, _FAILURE_DELTA),
      ),
      (
          'non_private_event_fails',
          [[1], [1]],
          dp_accounting.NonPrivateDpEvent(),
          (False, _FAILURE_DELTA),
      ),
      (
          'non_private_event_in_accoutant_fails',
          [[1], [1]],
          dp_accounting.pld.PLDAccountant().compose(
              dp_accounting.NonPrivateDpEvent()
          ),
          (False, _FAILURE_DELTA),
      ),
      (
          'list_of_events_determines_output',
          [[1], [1]],
          [
              dp_accounting.NoOpDpEvent(),
              dp_accounting.NonPrivateDpEvent(),
          ],
          (True, 0),
      ),
      (
          'list_of_accountants_determines_output',
          [[1], [1]],
          [
              dp_accounting.pld.PLDAccountant(),
              dp_accounting.pld.PLDAccountant().compose(
                  dp_accounting.NonPrivateDpEvent()
              ),
          ],
          (True, 0),
      ),
      (
          'list_of_events_doesnt_determine_output',
          [[1000], [1]],
          [
              dp_accounting.NoOpDpEvent(),
              dp_accounting.NonPrivateDpEvent(),
          ],
          (False, _FAILURE_DELTA),
      ),
  )
  def test_perform_calibration_from_samples_with_other_event(
      self, positive_samples, other_event, expected_result
  ):
    result = delta_calculation.perform_calibration_from_samples(
        1.0,
        0.1,
        positive_samples=_samples(positive_samples, [[1000], [1000]]),
        other_event=other_event,
    )
    self.assertEqual(result, expected_result)

  _FAILURE_DELTA_HALF_SUPPORT = delta_calculation.get_base_delta(1000, 0.1, 0.5)

  @parameterized.named_parameters(
      (
          'positive_log_weights_rescue_0',
          dict(
              positive_samples=[
                  _point_mass(10, log_weight=-20),
                  _point_mass(10, log_weight=0),
              ]
          ),
          (True, 0),
      ),
      (
          'negative_log_weights_sink_1',
          dict(
              positive_samples=[_point_mass(1)] * 2,
              negative_samples=[
                  _point_mass(10, log_weight=-20),
                  _point_mass(10, log_weight=0),
              ],
          ),
          (True, 0),
      ),
      (
          'support_bound_all_pass',
          dict(positive_samples=[_point_mass(1, support_bound=0.5)] * 2),
          (True, 1),
      ),
      (
          'support_bound_all_fail',
          dict(positive_samples=[_point_mass(10, support_bound=0.5)] * 2),
          (False, _FAILURE_DELTA_HALF_SUPPORT),
      ),
      (
          'support_bound_uses_max',
          dict(
              positive_samples=[
                  _point_mass(10, support_bound=0.2),
                  _point_mass(10, support_bound=0.5),
              ]
          ),
          (False, _FAILURE_DELTA_HALF_SUPPORT),
      ),
      (
          'none_negative_entry_skips_negative_check',
          dict(
              positive_samples=[_point_mass(1)] * 2,
              negative_samples=[None, _point_mass(1)],
          ),
          (True, 1),
      ),
      (
          'none_negative_entry_skips_negative_check_only_for_its_index',
          dict(
              positive_samples=[_point_mass(1)] * 2,
              negative_samples=[None, _point_mass(10)],
          ),
          (True, 0),
      ),
      (
          'none_negative_entry_does_not_count_toward_min_samples',
          dict(
              positive_samples=[_point_mass(10, count=2000)] * 2,
              negative_samples=[None, _point_mass(1, count=2000)],
          ),
          (False, delta_calculation.get_base_delta(2000, 0.1)),
      ),
  )
  def test_perform_calibration_from_samples_with_weights_and_bounds(
      self, kwargs, expected_result
  ):
    result = delta_calculation.perform_calibration_from_samples(
        1.0, 0.1, **kwargs
    )
    self.assertEqual(result, expected_result)

  def test_perform_calibration_accounts_per_verification(self):
    # 200 samples suffice with support bound 0.01 but not with support bound 1,
    # so this only passes because each verification uses its own bound.
    target_delta = 1e-3
    with self.assertRaises(ValueError):
      delta_calculation.get_base_delta(200, target_delta, 1.0)
    result = delta_calculation.perform_calibration_from_samples(
        1.0,
        target_delta,
        positive_samples=[
            _point_mass(1, count=200, support_bound=0.01),
            _point_mass(1, count=20_000),
        ],
    )
    self.assertEqual(result, (True, 1))

  @parameterized.named_parameters(
      ('negative_samples_length', dict(negative_samples=[_point_mass(1)])),
      (
          'support_bound_greater_than_one',
          dict(positive_samples=[_point_mass(1, support_bound=1.5)] * 2),
      ),
      (
          'log_weights_with_other_event',
          dict(
              positive_samples=[_point_mass(1, log_weight=0.0)] * 2,
              other_event=dp_accounting.NoOpDpEvent(),
          ),
      ),
  )
  def test_perform_calibration_from_samples_raises(self, kwargs):
    kwargs.setdefault('positive_samples', [_point_mass(1)] * 2)
    with self.assertRaises(ValueError):
      delta_calculation.perform_calibration_from_samples(1.0, 0.1, **kwargs)


if __name__ == '__main__':
  absltest.main()

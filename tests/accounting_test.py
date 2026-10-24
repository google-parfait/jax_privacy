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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import dp_accounting
from jax_privacy import accounting

MAKE_EVENT_FNS = (
    lambda nm: accounting.dpsgd_event(nm, 128, sampling_prob=0.01),
    functools.partial(
        accounting.amplified_bandmf_event,
        iterations=128,
        num_bands=16,
        sampling_prob=0.01,
    ),
)

PLD_EVENT_FNS = (
    functools.partial(
        accounting.truncated_dpsgd_event,
        iterations=128,
        sampling_prob=0.01,
        num_examples=1000,
        truncated_batch_size=16,
    ),
    functools.partial(
        accounting.truncated_amplified_bandmf_event,
        iterations=128,
        num_bands=16,
        sampling_prob=0.01,
        largest_group_size=1000,
        truncated_batch_size=16,
    ),
    functools.partial(
        accounting.random_allocation_dpsgd_event,
        iterations=100,
        total_participations=4,
    ),
    # Random allocation PLD is expensive, so keep this entry small.
    functools.partial(
        accounting.random_allocation_bandmf_event,
        iterations=12,
        num_bands=4,
        total_participations=1,
    ),
)

RDP_REPLACE_EVENT_FNS = (
    functools.partial(
        accounting.fixed_dpsgd_event,
        iterations=128,
        dataset_size=1000,
        batch_size=16,
        replace=False,
    ),
)

PLD_ACCOUNTANT = functools.partial(
    dp_accounting.pld.PLDAccountant, value_discretization_interval=1e-3
)
RDP_ACCOUNTANT = dp_accounting.rdp.RdpAccountant  # pylint: disable=invalid-name
RDP_REPLACE_ACCOUNTANT = functools.partial(
    dp_accounting.rdp.RdpAccountant,
    neighboring_relation=(
        dp_accounting.privacy_accountant.NeighboringRelation.REPLACE_ONE
    ),
)


def _make_test_case(event_fn, accountant):
  return {"event_fn": event_fn, "accountant": accountant}


TEST_CASES = (
    tuple(
        _make_test_case(ev, PLD_ACCOUNTANT)
        for ev in MAKE_EVENT_FNS + PLD_EVENT_FNS
    )
    + tuple(
        _make_test_case(event_fn, RDP_ACCOUNTANT) for event_fn in MAKE_EVENT_FNS
    )
    + tuple(
        _make_test_case(event_fn, RDP_REPLACE_ACCOUNTANT)
        for event_fn in RDP_REPLACE_EVENT_FNS
    )
)


class AccountingTest(parameterized.TestCase):

  @parameterized.parameters(*TEST_CASES)
  def test_get_epsilon(self, event_fn, accountant):
    event = event_fn(1.0)
    epsilon = accountant().compose(event).get_epsilon(target_delta=1e-6)
    self.assertGreaterEqual(epsilon, 0)

  @parameterized.parameters(*TEST_CASES)
  def test_calibration(self, event_fn, accountant):
    nm = dp_accounting.calibrate_dp_mechanism(
        accountant,
        event_fn,
        target_epsilon=1.0,
        target_delta=1e-6,
    )
    eps = accountant().compose(event_fn(nm)).get_epsilon(target_delta=1e-6)
    self.assertAlmostEqual(eps, 1.0, places=4)

  def test_fixed_dpsgd_event_replace_true_type(self):
    event = accounting.fixed_dpsgd_event(
        1.0,
        5,
        dataset_size=10,
        batch_size=3,
        replace=True,
    )
    self.assertIsInstance(event, dp_accounting.dp_event.SelfComposedDpEvent)
    self.assertIsInstance(
        event.event, dp_accounting.dp_event.SampledWithReplacementDpEvent
    )

  def test_dpsgd_event_zero_noise_does_not_crash(self):
    """noise_multiplier=0 is permitted and must not raise."""
    event = accounting.dpsgd_event(0.0, 10, sampling_prob=0.01)
    self.assertIsInstance(
        event.event.event, dp_accounting.dp_event.GaussianDpEvent
    )
    self.assertEqual(event.event.event.noise_multiplier, 0.0)

  def test_random_allocation_dpsgd_event_structure(self):
    event = accounting.random_allocation_dpsgd_event(
        noise_multiplier=2.5,
        iterations=50,
        total_participations=5,
    )
    self.assertIsInstance(event, dp_accounting.dp_event.RandomAllocationDpEvent)
    self.assertEqual(event.num_steps, 50)
    self.assertEqual(event.num_selected, 5)
    self.assertIsInstance(event.event, dp_accounting.dp_event.GaussianDpEvent)
    self.assertEqual(event.event.noise_multiplier, 2.5)

  def test_random_allocation_validation(self):
    with self.assertRaises(ValueError):
      accounting.random_allocation_dpsgd_event(
          noise_multiplier=1.0,
          iterations=10,
          total_participations=15,
      )

  @parameterized.parameters(
      # (iterations, num_bands, expected num_steps)
      (50, 1, 50),
      (50, 5, 10),
      # iterations is not a multiple of num_bands; accounting uses
      # ceil(iterations / num_bands) steps with truncation.
      (20, 3, 7),
  )
  def test_random_allocation_bandmf_event_structure(
      self, iterations, num_bands, expected_num_steps
  ):
    event = accounting.random_allocation_bandmf_event(
        noise_multiplier=2.5,
        iterations=iterations,
        num_bands=num_bands,
        total_participations=3,
    )
    self.assertIsInstance(event, dp_accounting.dp_event.RandomAllocationDpEvent)
    self.assertEqual(event.num_steps, expected_num_steps)
    self.assertEqual(event.num_selected, 3)
    self.assertIsInstance(event.event, dp_accounting.dp_event.GaussianDpEvent)
    self.assertEqual(event.event.noise_multiplier, 2.5)

  @parameterized.parameters(
      # num_bands must be positive.
      {"num_bands": 0, "total_participations": 1},
      # total_participations must be at most ceil(iterations / num_bands).
      {"num_bands": 5, "total_participations": 11},
  )
  def test_random_allocation_bandmf_validation(self, **kwargs):
    with self.assertRaises(ValueError):
      accounting.random_allocation_bandmf_event(
          noise_multiplier=1.0, iterations=50, **kwargs
      )

  @parameterized.parameters(0, -1)
  def test_amplified_bandmf_event_rejects_non_positive_num_bands(
      self, num_bands
  ):
    with self.assertRaisesRegex(ValueError, "num_bands"):
      accounting.amplified_bandmf_event(
          1.0, 128, num_bands=num_bands, sampling_prob=0.01
      )
    with self.assertRaisesRegex(ValueError, "num_bands"):
      accounting.truncated_amplified_bandmf_event(
          1.0,
          128,
          num_bands=num_bands,
          sampling_prob=0.01,
          largest_group_size=10,
          truncated_batch_size=8,
      )


if __name__ == "__main__":
  absltest.main()

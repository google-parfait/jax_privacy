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

"""Tests for Monte Carlo discretized privacy-loss sample accumulation."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax_privacy import batch_selection
from jax_privacy.experimental.monte_carlo import jax_privacy_loss_samples
import numpy as np
import scipy.stats

jax.config.update('jax_enable_x64', True)

_C_COL = np.array([0.5, 0.25], dtype=np.float64)
_SIGMA = 1.0

_STRATEGIES = [
    batch_selection.MinSepRandomAllocation(2, 12, 3),
    batch_selection.MinSepRandomAllocation(2, 12, 3, random_rotation=True),
    batch_selection.MinSepRandomAllocation(2, 12, 3, sampling_prob=0.5),
    batch_selection.RandomAllocationSampling(2, 12, 3),
    batch_selection.RandomAllocationSampling(2, 12, 3, sampling_prob=0.5),
]


class JaxPrivacyLossSamplesTest(parameterized.TestCase):

  @parameterized.parameters(
      batch_selection.MinSepRandomAllocation(1, 1, 1),
      batch_selection.RandomAllocationSampling(1, 1, 1),
  )
  def test_matches_gaussian_bin_probabilities(self, strategy):
    # With T=1, K=1, and c_col=[1.0], both L(y) (y ~ P) and -L(y) (y ~ Q) follow
    # N(mu, 2*mu) with mu = 1/(2*sigma^2). A 5-point grid over [-0.5, 1.5] puts
    # ~11-19% mass in every bin (lower clamp, interior bins, and +inf overflow).
    sigma = 1.0
    num_samples = 200_000
    grid_lo, grid_hi, grid_size = -0.5, 1.5, 5
    pos, neg = jax_privacy_loss_samples.privacy_loss_samples(
        strategy,
        sigma,
        np.array([1.0]),
        jax.random.PRNGKey(0),
        num_samples=num_samples,
        microbatch_size=2000,
        grid_lo=grid_lo,
        grid_hi=grid_hi,
        grid_size=grid_size,
    )
    mu = 1.0 / (2.0 * sigma**2)
    finite_grid = np.linspace(grid_lo, grid_hi, grid_size)
    cdf = scipy.stats.norm.cdf(finite_grid, loc=mu, scale=np.sqrt(2.0 * mu))
    expected_probs = np.diff(np.r_[0.0, cdf, 1.0])
    expected_grid = np.r_[finite_grid, np.inf]
    for samples in (pos, neg):
      self.assertEqual(samples.num_samples, num_samples)
      np.testing.assert_array_equal(samples.privacy_loss, expected_grid)
      np.testing.assert_allclose(
          samples.counts / num_samples, expected_probs, atol=0.01
      )

  @parameterized.parameters(*_STRATEGIES)
  def test_change_of_measure_identity(self, strategy):
    # For any dominating pair (P, Q), E_P[exp(-L)] = 1 and E_Q[exp(L)] = 1; this
    # fails if the pattern sampler and privacy-loss scorer do not match.
    num_samples = 10_000
    pos, neg = jax_privacy_loss_samples.privacy_loss_samples(
        strategy,
        _SIGMA,
        _C_COL,
        jax.random.PRNGKey(42),
        num_samples=num_samples,
        microbatch_size=1000,
    )
    for samples in (pos, neg):
      self.assertEqual(samples.num_samples, num_samples)
      self.assertTrue(np.all(samples.counts > 0))
      mean_exp_neg_loss = np.average(
          np.exp(-samples.privacy_loss), weights=samples.counts
      )
      self.assertAlmostEqual(mean_exp_neg_loss, 1.0, delta=0.03)

  def test_microbatch_invariance_and_rounding(self):
    strategy = _STRATEGIES[0]
    kwargs = dict(num_samples=256, grid_lo=-30.0, grid_hi=30.0, grid_size=601)
    ref_pos, ref_neg = jax_privacy_loss_samples.privacy_loss_samples(
        strategy, _SIGMA, _C_COL, jax.random.PRNGKey(7), **kwargs
    )
    for microbatch_size in (64, 256):
      pos, neg = jax_privacy_loss_samples.privacy_loss_samples(
          strategy,
          _SIGMA,
          _C_COL,
          jax.random.PRNGKey(7),
          microbatch_size=microbatch_size,
          **kwargs,
      )
      np.testing.assert_array_equal(pos.privacy_loss, ref_pos.privacy_loss)
      np.testing.assert_array_equal(pos.counts, ref_pos.counts)
      np.testing.assert_array_equal(neg.privacy_loss, ref_neg.privacy_loss)
      np.testing.assert_array_equal(neg.counts, ref_neg.counts)

    # Non-dividing num_samples rounds up to a whole multiple of microbatch_size.
    pos, neg = jax_privacy_loss_samples.privacy_loss_samples(
        strategy,
        _SIGMA,
        _C_COL,
        jax.random.PRNGKey(7),
        **(kwargs | dict(num_samples=100, microbatch_size=64)),
    )
    self.assertEqual(pos.num_samples, 128)
    self.assertEqual(neg.num_samples, 128)

  @parameterized.parameters(
      dict(num_samples=0),
      dict(microbatch_size=0),
      dict(grid_size=0),
      dict(strategy=batch_selection.CyclicPoissonSampling(0.5, 12)),
  )
  def test_invalid_inputs_raise(self, **kwargs):
    defaults = dict(
        strategy=_STRATEGIES[0],
        noise_multiplier=_SIGMA,
        c_col=_C_COL,
        key=jax.random.PRNGKey(9),
        num_samples=100,
    )
    with self.assertRaises(ValueError):
      jax_privacy_loss_samples.privacy_loss_samples(**(defaults | kwargs))


if __name__ == '__main__':
  absltest.main()

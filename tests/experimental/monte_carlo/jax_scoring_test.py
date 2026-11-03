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

"""Tests for privacy-loss scoring, checked against brute-force enumeration.

Verifies that the dynamic-programming recurrences of Section 3.3 and Appendix B
of https://arxiv.org/abs/2610.12712 match naive summation over all valid
participation subsets.
"""

import functools
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax_privacy import batch_selection
from jax_privacy.experimental.monte_carlo import jax_scoring
import numpy as np
import scipy.special

jax.config.update('jax_enable_x64', True)


def _llrs_np(c_col, sample, sigma):
  """Per-iteration Gaussian LLRs, computed directly for a single sample."""
  t = sample.shape[0]
  b = c_col.shape[0]
  dots = np.array([np.dot(c_col[: t - i], sample[i : i + b]) for i in range(t)])
  sq = np.array([np.sum(c_col[: min(b, t - i)] ** 2) for i in range(t)])
  return (2.0 * dots - sq) / (2.0 * sigma**2)


def _subset_llr(llr, subset, q):
  """Log E over kept subsets A of subset (each kept iid w.p. q)."""
  if q == 1.0:
    return sum(llr[j] for j in subset)
  # Enumerate every kept subset A explicitly, independent of the scorer's
  # closed-form per-position fold.
  terms = []
  for r in range(len(subset) + 1):
    for kept in itertools.combinations(subset, r):
      log_w = r * np.log(q) + (len(subset) - r) * np.log1p(-q)
      terms.append(log_w + sum(llr[j] for j in kept))
  return scipy.special.logsumexp(terms)


def _is_valid_subset(strategy, s):
  """Returns whether K-subset s is in the support of strategy."""
  if isinstance(strategy, batch_selection.MinSepRandomAllocation):
    # Algorithm 2 / Theorem 6 (linear) and Theorem 8 / Appendix B (cyclic) of
    # https://arxiv.org/abs/2610.12712.
    b = strategy.min_sep
    t = strategy.iterations
    wrap_ok = not strategy.random_rotation or s[0] + t - s[-1] >= b
    return min(np.diff(s), default=b) >= b and wrap_ok
  # Algorithm 1 / Theorem 5 of https://arxiv.org/abs/2610.12712 (for b | t).
  assert strategy.iterations % strategy.cycle_length == 0
  return len({x % strategy.cycle_length for x in s}) == 1


def _brute_force_privacy_loss(strategy, sample, sigma, c_col):
  """Brute-force log privacy loss by enumerating all valid K-subsets."""
  t = strategy.iterations
  k = strategy.total_participations
  q = strategy.sampling_prob
  llr = _llrs_np(c_col, sample, sigma)
  all_subsets = itertools.combinations(range(t), k)
  subsets = [s for s in all_subsets if _is_valid_subset(strategy, s)]
  terms = [_subset_llr(llr, s, q) for s in subsets]
  return scipy.special.logsumexp(terms) - np.log(len(terms))


class MinSepRandomAllocationScoringTest(parameterized.TestCase):

  @parameterized.product(
      t=[9, 12],
      k=[1, 2, 3],
      b=[2, 3],
      random_rotation=[False, True],
      q=[1.0, 0.5],
  )
  def test_matches_brute_force(self, t, k, b, random_rotation, q):
    seed = t * 100 + k * 10 + b + int(q * 100) + int(random_rotation) * 7
    rng = np.random.default_rng(seed)
    c_col = rng.standard_normal(b)
    sigma = 1.3
    sample = rng.standard_normal(t)
    strategy = batch_selection.MinSepRandomAllocation(
        iterations=t,
        total_participations=k,
        min_sep=b,
        random_rotation=random_rotation,
        sampling_prob=q,
    )
    got = jax_scoring.compute_privacy_loss(strategy, sample, sigma, c_col)
    expected = _brute_force_privacy_loss(strategy, sample, sigma, c_col)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


class RandomAllocationScoringTest(parameterized.TestCase):

  @parameterized.product(
      # (t=6, k=2, b=3) has k == t // b (every candidate selected), covering the
      # cyclic-Poisson reduction when q < 1.
      t=[6, 12],
      k=[1, 2],
      b=[2, 3],
      q=[1.0, 0.5],
  )
  def test_matches_brute_force(self, t, k, b, q):
    rng = np.random.default_rng(t * 100 + k * 10 + b + int(q * 100) + 1)
    c_col = rng.standard_normal(b)
    sigma = 0.9
    sample = rng.standard_normal(t)
    strategy = batch_selection.RandomAllocationSampling(
        iterations=t, total_participations=k, cycle_length=b, sampling_prob=q
    )
    got = jax_scoring.compute_privacy_loss(strategy, sample, sigma, c_col)
    expected = _brute_force_privacy_loss(strategy, sample, sigma, c_col)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


class ScoringApiTest(parameterized.TestCase):

  def test_min_sep_one_matches_across_strategies(self):
    # With b == 1 both strategies are plain random allocation, so they agree.
    rng = np.random.default_rng(0)
    t, k = 6, 2
    c_col = np.array([1.0])
    sigma = 1.1
    sample = rng.standard_normal(t)
    min_sep = batch_selection.MinSepRandomAllocation(
        iterations=t, total_participations=k, min_sep=1
    )
    ra = batch_selection.RandomAllocationSampling(
        iterations=t, total_participations=k, cycle_length=1
    )
    min_sep_loss = jax_scoring.compute_privacy_loss(
        min_sep, sample, sigma, c_col
    )
    ra_loss = jax_scoring.compute_privacy_loss(ra, sample, sigma, c_col)
    np.testing.assert_allclose(min_sep_loss, ra_loss, rtol=1e-6, atol=1e-6)

  def test_jit_and_vmap_over_samples(self):
    rng = np.random.default_rng(3)
    strategy = batch_selection.RandomAllocationSampling(
        iterations=8, total_participations=2, cycle_length=2
    )
    c_col = np.array([1.0, 0.5])
    samples = rng.standard_normal((8, 5))  # (iterations, num_samples)
    score_fn = lambda s: jax_scoring.compute_privacy_loss(
        strategy, s, 1.0, c_col
    )
    fn = jax.jit(jax.vmap(score_fn, in_axes=1))
    self.assertEqual(fn(samples).shape, (5,))

  def test_requires_1d_sample(self):
    strategy = batch_selection.MinSepRandomAllocation(
        iterations=8, total_participations=2, min_sep=2
    )
    bad_sample = np.zeros((8, 1))
    with self.assertRaises(ValueError):
      jax_scoring.compute_privacy_loss(
          strategy, bad_sample, 1.0, np.array([1.0])
      )

  @parameterized.parameters('min_sep', 'random_allocation')
  def test_sampling_prob_one_is_exact_noop(self, kind):
    rng = np.random.default_rng(4)
    t, k, b = 10, 3, 2
    c_col = rng.standard_normal(b)
    sigma = 1.2
    sample = rng.standard_normal(t)
    if kind == 'min_sep':
      make = functools.partial(
          batch_selection.MinSepRandomAllocation,
          iterations=t,
          total_participations=k,
          min_sep=b,
      )
    else:
      make = functools.partial(
          batch_selection.RandomAllocationSampling,
          iterations=t,
          total_participations=k,
          cycle_length=b,
      )
    s_default = make()
    s_one = make(sampling_prob=1.0)
    s_sub = make(sampling_prob=0.9)
    default = jax_scoring.compute_privacy_loss(s_default, sample, sigma, c_col)
    explicit = jax_scoring.compute_privacy_loss(s_one, sample, sigma, c_col)
    np.testing.assert_array_equal(default, explicit)
    # A slightly subsampled variant must differ, confirming the branch is live.
    subsampled = jax_scoring.compute_privacy_loss(s_sub, sample, sigma, c_col)
    self.assertNotAlmostEqual(float(default), float(subsampled))


if __name__ == '__main__':
  absltest.main()

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

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import dp_accounting
import jax.numpy as jnp
from jax_privacy import auditing
import numpy as np
import scipy.special
import scipy.stats

_binom = scipy.stats.binom
_logistic = scipy.special.expit


def _signed_area(a, b, c):
  """Computes (twice) the signed area of the triangle formed by three points."""
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


_rotations = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
_inversions = [[0, 2, 1], [1, 0, 2], [2, 1, 0]]


def _one_run_p_value_naive(
    m: int, n_guess: int, n_correct: int, eps: float, delta: float
) -> float:
  """Naive implementation of the one-shot p-value."""
  q = _logistic(eps)
  beta = _binom.sf(n_correct - 1, n_guess, q)

  alpha = 0
  alpha_sum = 0
  for i in range(1, n_correct + 1):
    alpha_sum += _binom.pmf(n_correct - i, n_guess, q)
    if alpha_sum > i * alpha:
      alpha = alpha_sum / i

  return min(beta + alpha * delta * 2 * m, 1)


def _deterministic_normal(mu, sigma, n):
  """Generates array of scores deterministically normally distributed."""
  cdf_vals = np.linspace(0, 1, n, endpoint=False) + 1 / (2 * n)
  return scipy.stats.norm.ppf(cdf_vals) * sigma + mu


def _scores_with_circular_roc(
    n_in: int, n_out: int
) -> tuple[np.ndarray, np.ndarray]:
  """Generates scores such that the ROC curve is a circular arc."""
  out_canary_scores = np.linspace(0, 1, n_out)
  in_canary_scores = np.sqrt(1 - np.linspace(0, 1, n_in) ** 2)
  return in_canary_scores, out_canary_scores


class CanaryScoreAuditorTest(parameterized.TestCase):

  def test_bootstrap_params_empty_quantiles(self):
    with self.assertRaisesRegex(ValueError, 'quantiles cannot be empty'):
      auditing.BootstrapParams(quantiles=[])

  @parameterized.named_parameters(
      ('scalar', 0.5),
      ('2d', [[0.25, 0.75]]),
  )
  def test_bootstrap_params_non_1d_quantiles(self, quantiles):
    with self.assertRaisesRegex(ValueError, 'quantiles must be a 1-D sequence'):
      auditing.BootstrapParams(quantiles=quantiles)

  @parameterized.named_parameters(('zero', (0, 0.5)), ('one', (0.5, 1)))
  def test_bootstrap_params_quantiles_out_of_range(self, quantiles):
    with self.assertRaisesRegex(ValueError, 'quantiles must be in'):
      auditing.BootstrapParams(quantiles=quantiles)

  def test_bootstrap_params_confidence_interval_illegal_confidence(self):
    for confidence in [-1, 0, 1, 2]:
      with self.assertRaisesRegex(ValueError, 'confidence must be in'):
        auditing.BootstrapParams.confidence_interval(confidence=confidence)

  @parameterized.named_parameters(
      ('95_percent', 0.95, [0.025, 0.975]),
      ('99_percent', 0.99, [0.005, 0.995]),
  )
  def test_bootstrap_params_confidence_interval_correct_quantiles(
      self,
      confidence,
      expected_quantiles,
  ):
    params = auditing.BootstrapParams.confidence_interval(confidence=confidence)
    np.testing.assert_almost_equal(params.quantiles, expected_quantiles)

  @parameterized.product(
      n=[10, 100, 1000],
      k_frac=[0.3, 0.4, 0.5, 0.8, 0.9],
      significance=[0.1, 0.03, 0.01],
  )
  def test_clopper_pearson_upper(self, n, k_frac, significance):
    trials = 1_000_000
    k = k_frac * n
    p = auditing._clopper_pearson_upper(k, n, significance)
    rng = np.random.default_rng(seed=0xBAD5EED)
    np.testing.assert_allclose(
        np.mean(rng.binomial(n, p, size=trials) <= k),
        significance,
        rtol=0.05,
    )

  @parameterized.parameters(
      ([[0, 0], [1, 0], [1, 1]], 1),
      ([[0, 0], [2, 0], [1, 2]], 4),
      ([[1, 0], [2, 3], [0, 2]], 5),
      ([[0, 0], [1, 1], [2, 2]], 0),
      ([[-4, 3], [6, -2], [-4, 3]], 0),
  )
  def test_signed_area(self, points, expected):
    points = np.array(points)
    for perm in _rotations:
      self.assertEqual(_signed_area(*points[perm]), expected)
    for perm in _inversions:
      self.assertEqual(_signed_area(*points[perm]), -expected)

  @parameterized.parameters(
      ((),),
      ((1,),),
      ((1, 2),),
      ((2, 1),),
      ((2, 3),),
      ((1, 2, 3),),
  )
  def test_pareto_frontier_bad_shape(self, shape):
    points = np.zeros(shape, dtype=np.int64)
    with self.assertRaisesRegex(ValueError, 'Expected at least two 2D points'):
      auditing._pareto_frontier(points)

  def test_pareto_frontier_unsorted(self):
    points = np.array([[1, 0], [0, 1]])
    with self.assertRaisesRegex(ValueError, 'Expected points to be sorted'):
      auditing._pareto_frontier(points)

  def test_pareto_frontier_two_points(self):
    points = np.array([[0, 0], [1, 1]])
    frontier = auditing._pareto_frontier(points)
    np.testing.assert_equal(frontier, [0, 1])

  def test_pareto_frontier_linear(self):
    n = 100
    points = np.stack([range(n), range(n)], axis=1)
    frontier = auditing._pareto_frontier(points)
    np.testing.assert_equal(frontier, [0, n - 1])

  def test_pareto_frontier_simple_1(self):
    points = np.array([[0, 0], [0, 2], [3, 2], [3, 5], [5, 5]])
    frontier = auditing._pareto_frontier(points)
    np.testing.assert_equal(frontier, [0, 1, 3, 4])

  def test_pareto_frontier_simple_2(self):
    points = np.array([[0, 0], [0, 2], [2, 2], [2, 3], [3, 3], [3, 5], [5, 5]])
    frontier = auditing._pareto_frontier(points)
    # Should not contain [3, 3], which is dominated by [0, 2] and [3, 5].
    np.testing.assert_equal(frontier, [0, 1, 5, 6])

  def test_pareto_frontier_simple_3(self):
    points = np.array([[0, 0], [0, 2], [1, 2], [1, 4], [3, 4], [3, 5], [5, 5]])
    frontier = auditing._pareto_frontier(points)
    # Should contain [1, 4], which is not dominated by [0, 2] and [3, 5].
    np.testing.assert_equal(frontier, [0, 1, 3, 5, 6])

  def test_pareto_frontier_simple_4(self):
    points = np.array([[0, 0], [0, 2], [1, 2], [1, 3], [2, 3], [2, 4], [4, 4]])
    frontier = auditing._pareto_frontier(points)
    # Should not contain [1, 3], which is a combination of [0, 2] and [2, 4].
    np.testing.assert_equal(frontier, [0, 1, 5, 6])

  @parameterized.named_parameters(
      ('increasing', np.sin, np.pi / 2),
      ('decreasing', np.cos, np.pi / 2),
      ('increasing_and_decreasing', np.sin, np.pi),
  )
  def test_pareto_frontier_convex(self, fn, bound):
    n = 100
    xs = np.linspace(0, bound, n)
    points = np.stack([xs, fn(xs)], axis=1)
    frontier = auditing._pareto_frontier(points)
    # On a convex function, the frontier should be the same as the points.
    np.testing.assert_almost_equal(frontier, np.arange(n))

  @parameterized.named_parameters(
      ('increasing', lambda x: -np.cos(x), np.pi / 2),
      ('decreasing', lambda x: -np.sin(x), np.pi / 2),
      ('decreasing_and_increasing', lambda x: -np.sin(x), np.pi),
  )
  def test_pareto_frontier_concave(self, fn, bound):
    n = 100
    xs = np.linspace(0, bound, n)
    points = np.stack([xs, fn(xs)], axis=1)
    frontier = auditing._pareto_frontier(points)
    # On a concave function, the frontier should be the first and last points.
    np.testing.assert_almost_equal(frontier, [0, n - 1])

  @parameterized.parameters(range(10))
  def test_pareto_frontier_random(self, seed):
    n = 80
    rng = np.random.default_rng(seed=0xBAD5EED + seed)
    xs = np.linspace(0, np.pi, n)
    ys = np.sin(xs) + rng.normal(scale=0.1, size=n)
    points = np.stack([xs, ys], axis=1)
    frontier = auditing._pareto_frontier(points)

    # Compare to simple cubic time algorithm.
    dominated_indices = {
        j
        for i, j, k in itertools.combinations(range(n), 3)
        if _signed_area(points[i], points[j], points[k]) >= 0
    }
    expected_frontier = sorted(set(range(n)) - dominated_indices)

    np.testing.assert_almost_equal(frontier, expected_frontier)

  def test_get_tn_fn_counts(self):
    in_canary_scores = [1.5, 2.5, 0.5, 3.5]
    out_canary_scores = [0, 3, 2, 1]
    thresholds, tn_counts, fn_counts = auditing._get_tn_fn_counts(
        in_canary_scores, out_canary_scores
    )
    np.testing.assert_equal(thresholds, [0, 0.5, 3.5, np.inf])
    np.testing.assert_equal(tn_counts, [0, 1, 4, 4])
    np.testing.assert_equal(fn_counts, [0, 0, 3, 4])

  def test_get_tn_fn_counts_ties_convex(self):
    in_canary_scores = [0, 1, 1, 2]
    out_canary_scores = [0, 0, 1, 1]
    thresholds, tn_counts, fn_counts = auditing._get_tn_fn_counts(
        in_canary_scores, out_canary_scores
    )
    np.testing.assert_equal(thresholds, [0, 1, 2, np.inf])
    np.testing.assert_equal(tn_counts, [0, 2, 4, 4])
    np.testing.assert_equal(fn_counts, [0, 1, 3, 4])

  def test_get_tn_fn_counts_ties_nonconvex(self):
    in_canary_scores = [0, 0, 1, 2]
    out_canary_scores = [0, 1, 1, 1]
    thresholds, tn_counts, fn_counts = auditing._get_tn_fn_counts(
        in_canary_scores, out_canary_scores
    )
    np.testing.assert_equal(thresholds, [0, 2, np.inf])
    np.testing.assert_equal(tn_counts, [0, 4, 4])
    np.testing.assert_equal(fn_counts, [0, 3, 4])

  def test_get_tn_fn_counts_zeros(self):
    in_canary_scores = np.zeros(4)
    out_canary_scores = np.zeros(4)
    thresholds, tn_counts, fn_counts = auditing._get_tn_fn_counts(
        in_canary_scores, out_canary_scores
    )
    np.testing.assert_equal(thresholds, [0, np.inf])
    np.testing.assert_equal(tn_counts, [0, 4])
    np.testing.assert_equal(fn_counts, [0, 4])

  @parameterized.product(
      n_in=(100, 1000),
      out_samples_ratio=(0.5, 1.0, 1.5),
      thresh=(0, 0.5, 1.0),
  )
  def test_epsilon_clopper_pearson_explicit(
      self, n_in, out_samples_ratio, thresh
  ):
    in_canary_scores = _deterministic_normal(1, 1, n_in)
    n_out = int(n_in * out_samples_ratio)
    out_canary_scores = _deterministic_normal(0, 1, n_out)
    significance = 0.1

    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    strategy = auditing.Explicit(thresh)
    eps = auditor.epsilon_clopper_pearson(
        significance, threshold_strategy=strategy
    )

    fn = np.sum(in_canary_scores < thresh)
    fp = np.sum(out_canary_scores >= thresh)
    tpr_lb = 1 - auditing._clopper_pearson_upper(fn, n_in, significance / 2)
    fpr_ub = auditing._clopper_pearson_upper(fp, n_out, significance / 2)
    expected_eps = max(0, np.log(tpr_lb / fpr_ub))
    np.testing.assert_allclose(eps, expected_eps)

  @parameterized.product(
      one_sided=(True, False),
      mu=(0.3, 1.0, 1.5),
      out_samples_ratio=(0.5, 1.0, 1.5),
      threshold_strategy=(auditing.Bonferroni(), auditing.Split(seed=0)),
  )
  def test_epsilon_clopper_pearson_tight(
      self, one_sided, mu, out_samples_ratio, threshold_strategy
  ):
    significance = 0.2
    delta = 0.1
    in_samples = 2_000_000
    out_samples = int(in_samples * out_samples_ratio)
    in_canary_scores = _deterministic_normal(mu, 1, in_samples)
    out_canary_scores = _deterministic_normal(0, 1, out_samples)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    eps_lb = auditor.epsilon_clopper_pearson(
        significance, delta, one_sided, threshold_strategy=threshold_strategy
    )
    true_eps = dp_accounting.get_epsilon_gaussian(1 / mu, delta)
    np.testing.assert_array_less(eps_lb, true_eps)
    np.testing.assert_allclose(eps_lb, true_eps, rtol=0.2)

  @parameterized.product(
      out_samples_ratio=(0.5, 1.0, 1.5),
      mu=(0.7, 1.0, 1.5),
  )
  def test_epsilon_clopper_pearson_tight_multi_split(
      self, out_samples_ratio, mu
  ):
    significance = 0.1
    delta = 0.1
    in_samples = 100_000
    threshold_strategy = auditing.MultiSplit(seed=0)
    out_samples = int(in_samples * out_samples_ratio)
    in_canary_scores = _deterministic_normal(mu, 1, in_samples)
    out_canary_scores = _deterministic_normal(0, 1, out_samples)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    eps_lb = auditor.epsilon_clopper_pearson(
        significance, delta, threshold_strategy=threshold_strategy
    )
    true_eps = dp_accounting.get_epsilon_gaussian(1 / mu, delta)
    np.testing.assert_array_less(eps_lb, true_eps)
    np.testing.assert_allclose(eps_lb, true_eps, rtol=0.2)

  @parameterized.product(
      min_count=(1, 50),
      out_samples_ratio=(0.5, 1.0, 1.5),
  )
  def test_epsilon_raw_counts_helper_accurate_large_delta(
      self, min_count, out_samples_ratio
  ):
    delta = 1e-2
    in_samples = 200_000
    out_samples = int(in_samples * out_samples_ratio)
    in_canary_scores = _deterministic_normal(1, 1, in_samples)
    out_canary_scores = _deterministic_normal(0, 1, out_samples)
    _, tn_counts, fn_counts = auditing._get_tn_fn_counts(
        in_canary_scores, out_canary_scores
    )
    tp_counts = (fn_counts[-1] - fn_counts)[::-1]
    fp_counts = (tn_counts[-1] - tn_counts)[::-1]
    eps = auditing._epsilon_raw_counts_helper(
        tp_counts, fp_counts, min_count, delta
    )
    true_eps = dp_accounting.get_epsilon_gaussian(1.0, delta)
    np.testing.assert_allclose(eps, true_eps, rtol=1e-1)

  @parameterized.product(
      min_count=(1, 3),
      n_neg=(10, 13),
      n_pos=(9, 12),
  )
  def test_epsilon_raw_counts_helper_worst_case(self, min_count, n_neg, n_pos):
    # Tests that the epsilon uses the minimum allowed FPR for a perfect
    # classifier.
    tp_counts = np.array([0, n_pos, n_pos])
    fp_counts = np.array([0, 0, n_neg])
    epsilon = auditing._epsilon_raw_counts_helper(
        tp_counts, fp_counts, min_count, delta=0
    )
    np.testing.assert_allclose(epsilon, np.log(n_neg / min_count))

  @parameterized.product(
      min_count=[1, 2],
      delta=[0, 1 / 8, 1 / 4, 3 / 8, 1 / 2],
  )
  def test_epsilon_raw_counts_helper_nonzero_delta(self, min_count, delta):
    tp_counts = np.array([0, 2, 6, 8])
    fp_counts = np.array([0, 0, 4, 8])
    epsilon = auditing._epsilon_raw_counts_helper(
        tp_counts, fp_counts, min_count, delta
    )
    if min_count == 0:
      expected_slopes = {0: np.inf, 1 / 8: np.inf}
    elif min_count == 1:
      expected_slopes = {0: 3, 1 / 8: 2}
    else:  # min_count == 2
      expected_slopes = {0: 2, 1 / 8: 3 / 2}
    expected_slopes.update({1 / 4: 1, 3 / 8: 3 / 4, 1 / 2: 1 / 2})
    np.testing.assert_allclose(epsilon, max(np.log(expected_slopes[delta]), 0))

  @parameterized.product(
      one_sided=(True, False),
      delta=(0, 0.1),
  )
  def test_epsilon_raw_counts_zero_if_no_min_counts(self, one_sided, delta):
    in_canary_scores = np.arange(10) + 0.5
    out_canary_scores = np.arange(10)
    min_count = 12
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    eps = auditor.epsilon_raw_counts(min_count, delta, one_sided)
    self.assertEqual(eps, 0.0)

  @parameterized.product(
      out_samples_ratio=(0.5, 1.0, 1.5),
      vectorized=(True, False),
  )
  def test_tpr_at_given_fpr_worst_case(self, out_samples_ratio, vectorized):
    """Test that TPR equals FPR for a worst-case dataset."""
    in_samples = 30
    out_samples = int(in_samples * out_samples_ratio)
    in_canary_scores = np.full(in_samples, 0.0)
    out_canary_scores = np.full(out_samples, 1.0)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    fprs = np.linspace(0, 1, 10)
    if vectorized:
      tprs = auditor.tpr_at_given_fpr(fprs)
    else:
      tprs = [auditor.tpr_at_given_fpr(fpr) for fpr in fprs]
    np.testing.assert_allclose(tprs, fprs)

  @parameterized.product(
      out_samples_ratio=(0.5, 1.0, 1.5),
      vectorized=(True, False),
  )
  def test_tpr_at_given_fpr_circular_roc(self, out_samples_ratio, vectorized):
    n_in = 10_000
    n_out = int(n_in * out_samples_ratio)
    in_canary_scores, out_canary_scores = _scores_with_circular_roc(n_in, n_out)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    fprs = np.linspace(0, 1, 10)
    if vectorized:
      tprs = auditor.tpr_at_given_fpr(fprs)
    else:
      tprs = [auditor.tpr_at_given_fpr(fpr) for fpr in fprs]
    expected_tprs = np.sqrt(fprs * (2 - fprs))
    np.testing.assert_allclose(tprs, expected_tprs, rtol=1e-3)

  @parameterized.product(
      out_samples_ratio=(0.5, 1.0, 1.5),
      vectorized=(True, False),
  )
  def test_tpr_at_given_fpr_best_case(self, out_samples_ratio, vectorized):
    """Test that TPR at FPR is 1.0 for a best-case dataset."""
    in_samples = 30
    out_samples = int(in_samples * out_samples_ratio)
    in_canary_scores = np.full(in_samples, 1.0)
    out_canary_scores = np.full(out_samples, 0.0)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    fprs = np.linspace(0, 1, 10)
    if vectorized:
      tprs = auditor.tpr_at_given_fpr(fprs)
    else:
      tprs = [auditor.tpr_at_given_fpr(fpr) for fpr in fprs]
    np.testing.assert_equal(tprs, np.ones_like(fprs))

  def test_tpr_at_given_fpr_simple(self):
    in_canary_scores = [1, 3]
    out_canary_scores = [0, 2]
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    fprs = [0, 0.25, 0.5, 0.75, 1.0]
    tprs = auditor.tpr_at_given_fpr(fprs)
    expected_tprs = [0.5, 0.75, 1.0, 1.0, 1.0]
    np.testing.assert_allclose(tprs, expected_tprs)

  @parameterized.product(
      out_samples_ratio=(0.5, 1.0, 1.5),
  )
  def test_tpr_at_given_fpr_random(self, out_samples_ratio):
    rng = np.random.default_rng(seed=0xBAD5EED)
    in_samples = 100_000
    out_samples = int(in_samples * out_samples_ratio)
    in_canary_scores = rng.uniform(0, 1, in_samples)
    out_canary_scores = rng.uniform(0, 1, out_samples)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    fprs = np.linspace(0, 1, 10)
    tprs = auditor.tpr_at_given_fpr(fprs)
    np.testing.assert_allclose(tprs, fprs, rtol=0.05, atol=1e-3)

  @parameterized.product(
      out_samples_ratio=(0.5, 1.0, 1.5),
  )
  def test_attack_auroc_worst_case(self, out_samples_ratio):
    """Test that auroc is 0.5 for a worst-case dataset."""
    in_samples = 30
    out_samples = int(in_samples * out_samples_ratio)
    in_canary_scores = np.full(in_samples, 0.0)
    out_canary_scores = np.full(out_samples, 1.0)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    self.assertAlmostEqual(auditor.attack_auroc(), 0.5)

  @parameterized.product(
      out_samples_ratio=(0.5, 1.0, 1.5),
  )
  def test_attack_auroc_best_case(self, out_samples_ratio):
    """Test that auroc is 1.0 for a best-case dataset."""
    in_samples = 30
    out_samples = int(in_samples * out_samples_ratio)
    in_canary_scores = np.full(in_samples, 1.0)
    out_canary_scores = np.full(out_samples, 0.0)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    self.assertAlmostEqual(auditor.attack_auroc(), 1.0)

  def test_attack_auroc_simple(self):
    in_canary_scores = [0, 4, 6, 7]
    out_canary_scores = [1, 2, 3, 5]

    # All FP/TP points: (0,0) (0,1) (0,2) (1,2) (1,3) (2,3) (2,4) (3,4) (4,4)
    # FP/TP frontier points: (0,0) (0,2) (1,3) (4,4)
    # Area using trapezoids: (0 + 2.5 + 10.5) / 16 = 13 / 16

    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    self.assertAlmostEqual(auditor.attack_auroc(), 13 / 16)

  @parameterized.product(
      in_samples=(10_000, 100_000),
      out_samples_ratio=(0.5, 1.0, 1.5),
  )
  def test_attack_auroc_random(self, in_samples, out_samples_ratio):
    rng = np.random.default_rng(seed=0xBAD5EED)
    out_samples = int(in_samples * out_samples_ratio)
    in_canary_scores = rng.uniform(0, 1, in_samples)
    out_canary_scores = rng.uniform(0, 1, out_samples)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    np.testing.assert_allclose(auditor.attack_auroc(), 0.5, rtol=0.05)

  @parameterized.product(
      out_samples_ratio=(0.5, 1.0, 1.5),
  )
  def test_attack_auroc_circular_roc(self, out_samples_ratio):
    n_in = 10_000
    n_out = int(n_in * out_samples_ratio)
    in_canary_scores, out_canary_scores = _scores_with_circular_roc(n_in, n_out)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    auroc = auditor.attack_auroc()
    self.assertAlmostEqual(auroc, np.pi / 4, places=3)

  @parameterized.product(
      prevalence=(None, 0.0, 0.1, 0.5, 0.7, 1.0),
      out_samples_ratio=(0.5, 1.0, 1.5),
      significance=(None, 0.05),
  )
  def test_max_accuracy_simple(
      self, prevalence, out_samples_ratio, significance
  ):
    # Use many canaries with the same score so the upper bound is tight.
    n_in = 1000
    n_out = int(n_in * out_samples_ratio)
    in_canary_scores = [1] * (n_in // 2) + [3] * (n_in // 2)
    out_canary_scores = [0] * (n_out // 2) + [2] * (n_out // 2)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    max_accuracy = auditor.max_accuracy(
        prevalence=prevalence, significance=significance
    )
    if prevalence is None:
      prevalence = n_in / (n_in + n_out)
    # Accuracy for a threshold between 0 and 1 (TPR=1, TNR=0.5)
    acc1 = prevalence + 0.5 * (1 - prevalence)
    # Accuracy for a threshold between 2 and 3 (TPR=0.5, TNR=1)
    acc2 = 0.5 * prevalence + (1 - prevalence)
    expected_max_accuracy = max(acc1, acc2)
    if significance is None:
      self.assertAlmostEqual(max_accuracy, expected_max_accuracy)
    else:
      np.testing.assert_allclose(max_accuracy, expected_max_accuracy, rtol=0.05)
      self.assertGreaterEqual(max_accuracy, expected_max_accuracy)

  @parameterized.product(
      prevalence=(0.0, 0.1, 0.5, 0.7, 1.0),
      significance=(None, 0.05),
      out_samples_ratio=(0.5, 1.0, 1.5),
  )
  def test_max_accuracy_circular_roc(
      self, prevalence, significance, out_samples_ratio
  ):
    n_in = 10_000
    n_out = int(n_in * out_samples_ratio)
    in_canary_scores, out_canary_scores = _scores_with_circular_roc(n_in, n_out)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    max_accuracy = auditor.max_accuracy(
        prevalence=prevalence, significance=significance
    )
    expected_max_accuracy = np.sqrt(prevalence**2 + (1 - prevalence) ** 2)

    if significance is None:
      self.assertAlmostEqual(max_accuracy, expected_max_accuracy, places=3)
    else:
      np.testing.assert_allclose(max_accuracy, expected_max_accuracy, rtol=0.05)
      self.assertGreaterEqual(max_accuracy, expected_max_accuracy)

  @parameterized.product(
      mu=(0.1, 0.3, 1.0, 3.0),
      out_samples_ratio=(0.5, 1.0, 1.5),
  )
  def test_epsilon_from_gdp_tight(self, mu, out_samples_ratio):
    rng = np.random.default_rng(seed=0xBAD5EED)
    significance = 0.1
    delta = 5e-2
    in_samples = 500_000
    out_samples = int(in_samples * out_samples_ratio)
    in_canary_scores = rng.normal(mu, 1, in_samples)
    out_canary_scores = rng.normal(0, 1, out_samples)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    eps = auditor.epsilon_from_gdp(significance, delta)
    true_eps = dp_accounting.get_epsilon_gaussian(1 / mu, delta)
    np.testing.assert_allclose(eps, true_eps, rtol=0.05)

  @parameterized.parameters(0.0, -1.0, -3.0)
  def test_epsilon_from_gdp_non_positive_mu_is_zero(self, mu):
    rng = np.random.default_rng(seed=0xBAD5EED)
    significance = 0.05
    delta = 1e-5
    m = 5000
    in_canary_scores = rng.normal(mu, 1, m)
    out_canary_scores = rng.normal(0, 1, m)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    eps = auditor.epsilon_from_gdp(significance, delta)
    self.assertEqual(eps, 0)

  @parameterized.product(
      quantiles=((0.025,), (0.975,), (0.025, 0.975), (0.025, 0.5, 0.975)),
      bootstrap_type=('quantile', 'bias_correction', 'acceleration'),
  )
  def test_bootstrap(self, quantiles, bootstrap_type):
    n = 3000

    # Compute interval for mean of scores, which we can also get exactly with
    # the central limit theorem. Use any strange distribution for the data.
    in_canary_scores = _deterministic_normal(np.e, np.pi, n // 2)
    out_canary_scores = np.linspace(0, 1, n // 2)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)

    def mean_score(a: auditing.CanaryScoreAuditor):
      return np.mean(
          np.concatenate([a._in_canary_scores, a._out_canary_scores])
      )

    bootstrap_params = auditing.BootstrapParams(
        quantiles=quantiles,
        bias_correction=(bootstrap_type != 'quantile'),
        acceleration=(bootstrap_type == 'acceleration'),
        seed=0xBAD5EED,
    )
    interval = auditor._bootstrap(mean_score, bootstrap_params)
    all_scores = np.concatenate([in_canary_scores, out_canary_scores])
    mu_hat, sigma_hat = np.mean(all_scores), np.std(all_scores)
    expected_interval = auditing._norm.ppf(
        q=quantiles,
        loc=mu_hat,
        scale=sigma_hat / np.sqrt(n),
    )
    np.testing.assert_allclose(interval, expected_interval, rtol=0.01)

  def test_tpr_at_given_fpr_bootstrap_raises_non_scalar_fpr(self):
    auditor = auditing.CanaryScoreAuditor(
        in_canary_scores=[1, 2, 3], out_canary_scores=[0, 1, 2]
    )
    bootstrap_params = auditing.BootstrapParams()
    fpr = [0.1, 0.2]
    with self.assertRaisesRegex(
        ValueError, 'fpr must be a scalar for bootstrap'
    ):
      auditor.tpr_at_given_fpr(fpr, bootstrap_params=bootstrap_params)

  @parameterized.product(
      metric_and_args=(
          ('epsilon_raw_counts', ()),
          ('tpr_at_given_fpr', (0.1,)),
          ('attack_auroc', ()),
      ),
      bootstrap_type=('quantile', 'bias_correction', 'acceleration'),
      mu=(0, 1, 10),
  )
  def test_auditing_metric_bootstrap(self, metric_and_args, bootstrap_type, mu):
    # Test that bootstrapped metrics run and return basically reasonable values.
    metric_fn_name, args = metric_and_args
    rng = np.random.default_rng(seed=0xBAD5EED)
    in_canary_scores = rng.normal(mu, size=356)
    out_canary_scores = rng.normal(size=432)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    bootstrap_params = auditing.BootstrapParams(
        bias_correction=(bootstrap_type != 'quantile'),
        acceleration=(bootstrap_type == 'acceleration'),
        seed=0xBAD5EED,
    )
    metric_fn = getattr(auditor, metric_fn_name)
    value = metric_fn(*args)
    interval = metric_fn(*args, bootstrap_params=bootstrap_params)
    self.assertLen(interval, 2)
    self.assertBetween(value, *interval)

  @parameterized.named_parameters(
      ('significance_zero', {'significance': 0.0}, 'significance'),
      ('significance_gt_one', {'significance': 1.1}, 'significance'),
      ('delta_negative', {'delta': -0.1}, 'delta'),
      ('delta_gt_one', {'delta': 1.1}, 'delta'),
      ('one_sided_false', {'one_sided': False}, 'one_sided must be True'),
  )
  def test_epsilon_one_run_raises_invalid_args(
      self, override_args, error_regex
  ):
    args = {'significance': 0.1, 'delta': 0.1, 'one_sided': True}
    args.update(override_args)
    auditor = auditing.CanaryScoreAuditor(
        in_canary_scores=[1, 2, 3], out_canary_scores=[0, 1, 2]
    )
    with self.assertRaisesRegex(ValueError, error_regex):
      auditor.epsilon_one_run(**args)

  @parameterized.product(
      m=(10, 100, 1000),
      n_guess=(1, 10, 100),
      n_wrong=(0, 1, 10, 100),
      eps=(0, 0.5, 1.0),
      delta=(0, 1e-6, 1e-3, 1e-1, 1),
  )
  def test_one_run_p_value(self, m, n_guess, n_wrong, eps, delta):
    if not n_wrong <= n_guess <= m:
      return
    n_correct = n_guess - n_wrong
    p = auditing._one_run_p_value(m, n_guess, n_correct, eps, delta)
    naive_p = _one_run_p_value_naive(m, n_guess, n_correct, eps, delta)
    self.assertAlmostEqual(p, naive_p)

  @parameterized.named_parameters(
      ('zero', False, 0, 0),
      ('zero_fdp', True, 0, 0),
      ('one', False, 1, 1.13346),
      ('one_fdp', True, 1, 2.164749),
      ('two', False, 2, 2.311945),
      ('two_fdp', True, 2, 4.7218075),
      ('four', False, 4, 4.395568),
      ('four_fdp', True, 4, 9.133752),
  )
  def test_epsilon_one_run_close(self, use_fdp, shift, expected_eps):
    method = 'epsilon_one_run_fdp' if use_fdp else 'epsilon_one_run'
    n = 10_000
    out_canary_scores = _deterministic_normal(0, 1, n)
    in_canary_scores = _deterministic_normal(shift, 1, n)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    significance = 0.05
    delta = 1e-6
    eps = getattr(auditor, method)(significance, delta)
    np.testing.assert_allclose(eps, expected_eps, rtol=1e-5)

  def test_auditor_raises_empty_scores(self):
    scores = []
    other_scores = [1, 2, 3]
    with self.assertRaisesRegex(ValueError, 'must be non-empty'):
      auditing.CanaryScoreAuditor(scores, other_scores)
    with self.assertRaisesRegex(ValueError, 'must be non-empty'):
      auditing.CanaryScoreAuditor(other_scores, scores)

  @parameterized.named_parameters(
      ('dim_0', 3),
      ('1x1', [[1]]),
      ('2x2', [[1, 2], [3, 4]]),
  )
  def test_auditor_raises_wrong_dim_scores(self, scores):
    other_scores = [1, 2, 3]
    with self.assertRaisesRegex(ValueError, 'must be 1-dimensional'):
      auditing.CanaryScoreAuditor(scores, other_scores)
    with self.assertRaisesRegex(ValueError, 'must be 1-dimensional'):
      auditing.CanaryScoreAuditor(other_scores, scores)

  def test_auditor_accepts_jax_array(self):
    in_canary_scores = jnp.ones((3,))
    out_canary_scores = jnp.zeros((5,))
    auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)

  def test_num_canaries_property(self):
    auditor = auditing.CanaryScoreAuditor(
        [1.0, 2.0, 3.0], [-1.0, 0.0, 0.5, 1.5]
    )
    self.assertEqual(auditor.num_canaries, (3, 4))

  def test_scalar_methods_return_python_float(self):
    auditor = auditing.CanaryScoreAuditor(np.arange(100) + 1.5, np.arange(100))
    self.assertIsInstance(auditor.epsilon_clopper_pearson(0.05, 0.01), float)
    self.assertIsInstance(
        auditor.epsilon_clopper_pearson(
            0.05, 0.01, threshold_strategy=auditing.MultiSplit(seed=0)
        ),
        float,
    )
    self.assertIsInstance(auditor.epsilon_raw_counts(min_count=5), float)
    self.assertIsInstance(auditor.tpr_at_given_fpr(0.1), float)
    self.assertIsInstance(auditor.attack_auroc(), float)
    self.assertIsInstance(auditor.max_accuracy(), float)
    self.assertIsInstance(auditor.max_accuracy(significance=0.05), float)
    self.assertIsInstance(auditor.epsilon_from_gdp(0.05, 0.01), float)
    self.assertIsInstance(auditor.epsilon_one_run(0.05, 0.01), float)
    self.assertIsInstance(auditor.epsilon_one_run_fdp(0.05, 0.01), float)


if __name__ == '__main__':
  absltest.main()

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

from collections.abc import Sequence
import itertools

from absl.testing import absltest
from absl.testing import parameterized
from jax_privacy import batch_selection
from jax_privacy.experimental.monte_carlo import sample_generation
import numpy as np
import scipy

# Reused test case inputs for b-min-sep sampling.
_DP_SGD_MODES = [
    np.array([1, 0, 1]),
    np.array([1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, 0, 1]),
    np.array([0, 0, 0]),
]
_SUM_OF_TWO_DP_SGD_MODES = [
    np.array([2, 0, 2]),
    np.array([2, 0, 1]),
    np.array([2, 0, 0]),
    np.array([1, 1, 1]),
    np.array([1, 1, 0]),
    np.array([1, 0, 2]),
    np.array([1, 0, 1]),
    np.array([0, 2, 0]),
    np.array([0, 1, 1]),
    np.array([0, 1, 0]),
    np.array([0, 0, 2]),
    np.array([0, 0, 1]),
    np.array([0, 0, 0]),
]
_BANDMF_MODES = [
    np.array([1.0, 0.5, 1.0]),
    np.array([1.0, 0.5, 0.0]),
    np.array([0.0, 1.0, 0.5]),
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 0.0, 0.0]),
]
_BANDMF_TRUNCATED_MODES = [
    np.array([2.0, 1.0, 2.0]),
    np.array([2.0, 1.0, 1.0]),
    np.array([2.0, 1.0, 0.0]),
    np.array([1.0, 0.5, 2.0]),
    np.array([1.0, 0.5, 1.0]),
    np.array([1.0, 0.5, 0.0]),
    np.array([0.0, 2.0, 1.0]),
    np.array([0.0, 1.0, 0.5]),
    np.array([0.0, 0.0, 2.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 0.0, 0.0]),
]
_COLD_START_DISTRIBUTION = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 8, 1 / 8])
_WARM_START_DISTRIBUTION = np.array([1 / 6, 1 / 6, 1 / 3, 1 / 6, 1 / 6])


def _assert_mode_frequencies(
    samples: np.ndarray,
    modes: Sequence[np.ndarray],
    distribution: np.ndarray,
) -> None:
  """Asserts the frequency of each mode among samples is within 6 sigma."""
  num_samples = samples.shape[1]
  # matches[i, j] is True if sample i equals mode j.
  matches = np.isclose(
      samples.T[:, None, :], np.asarray(modes)[None, :, :], atol=1e-6
  ).all(axis=-1)
  differences = np.abs(matches.sum(axis=0) - num_samples * distribution)
  stdev = np.sqrt(num_samples * distribution * (1 - distribution))
  assert np.all(differences <= 6 * stdev), (differences, stdev)


def _brute_force_b_min_sep_privacy_loss(
    strategy: batch_selection.BMinSepSampling,
    samples: np.ndarray,
    noise_multiplier: float,
    c_col: np.ndarray,
) -> np.ndarray:
  """Privacy loss by enumerating all participation patterns (no truncation)."""
  b, p, n = strategy.min_sep, strategy.sampling_prob, strategy.iterations
  c_matrix = scipy.linalg.toeplitz(
      np.concatenate([c_col, np.zeros(n - c_col.size)]), np.zeros(n)
  )
  # Probability that the last participation before round 0 was at round
  # -b + i, for i in range(b). Without warm start, it is always at -b.
  if strategy.warm_start:
    initial_probs = np.full(b, p / (1 + (b - 1) * p))
    initial_probs[0] = 1 / (1 + (b - 1) * p)
  else:
    initial_probs = np.array([1.0])
  log_terms = []
  for initial_offset, initial_prob in enumerate(initial_probs):
    for x in itertools.product([0, 1], repeat=n):
      last_participation = initial_offset - b
      log_prob = np.log(initial_prob)
      for i, x_i in enumerate(x):
        if i - last_participation < b:
          if x_i:
            log_prob = -np.inf
            break
          continue
        log_prob += np.log(p) if x_i else np.log1p(-p)
        if x_i:
          last_participation = i
      if log_prob == -np.inf:
        continue
      mode = c_matrix @ np.asarray(x, dtype=float)
      llrs = (2 * mode @ samples - mode @ mode) / (2 * noise_multiplier**2)
      log_terms.append(log_prob + llrs)
  return scipy.special.logsumexp(np.array(log_terms), axis=0)


class SampleGenerationTest(parameterized.TestCase):

  @parameterized.product(
      iterations=[1, 4, 7],
      kernel_size=[0, 1, 3, 4, 9],
      num_shifts=[None, 1, 3, 7, 11],
  )
  def test_sliding_dot_products_and_squared_norms(
      self, iterations, kernel_size, num_shifts
  ):
    rng = np.random.default_rng(0xBAD5EED)
    samples = rng.normal(size=(iterations, 3))
    kernel = rng.normal(size=kernel_size)
    dot_products, squared_norms = (
        sample_generation._sliding_dot_products_and_squared_norms(
            samples, kernel, num_shifts
        )
    )
    # Brute force: k_i is the kernel shifted down by i rounds and cut off at
    # the last round.
    if num_shifts is None:
      num_shifts = iterations
    shifted_kernels = np.zeros((num_shifts, iterations))
    for i in range(num_shifts):
      size = max(min(kernel_size, iterations - i), 0)
      shifted_kernels[i, i : i + size] = kernel[:size]
    np.testing.assert_allclose(
        dot_products, shifted_kernels @ samples, atol=1e-10
    )
    np.testing.assert_allclose(
        squared_norms, (shifted_kernels**2).sum(axis=1), atol=1e-10
    )

  @parameterized.parameters(
      (True, np.array([1.0]), np.array([1.0, 0.0, 1.0, 0.0])),
      (True, np.array([1.0, 0.5]), np.array([1.0, 0.5, 1.0, 0.5])),
      (False, np.array([1.0, 0.5]), np.array([0.0, 0.0, 0.0, 0.0])),
      (
          True,
          np.array([1.0, 0.5, 0.25, 0.125]),
          np.array([1.0, 0.5, 1.25, 0.625]),
      ),
      (
          False,
          np.array([1.0, 0.5, 0.25, 0.125]),
          np.array([0.0, 0.0, 0.0, 0.0]),
      ),
  )
  def test_generate_balls_in_bins_sample_low_noise(
      self,
      positive_sample,
      c_col,
      first_mode,
  ):
    sampling_scheme = batch_selection.BallsInBinsSampling(
        cycle_length=2, iterations=4
    )
    noise_multiplier = 1e-9
    # The distribution of samples should be evenly divided between the first
    # mode and the second mode, which is just the first mode shifted by 1
    # position. For positive_sample=False, both are zero.
    second_mode = np.zeros_like(first_mode)
    second_mode[1:] = first_mode[:-1]
    samples = sample_generation.generate_sample(
        sampling_scheme,
        noise_multiplier,
        c_col,
        positive_sample=positive_sample,
        num_samples=10000,
    )
    if positive_sample:
      _assert_mode_frequencies(
          samples, [first_mode, second_mode], np.array([0.5, 0.5])
      )
    else:
      _assert_mode_frequencies(samples, [first_mode], np.array([1.0]))

  @parameterized.parameters(2.0, -0.5)
  def test_generate_balls_in_bins_sample_low_noise_with_mode_scale(
      self, mode_scale
  ):
    sampling_scheme = batch_selection.BallsInBinsSampling(
        cycle_length=2, iterations=4
    )
    samples = sample_generation.generate_sample(
        sampling_scheme,
        noise_multiplier=1e-9,
        c_col=np.array([1.0, 0.5]),
        num_samples=10000,
        mode_scale=mode_scale,
    )
    modes = mode_scale * np.array([[1.0, 0.5, 1.0, 0.5], [0.0, 1.0, 0.5, 1.0]])
    _assert_mode_frequencies(samples, modes, np.array([0.5, 0.5]))

  def test_generate_sample_mode_scale_overrides_positive_sample(self):
    sampling_scheme = batch_selection.BallsInBinsSampling(
        cycle_length=2, iterations=4
    )
    samples = sample_generation.generate_sample(
        sampling_scheme,
        noise_multiplier=1e-9,
        c_col=np.array([1.0, 0.5]),
        positive_sample=False,
        num_samples=10000,
        mode_scale=-0.5,
    )
    modes = -0.5 * np.array([[1.0, 0.5, 1.0, 0.5], [0.0, 1.0, 0.5, 1.0]])
    _assert_mode_frequencies(samples, modes, np.array([0.5, 0.5]))

  def test_generate_sample_unit_mode_scale_matches_default(self):
    sampling_scheme = batch_selection.BallsInBinsSampling(
        cycle_length=2, iterations=4
    )
    kwargs = dict(
        noise_multiplier=1.0, c_col=np.array([1.0, 0.5]), num_samples=10
    )
    np.testing.assert_array_equal(
        sample_generation.generate_sample(
            sampling_scheme, seed=0xBAD5EED, mode_scale=1.0, **kwargs
        ),
        sample_generation.generate_sample(
            sampling_scheme, seed=0xBAD5EED, **kwargs
        ),
    )

  @parameterized.parameters(
      batch_selection.BMinSepSampling(
          min_sep=2, sampling_prob=0.5, iterations=4
      ),
      batch_selection.CyclicPoissonSampling(
          cycle_length=2, sampling_prob=0.5, iterations=4
      ),
  )
  def test_generate_sample_raises_for_mode_scale(self, strategy):
    with self.assertRaises(ValueError):
      sample_generation.generate_sample(
          strategy, 1.0, np.array([1.0]), mode_scale=2.0
      )

  @parameterized.parameters(
      (
          True,
          np.array([1.0]),
          0.5,
          False,
          _DP_SGD_MODES,
          _COLD_START_DISTRIBUTION,
      ),
      (
          True,
          np.array([1.0, 0.5]),
          0.5,
          False,
          _BANDMF_MODES,
          _COLD_START_DISTRIBUTION,
      ),
      (
          True,
          np.array([1.0, 0.5]),
          0.25,
          False,
          _BANDMF_MODES,
          np.array([1 / 16, 3 / 16, 3 / 16, 9 / 64, 27 / 64]),
      ),
      (
          True,
          np.array([1.0, 0.5]),
          0.5,
          True,
          _BANDMF_MODES,
          _WARM_START_DISTRIBUTION,
      ),
      (
          False,
          np.array([1.0]),
          0.5,
          False,
          [np.array([0.0, 0.0, 0.0])],
          np.array([1.0]),
      ),
  )
  def test_generate_b_min_sep_sample_low_noise(
      self,
      positive_sample,
      c_col,
      sampling_prob,
      warm_start,
      modes,
      mode_distribution,
  ):
    sampling_scheme = batch_selection.BMinSepSampling(
        sampling_prob=sampling_prob,
        min_sep=2,
        iterations=3,
        warm_start=warm_start,
    )
    samples = sample_generation.generate_sample(
        sampling_scheme,
        noise_multiplier=1e-9,
        c_col=c_col,
        positive_sample=positive_sample,
        num_samples=10000,
    )
    _assert_mode_frequencies(samples, modes, mode_distribution)

  @parameterized.parameters(
      # Test no warm-start, positive case, dataset size = truncated batch
      # size. Should be same as no truncation.
      (
          True,
          False,
          _BANDMF_MODES,
          _COLD_START_DISTRIBUTION,
          _DP_SGD_MODES,
          _COLD_START_DISTRIBUTION,
          2,
          2,
      ),
      # Test no warm-start, positive case.
      (
          True,
          False,
          _BANDMF_TRUNCATED_MODES,
          np.array([x / 128 for x in [2, 4, 10, 2, 12, 18, 4, 24, 5, 14, 33]]),
          _DP_SGD_MODES,
          _COLD_START_DISTRIBUTION,
          2,
          1,
      ),
      # Test no warm-start, positive case, larger dataset.
      (
          True,
          False,
          _BANDMF_TRUNCATED_MODES,
          np.array([
              x / (9 * 2**9)
              for x in [32, 144, 208, 60, 774, 894, 48, 1080, 70, 567, 731]
          ]),
          _SUM_OF_TWO_DP_SGD_MODES,
          np.array([x / 64 for x in [4, 8, 4, 8, 8, 4, 8, 4, 4, 4, 1, 2, 1]]),
          3,
          2,
      ),
      # Test no warm-start, positive case, dataset size - truncated batch
      # size > 1.
      (
          True,
          False,
          _BANDMF_TRUNCATED_MODES,
          np.array([
              x / (9 * 2**9)
              for x in [
                  116,
                  132,
                  520,
                  60,
                  162,
                  354,
                  240,
                  648,
                  310,
                  381,
                  1685,
              ]
          ]),
          _SUM_OF_TWO_DP_SGD_MODES,
          np.array([x / 64 for x in [4, 8, 4, 8, 8, 4, 8, 4, 4, 4, 1, 2, 1]]),
          3,
          1,
      ),
      # Test yes warm-start, positive case.
      (
          True,
          True,
          _BANDMF_TRUNCATED_MODES,
          np.array([x / 144 for x in [1, 2, 5, 2, 12, 18, 8, 32, 5, 18, 41]]),
          _DP_SGD_MODES,
          _WARM_START_DISTRIBUTION,
          2,
          1,
      ),
      # Test no warm-start, negative case.
      (
          False,
          False,
          [-x for x in _BANDMF_MODES],
          np.array([x / 128 for x in [2, 14, 4, 7, 101]]),
          _DP_SGD_MODES,
          _COLD_START_DISTRIBUTION,
          2,
          1,
      ),
  )
  def test_generate_b_min_sep_sample_low_noise_with_truncation(
      self,
      positive_sample,
      warm_start,
      modes,
      mode_distribution,
      rbs_modes,
      rbs_distribution,
      dataset_size,
      truncated_batch_size,
  ):
    c_col = np.array([1.0, 0.5])
    sampling_scheme = batch_selection.BMinSepSampling(
        sampling_prob=0.5,
        min_sep=2,
        iterations=3,
        warm_start=warm_start,
        truncated_batch_size=truncated_batch_size,
    )
    samples, rbs = sample_generation.generate_sample(
        sampling_scheme,
        noise_multiplier=1e-9,
        c_col=c_col,
        positive_sample=positive_sample,
        num_samples=10000,
        dataset_size=dataset_size,
    )
    _assert_mode_frequencies(samples, modes, mode_distribution)
    _assert_mode_frequencies(rbs, rbs_modes, rbs_distribution)

  @parameterized.parameters(
      (
          True,
          1 / 2,
          np.array([1.0, 0.5]),
          _BANDMF_MODES,
          np.array([1 / 8, 1 / 8, 1 / 4, 1 / 8, 3 / 8]),
      ),
      (
          True,
          1 / 2,
          np.array([1.0]),
          _DP_SGD_MODES,
          np.array([1 / 8, 1 / 8, 1 / 4, 1 / 8, 3 / 8]),
      ),
      (
          False,
          1 / 2,
          np.array([1.0, 0.5]),
          [np.array([0.0, 0.0, 0.0])],
          np.array([1.0]),
      ),
      (
          True,
          1 / 3,
          np.array([1.0, 0.5]),
          _BANDMF_MODES,
          np.array([1 / 18, 1 / 9, 1 / 6, 1 / 9, 5 / 9]),
      ),
  )
  def test_generate_cyclic_poisson_sample(
      self, positive_sample, sampling_prob, c_col, modes, distribution
  ):
    sampling_scheme = batch_selection.CyclicPoissonSampling(
        sampling_prob=sampling_prob,
        cycle_length=2,
        iterations=3,
        partition_type=batch_selection.PartitionType.INDEPENDENT,
    )
    samples = sample_generation.generate_sample(
        sampling_scheme,
        noise_multiplier=1e-9,
        c_col=c_col,
        positive_sample=positive_sample,
        num_samples=10000,
    )
    _assert_mode_frequencies(samples, modes, distribution)

  @parameterized.parameters([
      (
          np.array([[1.0], [0.0], [1.0], [0.0]]),
          np.array([1.0, 0.0]),
          [0.4337808304830272],
      ),
      (
          np.array([[1.0], [0.5], [1.0], [0.5]]),
          np.array([1.0, 0.5]),
          [0.9052974004451105],
      ),
      (
          np.array([[1.0], [1.0], [1.0], [1.0]]),
          np.array([1.0, 1.0, 0.5, 0.5]),
          [1.5799760835798948],
      ),
  ])
  def test_compute_privacy_loss_balls_in_bins(
      self, sample, c_col, expected_privacy_loss
  ):
    sampling_scheme = batch_selection.BallsInBinsSampling(
        cycle_length=2, iterations=4
    )
    noise_multiplier = 1.0
    privacy_loss = sample_generation.compute_privacy_loss(
        sampling_scheme,
        sample,
        noise_multiplier,
        c_col,
    )
    np.testing.assert_allclose(privacy_loss, expected_privacy_loss, atol=1e-6)

  @parameterized.parameters([
      (
          np.array([
              [1.0, 0.0],
              [0.5, 1.0],
              [1.0, 0.5],
              [0.5, 1.0],
          ]),
          np.array([1.0, 0.5]),
          [0.9052974004451105, 0.7802974004451104],
      ),
  ])
  def test_compute_privacy_loss_balls_in_bins_multiple_samples(
      self, sample, c_col, expected_privacy_loss
  ):
    sampling_scheme = batch_selection.BallsInBinsSampling(
        cycle_length=2, iterations=4
    )
    noise_multiplier = 1.0
    privacy_loss = sample_generation.compute_privacy_loss(
        sampling_scheme,
        sample,
        noise_multiplier,
        c_col,
    )
    np.testing.assert_allclose(privacy_loss, expected_privacy_loss, atol=1e-6)

  @parameterized.named_parameters(
      ('dp_sgd', 7, 3, [1.0], 1.0),
      ('bands_below_cycle', 7, 3, [1.0, 0.5], 1.0),
      ('bands_equal_cycle', 9, 3, [1.0, 0.5, 0.25], 1.0),
      ('bands_above_cycle', 10, 3, [1.0, 0.5, 0.25, 0.125, 0.0625], 1.0),
      ('single_epoch', 6, 6, [1.0, 0.5, 0.25], 1.0),
      ('cycle_above_iterations', 4, 6, [1.0, 0.5], 1.0),
      ('c_col_above_iterations', 3, 2, [1.0, 0.5, 0.25, 0.125], 1.0),
      ('mode_scale', 8, 3, [1.0, 0.5, 0.25], 2.5),
      ('negative_mode_scale', 8, 3, [1.0, 0.5, 0.25], -0.5),
  )
  def test_compute_privacy_loss_balls_in_bins_matches_dense_modes(
      self, iterations, cycle_length, c_col, mode_scale
  ):
    sampling_scheme = batch_selection.BallsInBinsSampling(
        cycle_length=cycle_length, iterations=iterations
    )
    noise_multiplier, c_col = 0.8, np.array(c_col)
    samples = np.random.default_rng(0xBAD5EED).normal(size=(iterations, 5))
    privacy_loss = sample_generation.compute_privacy_loss(
        sampling_scheme, samples, noise_multiplier, c_col, mode_scale=mode_scale
    )
    # Brute-force log P_w(y) / Q(y) by multiplying by the dense mode matrix.
    modes = mode_scale * sample_generation.balls_in_bins_modes(
        sampling_scheme, c_col
    )
    llrs = (2 * modes.T @ samples - (modes**2).sum(axis=0)[:, None]) / (
        2 * noise_multiplier**2
    )
    expected = scipy.special.logsumexp(llrs, axis=0) - np.log(cycle_length)
    np.testing.assert_allclose(privacy_loss, expected, atol=1e-10)

  @parameterized.parameters(0.5, 1.0, 1.5, -0.5)
  def test_compute_privacy_loss_balls_in_bins_mode_scale(self, mode_scale):
    sampling_scheme = batch_selection.BallsInBinsSampling(
        cycle_length=3, iterations=7
    )
    noise_multiplier, c_col = 0.8, np.array([1.0, 0.5, 0.25])
    samples = np.random.default_rng(0xBAD5EED).normal(size=(7, 5))
    privacy_loss = sample_generation.compute_privacy_loss(
        sampling_scheme, samples, noise_multiplier, c_col, mode_scale=mode_scale
    )
    # Brute-force log P_w(y) / Q(y) over the (scaled) balls-in-bins modes.
    modes = mode_scale * sample_generation._all_balls_in_bins_modes(
        7, 3, tuple(c_col)
    )
    llrs = (2 * modes.T @ samples - (modes**2).sum(axis=0)[:, None]) / (
        2 * noise_multiplier**2
    )
    expected = scipy.special.logsumexp(llrs, axis=0) - np.log(3)
    np.testing.assert_allclose(privacy_loss, expected, atol=1e-10)
    if mode_scale > 0:
      # Positive scales are equivalent to scaling the strategy matrix.
      np.testing.assert_allclose(
          privacy_loss,
          sample_generation.compute_privacy_loss(
              sampling_scheme, samples, noise_multiplier, mode_scale * c_col
          ),
          atol=1e-10,
      )

  @parameterized.parameters(
      batch_selection.BMinSepSampling(
          min_sep=2, sampling_prob=0.5, iterations=4
      ),
      batch_selection.CyclicPoissonSampling(
          cycle_length=2, sampling_prob=0.5, iterations=4
      ),
  )
  def test_compute_privacy_loss_raises_for_mode_scale(self, strategy):
    with self.assertRaises(ValueError):
      sample_generation.compute_privacy_loss(
          strategy, np.zeros((4, 1)), 1.0, np.array([1.0]), mode_scale=2.0
      )

  @parameterized.parameters([
      (
          np.array([[1.0], [1.0], [1.0]]),
          np.array([1.0]),
          False,
          [0.6070560625306676],
      ),
      (
          np.array([[1.0], [1.0], [1.0]]),
          np.array([1.0, 0.5]),
          False,
          [0.9239798890121712],
      ),
      (
          np.array([[1.0], [1.0], [1.0]]),
          np.array([1.0, 0.5]),
          True,
          [0.8329398380809252],
      ),
  ])
  def test_compute_privacy_loss_b_min_sep(
      self, sample, c_col, warm_start, expected_privacy_loss
  ):
    sampling_scheme = batch_selection.BMinSepSampling(
        sampling_prob=0.5,
        min_sep=2,
        iterations=3,
        warm_start=warm_start,
    )
    privacy_loss = sample_generation.compute_privacy_loss(
        sampling_scheme,
        sample,
        1.0,
        c_col,
    )
    np.testing.assert_allclose(privacy_loss, expected_privacy_loss, atol=1e-6)

  @parameterized.parameters([
      (
          np.array([[1.0], [1.0], [1.0]]),
          np.array([1.0, 0.5]),
          np.array([[1], [0], [0]]),
          1,
          False,
          [0.8407396632949528],
      ),
      (
          np.array([[1.0], [1.0], [1.0]]),
          np.array([1.0, 0.5]),
          np.array([[2], [0], [1]]),
          1,
          False,
          [0.6833726274094434],
      ),
      (
          np.array([[1.0], [1.0], [1.0]]),
          np.array([1.0, 0.5]),
          np.array([[2], [0], [1]]),
          2,
          False,
          [0.978400107545793],
      ),
      (
          np.array([[1.0], [1.0], [1.0]]),
          np.array([1.0, 0.5]),
          np.array([[2], [0], [1]]),
          1,
          True,
          [0.6643184939690996],
      ),
  ])
  def test_compute_privacy_loss_b_min_sep_with_truncation(
      self,
      sample,
      c_col,
      rest_batch_sizes,
      truncated_batch_size,
      warm_start,
      expected_privacy_loss,
  ):
    sampling_scheme = batch_selection.BMinSepSampling(
        sampling_prob=0.5,
        min_sep=2,
        iterations=3,
        warm_start=warm_start,
        truncated_batch_size=truncated_batch_size,
    )
    privacy_loss = sample_generation.compute_privacy_loss(
        sampling_scheme,
        sample,
        1.0,
        c_col,
        aux=rest_batch_sizes,
    )
    np.testing.assert_allclose(privacy_loss, expected_privacy_loss, atol=1e-6)

  @parameterized.parameters([
      dict(
          sample=np.array([
              [1.0, 0.0],
              [1.0, 1.0],
              [1.0, 1.0],
          ]),
          c_col=np.array([1.0, 0.5]),
          warm_start=False,
          expected_privacy_loss=[0.9239798890121712, 0.4155349444473233],
          truncated_batch_size=None,
          rest_batch_sizes=None,
      ),
      dict(
          sample=np.array([
              [1.0, 1.0],
              [1.0, 1.0],
              [1.0, 1.0],
          ]),
          c_col=np.array([1.0, 0.5]),
          warm_start=False,
          expected_privacy_loss=[0.8407396632949528, 0.6833726274094434],
          truncated_batch_size=1,
          rest_batch_sizes=np.array([[1, 2], [0, 0], [0, 1]]),
      ),
  ])
  def test_compute_privacy_loss_b_min_sep_multiple_samples(
      self,
      sample,
      c_col,
      warm_start,
      expected_privacy_loss,
      truncated_batch_size,
      rest_batch_sizes,
  ):
    sampling_scheme = batch_selection.BMinSepSampling(
        sampling_prob=0.5,
        min_sep=2,
        iterations=3,
        warm_start=warm_start,
        truncated_batch_size=truncated_batch_size,
    )
    privacy_loss = sample_generation.compute_privacy_loss(
        sampling_scheme,
        sample,
        1.0,
        c_col,
        aux=rest_batch_sizes,
    )
    np.testing.assert_allclose(privacy_loss, expected_privacy_loss, atol=1e-6)

  @parameterized.product(
      min_sep_and_c_col_size=[(2, 2), (3, 3), (4, 3), (4, 4)],
      warm_start=[False, True],
  )
  def test_compute_privacy_loss_b_min_sep_matches_brute_force(
      self, min_sep_and_c_col_size, warm_start
  ):
    min_sep, c_col_size = min_sep_and_c_col_size
    sampling_scheme = batch_selection.BMinSepSampling(
        sampling_prob=0.4,
        min_sep=min_sep,
        iterations=6,
        warm_start=warm_start,
    )
    rng = np.random.default_rng(0xBAD5EED)
    c_col = rng.uniform(0.2, 1.0, size=c_col_size)
    samples = rng.normal(size=(sampling_scheme.iterations, 5))
    privacy_loss = sample_generation.compute_privacy_loss(
        sampling_scheme, samples, 1.0, c_col
    )
    expected_privacy_loss = _brute_force_b_min_sep_privacy_loss(
        sampling_scheme, samples, 1.0, c_col
    )
    np.testing.assert_allclose(privacy_loss, expected_privacy_loss, atol=1e-10)

  @parameterized.parameters(False, True)
  def test_compute_privacy_loss_b_min_sep_no_truncation_events_matches(
      self, warm_start
  ):
    kwargs = dict(
        sampling_prob=0.4, min_sep=3, iterations=6, warm_start=warm_start
    )
    untruncated_scheme = batch_selection.BMinSepSampling(**kwargs)
    truncated_scheme = batch_selection.BMinSepSampling(
        truncated_batch_size=1, **kwargs
    )
    rng = np.random.default_rng(0xBAD5EED)
    c_col = rng.uniform(0.2, 1.0, size=3)
    samples = rng.normal(size=(untruncated_scheme.iterations, 5))
    # No batch ever reaches truncated_batch_size, so truncation never occurs.
    rest_batch_sizes = np.zeros_like(samples, dtype=np.int32)
    privacy_loss = sample_generation.compute_privacy_loss(
        truncated_scheme, samples, 1.0, c_col, aux=rest_batch_sizes
    )
    expected_privacy_loss = sample_generation.compute_privacy_loss(
        untruncated_scheme, samples, 1.0, c_col
    )
    np.testing.assert_allclose(privacy_loss, expected_privacy_loss, atol=1e-10)

  @parameterized.parameters([
      (
          np.array([[1.0], [0.0], [1.0]]),
          1 / 2,
          np.array([1.0]),
          [0.24576433028848393],
      ),
      (
          np.array([[1.0], [0.5], [1.0]]),
          1 / 2,
          np.array([1.0, 0.5]),
          [0.4468602908276156],
      ),
      (
          np.array([[1.0], [0.5], [1.0]]),
          1 / 3,
          np.array([1.0, 0.5]),
          [0.30744897807389276],
      ),
  ])
  def test_compute_privacy_loss_cyclic_poisson(
      self,
      sample,
      sampling_prob,
      c_col,
      expected_privacy_loss,
  ):
    sampling_scheme = batch_selection.CyclicPoissonSampling(
        sampling_prob=sampling_prob,
        cycle_length=2,
        iterations=3,
        partition_type=batch_selection.PartitionType.INDEPENDENT,
    )
    privacy_loss = sample_generation.compute_privacy_loss(
        sampling_scheme,
        sample,
        1.0,
        c_col,
    )
    np.testing.assert_allclose(privacy_loss, expected_privacy_loss, atol=1e-6)

  def test_compute_privacy_loss_cyclic_poisson_matches_balls_in_bins(
      self,
  ):
    for _ in range(1000):
      cycle_length = np.random.randint(1, 5)
      iterations = np.random.randint(cycle_length, 10)
      c_col = np.random.uniform(size=cycle_length)
      sample = np.random.uniform(size=(iterations, 1))
      sampling_scheme = batch_selection.CyclicPoissonSampling(
          sampling_prob=1.0,
          cycle_length=cycle_length,
          iterations=iterations,
          partition_type=batch_selection.PartitionType.INDEPENDENT,
      )
      equivalent_sampling_scheme = batch_selection.BallsInBinsSampling(
          cycle_length=cycle_length,
          iterations=iterations,
      )
      privacy_loss_1 = sample_generation.compute_privacy_loss(
          sampling_scheme,
          sample,
          1.0,
          c_col,
      )
      privacy_loss_2 = sample_generation.compute_privacy_loss(
          equivalent_sampling_scheme,
          sample,
          1.0,
          c_col,
      )
      np.testing.assert_almost_equal(privacy_loss_1, privacy_loss_2)

  @parameterized.parameters([
      (
          np.array([[1.0, 1.0], [0.5, 1.0], [1.0, 1.0]]),
          1 / 2,
          np.array([1.0, 0.5]),
          [0.4468602908276156, 0.6805952255579165],
      ),
      (
          np.array([[1.0, 1.0], [0.5, 1.0], [1.0, 1.0]]),
          1 / 3,
          np.array([1.0, 0.5]),
          [0.30744897807389276, 0.485401681402855],
      ),
  ])
  def test_compute_privacy_loss_cyclic_poisson_multiple_samples(
      self,
      sample,
      sampling_prob,
      c_col,
      expected_privacy_loss,
  ):
    sampling_scheme = batch_selection.CyclicPoissonSampling(
        sampling_prob=sampling_prob,
        cycle_length=2,
        iterations=3,
        partition_type=batch_selection.PartitionType.INDEPENDENT,
    )
    privacy_loss = sample_generation.compute_privacy_loss(
        sampling_scheme,
        sample,
        1.0,
        c_col,
    )
    np.testing.assert_allclose(privacy_loss, expected_privacy_loss, atol=1e-6)

  def test_get_privacy_loss_positive_sample(self):
    # Test that this method combines drawing a sample and computing its privacy
    # loss correctly on a low-noise example.
    sampling_scheme = batch_selection.BallsInBinsSampling(
        cycle_length=2, iterations=4
    )
    noise_multiplier = 1e-4
    c_col = np.array([1.0, 0.5])
    # In this setup, the privacy loss is very close to 1.25e8 for samples from
    # the first mode, and very close to 1.125e8 for samples from the second
    # mode.
    pl_samples, _ = sample_generation.get_privacy_loss_sample(
        sampling_scheme,
        noise_multiplier,
        c_col,
        positive_sample=True,
        num_samples=10000,
    )
    first_mode_count = sum(np.isclose(pl_samples, 1.25e8, atol=1e5))
    second_mode_count = sum(np.isclose(pl_samples, 1.125e8, atol=1e5))
    self.assertEqual(first_mode_count + second_mode_count, 10000)
    self.assertLess(first_mode_count, 5300)
    self.assertGreater(first_mode_count, 4700)

  def test_get_privacy_loss_negative_sample(self):
    # Test that this method combines drawing a sample and computing its privacy
    # loss correctly on a low-noise example.
    sampling_scheme = batch_selection.BallsInBinsSampling(
        cycle_length=2, iterations=4
    )
    noise_multiplier = 1e-4
    c_col = np.array([1.0, 0.5])
    # In this setup, the privacy loss is very close to 1.125e8 always.
    pl_samples, _ = sample_generation.get_privacy_loss_sample(
        sampling_scheme,
        noise_multiplier,
        c_col,
        positive_sample=False,
        num_samples=10000,
    )
    mode_count = sum(np.isclose(pl_samples, 1.125e8, atol=1e5))
    self.assertEqual(mode_count, 10000)

  def test_get_privacy_loss_and_sample_truncated_b_min_sep(self):
    # Test that we can also get the sample (and aux data) if desired.
    sampling_scheme = batch_selection.BMinSepSampling(
        sampling_prob=0.5,
        min_sep=2,
        iterations=3,
        warm_start=False,
        truncated_batch_size=1,
    )
    noise_multiplier = 1e-4
    c_col = np.array([1.0, 0.5])
    _, sample = sample_generation.get_privacy_loss_sample(
        sampling_scheme,
        noise_multiplier,
        c_col,
        positive_sample=True,
        dataset_size=3,
        num_samples=3,
    )
    _, _ = sample


if __name__ == '__main__':
  absltest.main()

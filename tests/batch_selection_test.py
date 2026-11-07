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

import collections
import itertools
import math

from absl.testing import absltest
from absl.testing import parameterized
from jax_privacy import batch_selection
import numpy as np


def _get_unique_elements(batches):
  """Returns the set of unique elements from a list of batches."""
  return np.unique_values(np.concatenate(batches))


def _check_batch_sizes_equal(batches, min_batch_size: int, max_batch_size: int):
  """Checks that all batches in the list have the expected size."""
  for batch in batches:
    assert min_batch_size <= len(batch) <= max_batch_size


def _check_no_repeated_indices(batches):
  """Checks that no index appears in multiple batches."""
  combined = np.concatenate(batches)
  assert len(combined) == len(set(combined))


def _check_cyclic_property(batches, cycle_length):
  """Checks that the batches have the cyclic property."""
  for i in range(cycle_length, len(batches)):
    np.testing.assert_array_equal(
        sorted(batches[i]), sorted(batches[i - cycle_length])
    )
  _check_subset_of_kb_participation(batches, cycle_length)


def _check_subset_of_kb_participation(batches, cycle_length):
  """Checks each element can only participate in every cycle_length-th round."""
  partition = []
  for i in range(cycle_length):
    partition.append(_get_unique_elements(batches[i::cycle_length]))
  for i in range(cycle_length):
    for j in range(i + 1, cycle_length):
      assert np.intersect1d(partition[i], partition[j]).size == 0


def _check_b_min_sep_property(batches, min_sep):
  """Checks that the batches have the b-min-sep property."""
  for i in range(len(batches)):
    for j in range(i + 1, min(len(batches), i + min_sep)):
      assert np.intersect1d(batches[i], batches[j]).size == 0


def _check_element_range(batches, num_examples):
  """Checks that all batches are in the range [0, num_examples)."""
  for batch in batches:
    assert np.all(batch < num_examples)
    assert np.all(batch >= 0)


def _check_signed_indices(batches):
  for batch in batches:
    assert np.issubdtype(batch.dtype, np.signedinteger)


def _check_all_equal(x):
  assert np.all(x == x[0]), f'Elements of x are not all equal: {x}'


_EQUAL_SPLIT = batch_selection.PartitionType.EQUAL_SPLIT
_INDEPENDENT = batch_selection.PartitionType.INDEPENDENT

_MIN_SEP_STRATEGIES = [
    batch_selection.CyclicPoissonSampling(0.5, 30, cycle_length=3),
    batch_selection.BallsInBinsSampling(30, cycle_length=5),
    batch_selection.RandomAllocationSampling(4, 30, cycle_length=3),
    batch_selection.RandomAllocationSampling(4, 30, 3, sampling_prob=0.5),
    batch_selection.BMinSepSampling(0.2, 50, min_sep=2, warm_start=True),
    batch_selection.BMinSepSampling(0.2, 50, min_sep=4, warm_start=False),
    batch_selection.MinSepRandomAllocation(4, 30, min_sep=3),
    batch_selection.MinSepRandomAllocation(4, 30, 3, random_rotation=True),
    batch_selection.MinSepRandomAllocation(4, 30, 3, sampling_prob=0.5),
]

_CYCLIC_PARTITION_STRATEGIES = [
    batch_selection.CyclicPoissonSampling(0.5, 20, None, 3, _INDEPENDENT),
    batch_selection.CyclicPoissonSampling(0.5, 20, None, 3, _EQUAL_SPLIT),
    batch_selection.BallsInBinsSampling(20, cycle_length=4),
    batch_selection.RandomAllocationSampling(4, 20, 3, _INDEPENDENT),
    batch_selection.RandomAllocationSampling(4, 20, 3, _EQUAL_SPLIT),
]

# Strategies in the saturated regime (sampling_prob=1 or k = t / b), where the
# first b batches partition the dataset and the schedule repeats every b steps.
_PERIODIC_STRATEGIES = [
    batch_selection.CyclicPoissonSampling(1.0, 10, None, 3, _INDEPENDENT),
    batch_selection.CyclicPoissonSampling(1.0, 10, None, 3, _EQUAL_SPLIT),
    batch_selection.BallsInBinsSampling(10, cycle_length=3),
    batch_selection.BallsInBinsSampling(20, cycle_length=15),
    batch_selection.RandomAllocationSampling(3, 9, cycle_length=3),
    batch_selection.BMinSepSampling(1.0, 10, min_sep=2, warm_start=True),
    batch_selection.BMinSepSampling(1.0, 10, min_sep=2, warm_start=False),
]

_EXACT_K_STRATEGIES = [
    batch_selection.BallsInBinsSampling(20, cycle_length=5),
    batch_selection.RandomAllocationSampling(0, 10),
    batch_selection.RandomAllocationSampling(1, 20, cycle_length=1),
    batch_selection.RandomAllocationSampling(3, 20, cycle_length=2),
    batch_selection.RandomAllocationSampling(5, 21, cycle_length=3),
    batch_selection.MinSepRandomAllocation(1, 10),
    batch_selection.MinSepRandomAllocation(3, 20, min_sep=4),
    batch_selection.MinSepRandomAllocation(4, 20, 3, random_rotation=True),
]

_SUBSAMPLED_K_STRATEGIES = [
    batch_selection.RandomAllocationSampling(8, 96, 4, _INDEPENDENT, 0.5),
    batch_selection.RandomAllocationSampling(8, 96, 4, _EQUAL_SPLIT, 0.5),
    batch_selection.MinSepRandomAllocation(8, 96, 4, False, 0.5),
    batch_selection.MinSepRandomAllocation(8, 96, 4, True, 0.5),
]

_TRUNCATED_STRATEGIES = [
    batch_selection.CyclicPoissonSampling(0.5, 50, truncated_batch_size=4),
    batch_selection.CyclicPoissonSampling(0.5, 50, 4, cycle_length=3),
    batch_selection.BMinSepSampling(0.5, 50, 2, truncated_batch_size=4),
]

_STRATEGIES = [
    batch_selection.CyclicPoissonSampling(0.25, 12),
    batch_selection.FixedBatchSampling(4, 6),
    batch_selection.FixedBatchSampling(10, 4, replace=True),
    batch_selection.BMinSepSampling(0.2, 12, min_sep=1),
    *_MIN_SEP_STRATEGIES,
    *_CYCLIC_PARTITION_STRATEGIES,
    *_PERIODIC_STRATEGIES,
    *_EXACT_K_STRATEGIES,
    *_SUBSAMPLED_K_STRATEGIES,
    *_TRUNCATED_STRATEGIES,
]


class BatchSelectionTest(parameterized.TestCase):

  @parameterized.product(strategy=_STRATEGIES, num_examples=[0, 1, 50])
  def test_output_invariants(self, strategy, num_examples):
    """Checks batch count, index range, signed dtype, and uniqueness."""
    if isinstance(strategy, batch_selection.FixedBatchSampling):
      min_examples = 1 if strategy.replace else strategy.batch_size
      if num_examples < min_examples:
        with self.assertRaises(ValueError):
          list(strategy.batch_iterator(num_examples, rng=0))
        return

    batches = list(strategy.batch_iterator(num_examples, rng=0))
    self.assertLen(batches, strategy.iterations)
    _check_element_range(batches, num_examples)
    _check_signed_indices(batches)
    if isinstance(strategy, batch_selection.FixedBatchSampling):
      size = strategy.batch_size
      _check_batch_sizes_equal(batches, size, size)
    if not getattr(strategy, 'replace', False):
      for batch in batches:
        self.assertEqual(len(batch), len(set(batch.tolist())))

  @parameterized.parameters(*_STRATEGIES)
  def test_is_deterministic(self, strategy):
    """Strategies should produce identical batches given the same RNG."""
    batches_a = list(strategy.batch_iterator(50, rng=0))
    batches_b = list(strategy.batch_iterator(50, rng=0))
    for batch_a, batch_b in zip(batches_a, batches_b, strict=True):
      np.testing.assert_array_equal(batch_a, batch_b)

  @parameterized.parameters(*_MIN_SEP_STRATEGIES)
  def test_min_separation(self, strategy):
    """Consecutive participations of an example are at least min_sep apart."""
    batches = list(strategy.batch_iterator(100, rng=0))
    min_sep = getattr(strategy, 'min_sep', None) or strategy.cycle_length
    _check_b_min_sep_property(batches, min_sep)

  @parameterized.parameters(*_CYCLIC_PARTITION_STRATEGIES)
  def test_cyclic_partition(self, strategy):
    """Examples only participate in their residue class mod cycle_length."""
    num_examples = 50
    batches = list(strategy.batch_iterator(num_examples, rng=0))
    b = strategy.cycle_length
    _check_subset_of_kb_participation(batches, b)
    if getattr(strategy, 'partition_type', None) == _EQUAL_SPLIT:
      num_unique = _get_unique_elements(batches).size
      self.assertLessEqual(num_unique, (num_examples // b) * b)

  @parameterized.parameters(*_PERIODIC_STRATEGIES)
  def test_periodic_schedule_repeats_every_cycle(self, strategy):
    """Saturated cyclic schedules partition the data and repeat each cycle."""
    num_examples = 50
    batches = list(strategy.batch_iterator(num_examples, rng=0))
    cycle_length = getattr(strategy, 'min_sep', None) or strategy.cycle_length
    _check_no_repeated_indices(batches[:cycle_length])
    _check_cyclic_property(batches, cycle_length)
    if getattr(strategy, 'partition_type', None) == _EQUAL_SPLIT:
      batch_size = num_examples // cycle_length
      _check_batch_sizes_equal(batches, batch_size, batch_size)
    elif getattr(strategy, 'warm_start', True):
      total_in_cycle = sum(len(b) for b in batches[:cycle_length])
      self.assertEqual(total_in_cycle, num_examples)
    else:
      self.assertLen(batches[0], num_examples)
      self.assertEmpty(batches[1])

  @parameterized.parameters(*_EXACT_K_STRATEGIES)
  def test_exact_k_participations(self, strategy):
    """Every example participates in *exactly* k batches."""
    num_examples = 100
    batches = list(strategy.batch_iterator(num_examples, rng=0))
    counts = np.bincount(np.concatenate(batches), minlength=num_examples)
    if hasattr(strategy, 'total_participations'):
      k = strategy.total_participations
    else:
      k = strategy.iterations // strategy.cycle_length
    np.testing.assert_array_equal(counts, k)

  def test_equal_split_discards_remainder_in_exact_k(self):
    """EQUAL_SPLIT gives k participations to kept examples and drops rest."""
    num_examples, cycle_length, k = 10, 3, 3
    strategy = batch_selection.RandomAllocationSampling(
        k, 12, cycle_length=cycle_length, partition_type=_EQUAL_SPLIT
    )
    batches = list(strategy.batch_iterator(num_examples, rng=0))
    counts = collections.Counter(int(x) for x in np.concatenate(batches))
    self.assertLen(counts, (num_examples // cycle_length) * cycle_length)
    for count in counts.values():
      self.assertEqual(count, k)

  def test_random_allocation_uneven_iterations_truncation(self):
    """Uneven iterations logs a warning and truncates the last cycle."""
    k, cycle_length, iterations = 3, 3, 20
    with self.assertLogs(level='WARNING') as logs:
      strategy = batch_selection.RandomAllocationSampling(
          k, iterations, cycle_length=cycle_length
      )
    self.assertIn('multiple of cycle_length', logs.output[0])
    batches = list(strategy.batch_iterator(100, rng=0))
    counts = collections.Counter(int(x) for x in np.concatenate(batches))
    for count in counts.values():
      self.assertIn(count, (k - 1, k))
    for step, batch in enumerate(batches):
      if step % cycle_length < iterations % cycle_length:
        for idx in batch:
          self.assertEqual(counts[int(idx)], k)

  @parameterized.parameters(*_SUBSAMPLED_K_STRATEGIES)
  def test_subsampled_k_participations(self, strategy):
    """Subsampled fixed-k strategies have count <= k and mean ~= k * p."""
    n = 20000
    batches = list(strategy.batch_iterator(n, rng=0))
    counts = np.bincount(np.concatenate(batches), minlength=n)
    self.assertLessEqual(counts.max(), strategy.total_participations)
    expected = strategy.total_participations * strategy.sampling_prob
    self.assertAlmostEqual(counts.mean(), expected, delta=0.05)

  @parameterized.parameters(*_TRUNCATED_STRATEGIES)
  def test_truncated_batch_size(self, strategy):
    """Batches never exceed truncated_batch_size when set."""
    batches = list(strategy.batch_iterator(100, rng=0))
    _check_batch_sizes_equal(batches, 0, strategy.truncated_batch_size)

  @parameterized.parameters(
      batch_selection.CyclicPoissonSampling(0.1, 2000),
      batch_selection.BMinSepSampling(0.1, 2000, min_sep=1),
  )
  def test_poisson_inclusion_frequency(self, strategy):
    """Plain Poisson strategies include examples at rate ~= sampling_prob."""
    num_examples = 200
    batches = list(strategy.batch_iterator(num_examples, rng=0))
    total = sum(len(b) for b in batches)
    freq = total / (strategy.iterations * num_examples)
    self.assertAlmostEqual(freq, strategy.sampling_prob, delta=0.005)

  @parameterized.parameters(
      batch_selection.BallsInBinsSampling(6, cycle_length=3),
      batch_selection.RandomAllocationSampling(2, 5, cycle_length=1),
      batch_selection.RandomAllocationSampling(2, 6, cycle_length=2),
      batch_selection.MinSepRandomAllocation(2, 6, min_sep=2),
      batch_selection.MinSepRandomAllocation(3, 7, min_sep=2),
      batch_selection.MinSepRandomAllocation(2, 6, 2, random_rotation=True),
      batch_selection.MinSepRandomAllocation(3, 7, 2, random_rotation=True),
  )
  def test_uniform_over_valid_participation_patterns(self, strategy):
    """Fixed-k strategies sample uniformly over all valid k-step patterns."""
    n, t = 30000, strategy.iterations
    if hasattr(strategy, 'total_participations'):
      k = strategy.total_participations
    else:
      k = t // strategy.cycle_length

    def is_valid(steps):
      gaps = [y - x for x, y in itertools.pairwise(steps)]
      if isinstance(strategy, batch_selection.MinSepRandomAllocation):
        if strategy.random_rotation:
          gaps.append(steps[0] + t - steps[-1])
        return min(gaps) >= strategy.min_sep
      b = strategy.cycle_length
      if isinstance(strategy, batch_selection.BallsInBinsSampling):
        return all(gap == b for gap in gaps)
      return len({s % b for s in steps}) <= 1

    # Encode each example's realized participation set as a t-bit integer mask
    # (bit s is 1 iff the example participated in step s) and compare the
    # empirical histogram over [0, 2**t) to the uniform distribution on valid
    # k-subsets.
    masks = np.zeros(n, dtype=np.int64)
    for step, batch in enumerate(strategy.batch_iterator(n, rng=0)):
      masks[batch] |= 1 << step
    counts = np.bincount(masks, minlength=1 << t) / n

    all_combos = itertools.combinations(range(t), k)
    valid_masks = [sum(1 << s for s in c) for c in all_combos if is_valid(c)]
    expected = np.zeros(1 << t)
    expected[valid_masks] = 1 / len(valid_masks)
    np.testing.assert_allclose(counts, expected, atol=0.015)

  @parameterized.parameters(_INDEPENDENT, _EQUAL_SPLIT)
  def test_random_allocation_matches_cyclic_poisson(self, partition_type):
    """Saturated RandomAllocationSampling with prob=q matches CyclicPoisson."""
    b, t, q, n = 4, 40, 0.5, 30000
    k = t // b
    ra = batch_selection.RandomAllocationSampling(k, t, b, partition_type, q)
    cyclic = batch_selection.CyclicPoissonSampling(
        q, t, cycle_length=b, partition_type=partition_type
    )

    def count_hist(strategy):
      indices = np.concatenate(list(strategy.batch_iterator(n, rng=0)))
      counts = np.bincount(indices, minlength=n)
      return np.bincount(counts, minlength=k + 1) / n

    np.testing.assert_allclose(count_hist(ra), count_hist(cyclic), atol=0.02)

  @parameterized.parameters(
      {'total_participations': 11},
      {'total_participations': 5, 'cycle_length': 3},
      {'total_participations': -1},
      {'cycle_length': 0},
      {'sampling_prob': -0.1},
      {'sampling_prob': 1.1},
  )
  def test_random_allocation_sampling_validation(self, **kwargs):
    defaults = {'total_participations': 2, 'iterations': 10}
    with self.assertRaises(ValueError):
      batch_selection.RandomAllocationSampling(**(defaults | kwargs))

  def test_min_sep_random_allocation_cyclic_wrap_around(self):
    """random_rotation=True also enforces min_sep across the t -> 0 boundary."""
    min_sep = 4
    strategy = batch_selection.MinSepRandomAllocation(
        3, 15, min_sep=min_sep, random_rotation=True
    )
    batches = list(strategy.batch_iterator(200, rng=0))
    _check_b_min_sep_property(batches + batches[: min_sep - 1], min_sep)

  def test_min_sep_random_allocation_random_rotation_flattens_marginal(self):
    """random_rotation=True yields uniform per-step marginal k / t."""
    k, t, min_sep, n = 3, 10, 3, 20000
    strategy = batch_selection.MinSepRandomAllocation(
        k, t, min_sep=min_sep, random_rotation=True
    )
    sizes = [len(b) / n for b in strategy.batch_iterator(n, rng=0)]
    np.testing.assert_allclose(sizes, k / t, atol=0.02)

  @parameterized.parameters(
      {'total_participations': 0},
      {'iterations': 0},
      {'min_sep': 0},
      {'sampling_prob': -0.1},
      {'sampling_prob': 1.1},
      {'min_sep': 4, 'random_rotation': False},
      {'min_sep': 4, 'random_rotation': True},
  )
  def test_min_sep_random_allocation_validation(self, **kwargs):
    defaults = {'total_participations': 3, 'iterations': 10}
    with self.assertRaises(ValueError):
      batch_selection.MinSepRandomAllocation(**(defaults | kwargs))

  def test_user_selection_strategy(self):
    """Tests for UserSelectionStrategy."""
    base_strategy = batch_selection.CyclicPoissonSampling(
        sampling_prob=0.5,
        iterations=5,
        cycle_length=2,
    )
    strategy = batch_selection.UserSelectionStrategy(
        base_strategy, examples_per_user_per_batch=2, shuffle_per_user=True
    )
    unique_ids = np.random.choice(2**30, size=16, replace=False)
    user_ids = np.concatenate([[unique_ids[i]] * i for i in range(16)])
    np.random.shuffle(user_ids)
    for batch in strategy.batch_iterator(user_ids, rng=0):
      for row in batch:
        _check_all_equal(user_ids[row])
      user_batch = user_ids[batch[:, 0]]
      self.assertLen(set(user_batch), user_batch.shape[0])

  def test_user_selection_preserves_example_order_without_shuffle(self):
    """UserSelectionStrategy keeps per-user example order when not shuffled."""

    class _FixedBatchStrategy(batch_selection.BatchSelectionStrategy):

      def batch_iterator(self, num_examples: int, rng=None):
        del num_examples, rng
        yield np.array([2, 0, 1], dtype=np.int32)

    strategy = batch_selection.UserSelectionStrategy(
        _FixedBatchStrategy(),
        examples_per_user_per_batch=2,
        shuffle_per_user=False,
    )
    user_ids = np.array([10, 10, 20, 20, 20, 30])
    batch = next(strategy.batch_iterator(user_ids, rng=0))
    np.testing.assert_array_equal(batch[0], np.array([5, 5]))
    np.testing.assert_array_equal(batch[1], np.array([0, 1]))
    np.testing.assert_array_equal(batch[2], np.array([2, 3]))

  @parameterized.named_parameters(
      ('empty_sample', np.array([0, 1])),
      ('empty_dataset', np.array([], dtype=np.int64)),
  )
  def test_user_selection_returns_typed_2d_empty_batch(self, user_ids):
    base_strategy = batch_selection.CyclicPoissonSampling(
        sampling_prob=0, iterations=1
    )
    strategy = batch_selection.UserSelectionStrategy(
        base_strategy, examples_per_user_per_batch=2
    )

    batch = next(strategy.batch_iterator(user_ids, rng=0))

    self.assertEqual(batch.shape, (0, 2))
    self.assertTrue(np.issubdtype(batch.dtype, np.signedinteger))


class BatchPaddingTest(parameterized.TestCase):

  @parameterized.parameters(
      (10, 4, None),
      (10, 4, 2),
      (10, 5, None),
      (10, 1, None),
      (17, 6, 6),
      (17, 6, 3),
      (17, 6, 2),
      (17, 6, 1),
      (17, 6, None),
      (17, 17, None),
  )
  def test_split_and_pad(
      self, global_batch_size, minibatch_size, microbatch_size
  ):
    num_examples = 1000
    indices = np.random.randint(0, num_examples, size=global_batch_size)
    minibatches = batch_selection.split_and_pad_global_batch(
        indices, minibatch_size, microbatch_size
    )
    self.assertLen(minibatches, math.ceil(global_batch_size / minibatch_size))

    for minibatch in minibatches[:-1]:
      self.assertLen(minibatch, minibatch_size)
      self.assertTrue(np.all(minibatch != -1))

    self.assertLen(minibatches[-1], minibatch_size)
    self.assertFalse(np.all(minibatches[-1] == -1))

    new_indices = np.concatenate(minibatches)
    np.testing.assert_array_equal(
        np.sort(new_indices[new_indices != -1]), np.sort(indices)
    )

  @parameterized.parameters(
      (10, 4),
      (10, 5),
      (10, 1),
      (17, 6),
      (17, 17),
  )
  def test_pad_to_multiple_of(self, global_batch_size, multiple):
    indices = np.random.randint(0, 1000, size=global_batch_size)
    new_indices = batch_selection.pad_to_multiple_of(indices, multiple)
    self.assertEqual(new_indices.size % multiple, 0)
    np.testing.assert_array_equal(
        np.sort(new_indices[new_indices != -1]), np.sort(indices)
    )

  def test_pad_to_multiple_of_empty_indices(self):
    new_indices = batch_selection.pad_to_multiple_of(
        np.array([], dtype=np.int32), multiple=4
    )
    np.testing.assert_array_equal(new_indices, np.array([], dtype=np.int32))

  @parameterized.parameters(
      (10, 12, 3),
      (10, 12, 4),
      (10, 12, 6),
      (7, 12, 3),
      (7, 12, 4),
      (5, 6, 2),
      (5, 6, 3),
  )
  def test_pad_to_multiple_of_with_microbatch_size(
      self, global_batch_size, multiple, microbatch_size
  ):
    indices = np.arange(global_batch_size)
    padded = batch_selection.pad_to_multiple_of(
        indices, multiple, microbatch_size=microbatch_size
    )
    padded_size = padded.shape[0]
    self.assertEqual(padded_size % multiple, 0)
    self.assertEqual(padded_size % microbatch_size, 0)
    # All original indices are preserved.
    np.testing.assert_array_equal(
        np.sort(padded[padded != -1]), np.sort(indices)
    )
    # Simulate the Fortran-order reshape that microbatching performs.
    num_microbatches = padded_size // microbatch_size
    microbatches = padded.reshape(num_microbatches, microbatch_size, order='F')
    # Padding should be clustered in the last microbatch(es).
    is_padding = microbatches == -1
    for i in range(num_microbatches):
      if is_padding[i].any():
        # Every subsequent microbatch must be all-padding.
        for j in range(i + 1, num_microbatches):
          self.assertTrue(
              is_padding[j].all(),
              f'Microbatch {j} has real examples after microbatch {i} has'
              ' padding.',
          )
        break

  @parameterized.parameters(
      (4, 6),
      (32, 128),
      (12, 5),
  )
  def test_pad_to_multiple_of_incompatible_microbatch_size_raises(
      self, multiple, microbatch_size
  ):
    with self.assertRaisesRegex(ValueError, 'must be divisible by'):
      batch_selection.pad_to_multiple_of(
          np.arange(10), multiple, microbatch_size=microbatch_size
      )

  def test_pad_to_multiple_of_microbatch_worked_example(self):
    """Worked example: 10 real indices padded to 12 with microbatch_size=3."""
    indices = np.arange(10)
    padded = batch_selection.pad_to_multiple_of(
        indices, multiple=12, microbatch_size=3
    )
    # Simulate the Fortran-order reshape: (12,) -> (4, 3).
    microbatches = padded.reshape(4, 3, order='F')
    # Microbatches 0-2 should be all real data; microbatch 3 should contain
    # the 2 padding examples.
    for i in range(3):
      self.assertTrue(
          (microbatches[i] != -1).all(),
          f'Microbatch {i} should be all real data, got {microbatches[i]}.',
      )
    num_padding_in_last = (microbatches[3] == -1).sum()
    self.assertEqual(num_padding_in_last, 2)

  @parameterized.parameters(
      (0, np.int8),
      (127, np.int8),
      (128, np.int16),
      (32767, np.int16),
      (32768, np.int32),
      (2**31 - 1, np.int32),
      (2**31, np.int64),
  )
  def test_get_dtype_fits_positive_num_examples(
      self, num_examples, expected_dtype
  ):
    dtype = batch_selection._get_dtype(num_examples)
    self.assertEqual(dtype, np.dtype(expected_dtype))
    # Verify that scalar modulo by num_examples (e.g. in PyGrain MapDataset)
    # does not raise OverflowError under NumPy 2.0 NEP 50.
    if num_examples > 0:
      self.assertEqual(dtype.type(-1) % num_examples, num_examples - 1)


if __name__ == '__main__':
  absltest.main()

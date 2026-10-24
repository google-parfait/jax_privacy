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


def _check_max_participation(batches, max_participation):
  """Checks that the batches have the max participation property."""
  counts = collections.defaultdict(int)
  for batch in batches:
    for x in batch:
      counts[x] += 1
  assert all(count <= max_participation for count in counts.values())


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


class BatchSelectionTest(parameterized.TestCase):

  @parameterized.product(
      partition_type=[
          batch_selection.PartitionType.EQUAL_SPLIT,
          batch_selection.PartitionType.INDEPENDENT,
      ],
      num_examples=[10],
      cycle_length=[3],
      iterations=[5],
  )
  def test_cyclic_participation(
      self, *, partition_type, num_examples, cycle_length, iterations
  ):
    """Tests the use of CyclicPoissonSampling instantiated to do shuffling."""
    strategy = batch_selection.CyclicPoissonSampling(
        sampling_prob=1.0,
        iterations=iterations,
        cycle_length=cycle_length,
        partition_type=partition_type,
    )
    batches = list(strategy.batch_iterator(num_examples, rng=0))

    self.assertLen(batches, iterations)
    if partition_type == batch_selection.PartitionType.EQUAL_SPLIT:
      batch_size = num_examples // cycle_length
      _check_batch_sizes_equal(batches, batch_size, batch_size)
    _check_no_repeated_indices(batches[:cycle_length])
    _check_cyclic_property(batches, cycle_length)
    _check_element_range(batches, num_examples)
    _check_signed_indices(batches)

  @parameterized.product(
      num_examples=[100],
      iterations=[1000],
      cycle_length=[1, 3],
      expected_batch_size=[3],
      partition_type=[
          batch_selection.PartitionType.INDEPENDENT,
          batch_selection.PartitionType.EQUAL_SPLIT,
      ],
      truncated_batch_size=[None, 4],
  )
  def test_poisson_sampling(
      self,
      *,
      num_examples,
      iterations,
      cycle_length,
      expected_batch_size,
      partition_type,
      truncated_batch_size,
  ):
    """Tests for Poisson sampling, potentially cyclic and truncated."""
    sampling_prob = expected_batch_size / (num_examples // cycle_length)
    strategy = batch_selection.CyclicPoissonSampling(
        sampling_prob=sampling_prob,
        iterations=iterations,
        cycle_length=cycle_length,
        partition_type=partition_type,
        truncated_batch_size=truncated_batch_size,
    )
    batches = list(strategy.batch_iterator(num_examples, rng=0))

    self.assertLen(batches, iterations)
    min_batch_size = 0
    if truncated_batch_size:
      max_batch_size = truncated_batch_size
    else:
      max_batch_size = num_examples // cycle_length

    _check_batch_sizes_equal(batches, min_batch_size, max_batch_size)
    for start_index in range(0, iterations, cycle_length):
      _check_no_repeated_indices(
          batches[start_index : start_index + cycle_length]
      )
    _check_element_range(batches, num_examples)
    _check_subset_of_kb_participation(batches, cycle_length)
    # Make sure elements are discarded when using even partition.
    if partition_type == batch_selection.PartitionType.EQUAL_SPLIT:
      distinct_elements = _get_unique_elements(batches)
      self.assertLessEqual(
          distinct_elements.size, (num_examples // cycle_length) * cycle_length
      )
    # Make sure the expected batch size is close to the average batch size.
    if truncated_batch_size is not None:
      self.assertAlmostEqual(
          sum(len(batch) for batch in batches)
          / (iterations * expected_batch_size),
          1.0,
          delta=4 * (iterations * expected_batch_size) ** 0.5,  # ~4 std dev
      )

  def test_poisson_sampling_with_large_cycle_length(self):
    """Test for Poisson sampling with cycle_length > num_examples."""
    num_examples = 10
    cycle_length = 11
    iterations = 1100
    sampling_prob = 0.5
    strategy = batch_selection.CyclicPoissonSampling(
        sampling_prob=sampling_prob,
        iterations=iterations,
        cycle_length=cycle_length,
        partition_type=batch_selection.PartitionType.EQUAL_SPLIT,
    )
    batches = list(strategy.batch_iterator(num_examples, rng=0))

    self.assertLen(batches, iterations)
    min_batch_size = 0
    max_batch_size = 0
    _check_batch_sizes_equal(batches, min_batch_size, max_batch_size)

  @parameterized.product(
      num_examples=[100],
      cycle_length=[10],
      iterations=[10, 20, 21],
  )
  def test_balls_in_bins_sampling(self, num_examples, cycle_length, iterations):
    """Tests for balls-in-bins."""
    strategy = batch_selection.BallsInBinsSampling(
        iterations=iterations,
        cycle_length=cycle_length,
    )
    batches = list(strategy.batch_iterator(num_examples, rng=0))
    self.assertLen(batches, iterations)
    _check_element_range(batches, num_examples)
    self.assertEqual(
        sum(len(batch) for batch in batches[:cycle_length]),
        num_examples,
    )
    _check_no_repeated_indices(batches[:cycle_length])
    _check_cyclic_property(batches, cycle_length)
    _check_signed_indices(batches)

  def test_balls_in_bins_sampling_with_large_cycle_length(self):
    """Test for balls-in-bins with cycle_length > num_examples."""
    num_examples = 10
    cycle_length = 20
    iterations = 40
    strategy = batch_selection.BallsInBinsSampling(
        iterations=iterations,
        cycle_length=cycle_length,
    )
    batches = list(strategy.batch_iterator(num_examples, rng=0))
    self.assertLen(batches, iterations)
    _check_element_range(batches, num_examples)
    self.assertEqual(
        sum(len(batch) for batch in batches[:cycle_length]),
        num_examples,
    )
    _check_no_repeated_indices(batches[:cycle_length])
    _check_cyclic_property(batches, cycle_length)

  @parameterized.product(
      num_examples=[10, 100],
      num_participations=[1, 3, 5],
      # cycle_length=1 is plain random allocation; > 1 is the cyclic variant.
      cycle_length=[1, 2, 3],
      # 20 is a multiple of 2 but not of 3, and 21 is a multiple of 3 but not
      # of 2, so both the even and uneven cases are covered.
      iterations=[20, 21],
      partition_type=[
          batch_selection.PartitionType.INDEPENDENT,
          batch_selection.PartitionType.EQUAL_SPLIT,
      ],
  )
  def test_random_allocation_sampling(
      self,
      num_examples,
      num_participations,
      cycle_length,
      iterations,
      partition_type,
  ):
    """Tests random allocation participation counts and truncation."""
    strategy = batch_selection.RandomAllocationSampling(
        total_participations=num_participations,
        iterations=iterations,
        cycle_length=cycle_length,
        partition_type=partition_type,
    )
    batches = list(strategy.batch_iterator(num_examples, rng=0))

    self.assertLen(batches, iterations)
    _check_element_range(batches, num_examples)
    _check_signed_indices(batches)
    _check_max_participation(batches, num_participations)
    _check_subset_of_kb_participation(batches, cycle_length)
    counts = collections.Counter(int(x) for x in np.concatenate(batches))
    if iterations % cycle_length == 0:
      # Without truncation, each example appears in *exactly* k batches.
      for count in counts.values():
        self.assertEqual(count, num_participations)
      expected_examples = num_examples
      if partition_type == batch_selection.PartitionType.EQUAL_SPLIT:
        expected_examples = (num_examples // cycle_length) * cycle_length
      self.assertLen(counts, expected_examples)
    else:
      # With truncation, groups < iterations % cycle_length run all
      # ceil(iterations / cycle_length) cycles and participate exactly k times,
      # while truncated groups lose their last round and participate k-1 or k
      # times.
      for count in counts.values():
        self.assertIn(count, (num_participations - 1, num_participations))
      for step, batch in enumerate(batches):
        if step % cycle_length < iterations % cycle_length:
          for idx in batch:
            self.assertEqual(counts[int(idx)], num_participations)
    # Within each batch, no example should appear twice.
    for batch in batches:
      self.assertEqual(len(batch), len(set(batch.tolist())))

  @parameterized.parameters(1, 3, 5)
  def test_random_allocation_sampling_total_batch_size(self, cycle_length):
    """Total batch size across all iterations is exactly num_examples * k."""
    num_examples, num_participations, iterations = 1000, 3, 60
    strategy = batch_selection.RandomAllocationSampling(
        total_participations=num_participations,
        iterations=iterations,
        cycle_length=cycle_length,
    )
    batches = list(strategy.batch_iterator(num_examples, rng=0))
    # When iterations is a multiple of cycle_length, every example
    # participates in exactly k iterations, so the total number of
    # participations is deterministic even though its split across iterations is
    # random.
    self.assertEqual(
        sum(len(batch) for batch in batches), num_examples * num_participations
    )

  @parameterized.parameters(1, 3)
  def test_random_allocation_sampling_saturated(self, cycle_length):
    """Every eligible example participates when k == iterations // b."""
    num_examples, iterations = 10, 9
    strategy = batch_selection.RandomAllocationSampling(
        total_participations=iterations // cycle_length,
        iterations=iterations,
        cycle_length=cycle_length,
    )
    batches = list(strategy.batch_iterator(num_examples, rng=0))
    self.assertLen(batches, iterations)
    self.assertEqual(
        sum(len(batch) for batch in batches[:cycle_length]),
        num_examples,
    )
    _check_no_repeated_indices(batches[:cycle_length])
    _check_cyclic_property(batches, cycle_length)

  def test_random_allocation_sampling_k_equals_zero(self):
    """All batches should be empty when num_participations=0."""
    strategy = batch_selection.RandomAllocationSampling(
        total_participations=0,
        iterations=5,
    )
    batches = list(strategy.batch_iterator(10, rng=0))
    self.assertLen(batches, 5)
    for batch in batches:
      self.assertEmpty(batch)

  @parameterized.parameters(1, 3)
  def test_random_allocation_sampling_is_deterministic(self, cycle_length):
    """RandomAllocationSampling should respect the provided RNG."""
    strategy = batch_selection.RandomAllocationSampling(
        total_participations=2,
        iterations=10,
        cycle_length=cycle_length,
    )
    batches_a = list(strategy.batch_iterator(50, rng=0))
    batches_b = list(strategy.batch_iterator(50, rng=0))
    for batch_a, batch_b in zip(batches_a, batches_b, strict=True):
      np.testing.assert_array_equal(batch_a, batch_b)

  @parameterized.parameters(1, 3)
  def test_random_allocation_sampling_zero_examples(self, cycle_length):
    """Should produce empty batches when there are no examples."""
    strategy = batch_selection.RandomAllocationSampling(
        total_participations=2,
        iterations=6,
        cycle_length=cycle_length,
    )
    batches = list(strategy.batch_iterator(0, rng=0))
    self.assertLen(batches, 6)
    for batch in batches:
      self.assertEmpty(batch)

  @parameterized.parameters(
      # total_participations > ceil(iterations / cycle_length).
      {'total_participations': 11, 'iterations': 10, 'cycle_length': 1},
      {'total_participations': 5, 'iterations': 10, 'cycle_length': 3},
      {'total_participations': -1, 'iterations': 10, 'cycle_length': 1},
      {'total_participations': 1, 'iterations': 10, 'cycle_length': 0},
      {'total_participations': 1, 'iterations': 10, 'sampling_prob': -0.1},
      {'total_participations': 1, 'iterations': 10, 'sampling_prob': 1.1},
  )
  def test_random_allocation_sampling_validation(self, **kwargs):
    with self.assertRaises(ValueError):
      batch_selection.RandomAllocationSampling(**kwargs)

  def test_random_allocation_uneven_iterations_logs_warning(self):
    with self.assertLogs(level='WARNING') as logs:
      batch_selection.RandomAllocationSampling(
          total_participations=2, iterations=10, cycle_length=3
      )
    self.assertIn('multiple of cycle_length', logs.output[0])

  @parameterized.parameters(
      batch_selection.PartitionType.INDEPENDENT,
      batch_selection.PartitionType.EQUAL_SPLIT,
  )
  def test_random_allocation_subsampling(self, partition_type):
    b, k, t, q, n = 4, 8, 96, 0.5, 20000
    strategy = batch_selection.RandomAllocationSampling(
        k, t, cycle_length=b, partition_type=partition_type, sampling_prob=q
    )
    batches = list(strategy.batch_iterator(n, rng=0))
    _check_b_min_sep_property(batches, b)
    _check_subset_of_kb_participation(batches, b)
    counts = np.bincount(np.concatenate(batches), minlength=n)
    self.assertLessEqual(counts.max(), k)
    self.assertAlmostEqual(counts.mean(), k * q, delta=0.05)

  @parameterized.parameters(
      batch_selection.PartitionType.INDEPENDENT,
      batch_selection.PartitionType.EQUAL_SPLIT,
  )
  def test_random_allocation_matches_cyclic_poisson(self, partition_type):
    b, t, q, n = 4, 40, 0.5, 30000
    k = t // b
    ra = batch_selection.RandomAllocationSampling(
        k, t, cycle_length=b, partition_type=partition_type, sampling_prob=q
    )
    cyclic = batch_selection.CyclicPoissonSampling(
        q, t, cycle_length=b, partition_type=partition_type
    )

    def count_hist(strategy):
      indices = np.concatenate(list(strategy.batch_iterator(n, rng=0)))
      counts = np.bincount(indices, minlength=n)
      return np.bincount(counts, minlength=k + 1) / n

    np.testing.assert_allclose(count_hist(ra), count_hist(cyclic), atol=0.02)

  def test_cyclic_poisson_sampling_independent_is_deterministic(self):
    """CyclicPoissonSampling should respect the provided RNG."""
    strategy = batch_selection.CyclicPoissonSampling(
        sampling_prob=0.3,
        iterations=5,
        cycle_length=3,
        partition_type=batch_selection.PartitionType.INDEPENDENT,
    )
    batches_a = list(strategy.batch_iterator(50, rng=0))
    batches_b = list(strategy.batch_iterator(50, rng=0))
    for batch_a, batch_b in zip(batches_a, batches_b, strict=True):
      np.testing.assert_array_equal(batch_a, batch_b)

  @parameterized.product(
      num_examples=[0, 1, 100],
      sampling_prob=[0.1],
      iterations=[1000],
      min_sep=[2, 4],
      warm_start=[True, False],
      truncated_batch_size=[None, 4],
  )
  def test_b_min_sep_sampling(
      self,
      num_examples,
      sampling_prob,
      iterations,
      min_sep,
      warm_start,
      truncated_batch_size,
  ):
    """Tests for b-min-sep sampling."""
    strategy = batch_selection.BMinSepSampling(
        sampling_prob=sampling_prob,
        iterations=iterations,
        min_sep=min_sep,
        warm_start=warm_start,
        truncated_batch_size=truncated_batch_size,
    )
    batches = list(strategy.batch_iterator(num_examples, rng=0))
    self.assertLen(batches, iterations)
    _check_element_range(batches, num_examples)
    _check_b_min_sep_property(batches, min_sep)
    # 2 * iterations * sampling_prob is an overestimate of the tail bound on
    # participations as long as iterations * sampling_prob is large enough, but
    # will still catch errors such as e.g. sampling_prob being ignored.
    _check_max_participation(batches, 2 * iterations * sampling_prob)
    if truncated_batch_size:
      self.assertLessEqual(
          max(len(batch) for batch in batches), truncated_batch_size
      )

  @parameterized.product(
      num_examples=[1000],
      iterations=[10],
      warm_start=[True, False],
      truncated_batch_size=[None, 4],
  )
  def test_b_min_sep_sampling_prob_one(
      self, num_examples, iterations, warm_start, truncated_batch_size
  ):
    """Tests for b-min-sep sampling with sampling_prob = 1."""
    strategy = batch_selection.BMinSepSampling(
        sampling_prob=1.0,
        iterations=iterations,
        min_sep=2,
        warm_start=warm_start,
        truncated_batch_size=truncated_batch_size,
    )
    batches = list(strategy.batch_iterator(num_examples, rng=0))
    self.assertLen(batches, iterations)
    _check_element_range(batches, num_examples)
    if truncated_batch_size:
      self.assertLessEqual(
          max(len(batch) for batch in batches), truncated_batch_size
      )
    else:
      # Reduces to balls-in-bins.
      _check_cyclic_property(batches, 2)
      if not warm_start:
        # Without warm-start, first batch will have all examples.
        self.assertLen(batches[0], num_examples)
        self.assertEmpty(batches[1])
      else:
        # With warm-start, first two batches should each have about half the
        # examples.
        self.assertEqual(len(batches[0]) + len(batches[1]), num_examples)
        self.assertGreaterEqual(len(batches[0]), 400)
        self.assertLessEqual(len(batches[0]), 600)

  def test_b_min_sep_sampling_min_sep_one_warm_start(self):
    """min_sep=1 with default warm_start reduces to plain Poisson(p)."""
    sampling_prob = 0.01
    iterations = 5
    num_examples = 1000
    strategy = batch_selection.BMinSepSampling(
        sampling_prob=sampling_prob,
        iterations=iterations,
        min_sep=1,
    )
    # warm_start defaults to True; with min_sep == 1 (b == 1) there is no
    # separation constraint and the warm-start history must be empty.
    self.assertTrue(strategy.warm_start)
    batches = list(strategy.batch_iterator(num_examples, rng=0))
    self.assertLen(batches, iterations)
    _check_element_range(batches, num_examples)
    _check_signed_indices(batches)

  def test_b_min_sep_sampling_min_sep_one_poisson_frequency(self):
    """Over many iterations, min_sep=1 gives per-example frequency ~= p."""
    sampling_prob = 0.1
    iterations = 2000
    num_examples = 200
    strategy = batch_selection.BMinSepSampling(
        sampling_prob=sampling_prob,
        iterations=iterations,
        min_sep=1,
    )
    batches = list(strategy.batch_iterator(num_examples, rng=0))
    self.assertLen(batches, iterations)
    mean_inclusion_freq = sum(len(batch) for batch in batches) / (
        iterations * num_examples
    )
    self.assertAlmostEqual(mean_inclusion_freq, sampling_prob, delta=0.003)

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

  def test_fixed_batch_sampling(self):
    """Tests for FixedBatchSampling."""
    strategy = batch_selection.FixedBatchSampling(
        batch_size=4,
        iterations=6,
    )
    batches = list(strategy.batch_iterator(10, rng=0))
    self.assertLen(batches, 6)
    _check_batch_sizes_equal(batches, 4, 4)
    _check_element_range(batches, 10)
    _check_signed_indices(batches)
    for batch in batches:
      self.assertEqual(len(batch), len(set(batch)))

  def test_fixed_batch_sampling_rejects_oversized(self):
    strategy = batch_selection.FixedBatchSampling(
        batch_size=11,
        iterations=1,
    )
    with self.assertRaises(ValueError):
      list(strategy.batch_iterator(10, rng=0))

  def test_fixed_batch_sampling_with_replacement(self):
    strategy = batch_selection.FixedBatchSampling(
        batch_size=10,
        iterations=2,
        replace=True,
    )
    batches = list(strategy.batch_iterator(5, rng=0))
    self.assertLen(batches, 2)
    _check_batch_sizes_equal(batches, 10, 10)
    _check_element_range(batches, 5)


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

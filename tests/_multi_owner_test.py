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
from jax_privacy import _multi_owner
import numpy as np

MultiOwnerGraph = _multi_owner.MultiOwnerGraph
MultiOwnerMinSepSampling = _multi_owner.MultiOwnerMinSepSampling


def _check_slot_level_min_sep(batches, data, min_sep):
  """Checks batch-level min-sep for every user in multi-owner data."""
  ex_to_users = collections.defaultdict(set)
  for i in range(data.num_edges):
    ex_to_users[int(data.example_ids[i])].add(int(data.user_ids[i]))
  user_appearances = collections.defaultdict(list)
  for t, batch in enumerate(batches):
    for ex in batch:
      for u in ex_to_users[int(ex)]:
        user_appearances[u].append(t)
  for u, appearances in user_appearances.items():
    for i in range(len(appearances) - 1):
      diff = appearances[i + 1] - appearances[i]
      assert diff >= min_sep, (
          f'User {u} in batches {appearances[i]} and {appearances[i + 1]}, '
          f'separation {diff} < {min_sep}'
      )


def _get_user_participation_counts(batches, attribution):
  """Returns {user: number_of_batches_containing_that_user}."""
  ex_to_users = collections.defaultdict(set)
  for i in range(attribution.num_edges):
    ex_to_users[int(attribution.example_ids[i])].add(
        int(attribution.user_ids[i])
    )
  user_counts = collections.defaultdict(int)
  for batch in batches:
    users_in_batch = set()
    for ex in batch:
      for user in ex_to_users[int(ex)]:
        users_in_batch.add(user)
    for user in users_in_batch:
      user_counts[user] += 1
  return user_counts


class MultiOwnerGraphTest(parameterized.TestCase):

  def test_from_owners_per_example(self):
    data = MultiOwnerGraph.from_owners_per_example([[0, 1], [1, 2], [2]])
    self.assertEqual(data.num_examples, 3)
    self.assertEqual(data.num_users, 3)
    self.assertEqual(data.num_edges, 5)

  def test_from_user_to_examples(self):
    data = MultiOwnerGraph.from_user_to_examples({0: [0, 1], 1: [1, 2], 2: [2]})
    self.assertEqual(data.num_examples, 3)
    self.assertEqual(data.num_users, 3)
    self.assertEqual(data.num_edges, 5)

  def test_direct_construction(self):
    data = MultiOwnerGraph(
        example_ids=np.array([0, 0, 1, 1, 2]),
        user_ids=np.array([0, 1, 1, 2, 2]),
    )
    self.assertEqual(data.num_examples, 3)
    self.assertEqual(data.num_users, 3)
    self.assertEqual(data.num_edges, 5)

  def test_shape_mismatch_raises(self):
    with self.assertRaises(ValueError):
      MultiOwnerGraph(example_ids=np.array([0, 1]), user_ids=np.array([0]))

  def test_non_1d_raises(self):
    with self.assertRaises(ValueError):
      MultiOwnerGraph(
          example_ids=np.array([[0, 1]]), user_ids=np.array([[0, 1]])
      )

  def test_from_owners_per_example_renumbers_user_ids(self):
    data = MultiOwnerGraph.from_owners_per_example([[100, 200], [200, 300]])
    # User IDs should be renumbered to [0, 3).
    self.assertEqual(data.num_users, 3)
    self.assertTrue(np.all(data.user_ids >= 0))
    self.assertTrue(np.all(data.user_ids < 3))

  def test_csr_property(self):
    # Example 0 -> users {0, 1}, Example 1 -> users {1, 2}, Example 2 -> {2}.
    data = MultiOwnerGraph(
        example_ids=np.array([0, 0, 1, 1, 2]),
        user_ids=np.array([0, 1, 1, 2, 2]),
    )
    csr = data.csr
    np.testing.assert_array_equal(csr.degree, [2, 2, 1])
    np.testing.assert_array_equal(csr.indptr, [0, 2, 4, 5])
    # users_of should return the correct user slice for each example.
    np.testing.assert_array_equal(sorted(csr.users_of(0)), [0, 1])
    np.testing.assert_array_equal(sorted(csr.users_of(1)), [1, 2])
    np.testing.assert_array_equal(csr.users_of(2), [2])

  def test_from_owners_per_example_unowned_raises(self):
    """An example with no owners is almost certainly a bug."""
    with self.assertRaises(ValueError):
      MultiOwnerGraph.from_owners_per_example([[0, 1], [], [2]])

  def test_from_user_to_examples_empty_user_skipped(self):
    """Users mapped to empty example lists are silently ignored."""
    data = MultiOwnerGraph.from_user_to_examples({0: [0, 1], 1: [], 2: [1, 2]})
    # User 1 has no examples, so only users 0 and 2 contribute.
    self.assertEqual(data.num_users, 2)
    self.assertEqual(data.num_examples, 3)

  def test_from_owners_per_example_noncontiguous_user_ids(self):
    """Large, non-contiguous raw user IDs are renumbered correctly."""
    data = MultiOwnerGraph.from_owners_per_example(
        [[1000, 5000], [5000, 9999], [9999]]
    )
    self.assertEqual(data.num_users, 3)
    self.assertEqual(data.num_examples, 3)
    self.assertTrue(np.all(data.user_ids >= 0))
    self.assertTrue(np.all(data.user_ids < 3))

  def test_from_user_to_examples_noncontiguous_ids(self):
    """Large, non-contiguous raw IDs are renumbered correctly."""
    data = MultiOwnerGraph.from_user_to_examples(
        {42: [100, 200], 99: [200, 300]}
    )
    self.assertEqual(data.num_users, 2)
    self.assertEqual(data.num_examples, 3)
    self.assertTrue(np.all(data.example_ids >= 0))
    self.assertTrue(np.all(data.example_ids < 3))
    self.assertTrue(np.all(data.user_ids >= 0))
    self.assertTrue(np.all(data.user_ids < 2))

  def test_duplicate_edges_rejected(self):
    """Duplicate (example, user) edges raise ValueError."""
    with self.assertRaises(ValueError):
      MultiOwnerGraph(example_ids=np.array([0, 0]), user_ids=np.array([1, 1]))


class GreedyContributionBoundTest(parameterized.TestCase):

  def _verify_no_user_repeated(self, attribution, selected):
    """Checks that no user appears more than once among selected examples."""
    sel_mask = np.zeros(attribution.num_examples, dtype=bool)
    sel_mask[selected] = True
    edge_sel = sel_mask[attribution.example_ids]
    counts = np.bincount(
        attribution.user_ids[edge_sel], minlength=attribution.num_users
    )
    np.testing.assert_array_less(counts, 2, err_msg='User appeared twice')

  def test_independent_set(self):
    """The result is an independent set (no user appears twice)."""
    data = MultiOwnerGraph.from_owners_per_example([[0, 1], [1, 2], [0, 2]])
    selected = _multi_owner.greedy_contribution_bound(data)
    self._verify_no_user_repeated(data, selected)
    self.assertNotEmpty(selected)

  def test_pentagon_graph(self):
    """Pentagon graph: each example shares users with neighbors."""
    data = MultiOwnerGraph.from_owners_per_example(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
    )
    selected = _multi_owner.greedy_contribution_bound(data)
    self._verify_no_user_repeated(data, selected)

  def test_single_owner_examples(self):
    """Single-owner examples: each user has 3 examples, only 1 selected."""
    data = MultiOwnerGraph.from_owners_per_example(
        [[0], [0], [0], [1], [1], [1]]
    )
    selected = _multi_owner.greedy_contribution_bound(data)
    self._verify_no_user_repeated(data, selected)
    # With k=1, exactly 1 per user.
    self.assertLen(selected, 2)

  def test_mixed_single_and_multi_owner(self):
    """Mix of single and multi-owner examples."""
    data = MultiOwnerGraph.from_owners_per_example(
        [[0], [1], [0, 1], [2], [2, 0]]
    )
    selected = _multi_owner.greedy_contribution_bound(data)
    self._verify_no_user_repeated(data, selected)

  def test_disjoint_users_selects_all(self):
    """If no user is shared, all examples should be selected."""
    data = MultiOwnerGraph.from_owners_per_example([[0], [1], [2]])
    selected = _multi_owner.greedy_contribution_bound(data)
    self.assertLen(selected, 3)

  def test_max_degree_boundary(self):
    """Verifies that max_degree correctly thresholds example selection."""
    data = MultiOwnerGraph.from_owners_per_example([[0], [1], [2, 3], [4, 5]])

    selected_1 = _multi_owner.greedy_contribution_bound(data, max_degree=1)
    np.testing.assert_array_equal(sorted(selected_1), [0, 1])

    selected_2 = _multi_owner.greedy_contribution_bound(data, max_degree=2)
    np.testing.assert_array_equal(sorted(selected_2), [0, 1, 2, 3])

  @parameterized.parameters(
      # (n_examples, n_users, max_owners, max_degree, seed)
      (50, 20, 2, 1, 42),
      (50, 20, 2, 2, 42),
      (100, 30, 3, 1, 0),
      (100, 30, 3, 2, 0),
      (100, 30, 3, 3, 0),
      (200, 50, 5, 1, 7),
      (200, 50, 5, 3, 7),
      (200, 50, 5, 5, 7),
      (100, 100, 4, 3, 123),
      (500, 100, 2, 1, 99),
      (500, 100, 2, 2, 99),
  )
  def test_random_graph_properties(
      self, n_examples, n_users, max_owners, max_degree, seed
  ):
    """Verifies constraint satisfaction, max_degree, and local optimality."""
    data = _create_random_graph(n_examples, n_users, max_owners, seed)
    selected = _multi_owner.greedy_contribution_bound(
        data, max_degree=max_degree
    )

    # 1. Constraint satisfaction: no user is repeated.
    self._verify_no_user_repeated(data, selected)

    # Determine users that are owned by selected examples.
    sel_mask = np.zeros(data.num_examples, dtype=bool)
    sel_mask[selected] = True
    selected_users = np.unique(data.user_ids[sel_mask[data.example_ids]])

    # 2. Constraint satisfaction: selected examples satisfy max_degree.
    for ex in selected:
      self.assertLessEqual(
          len(data.csr.users_of(ex)),
          max_degree,
          msg=f'Selected example {ex} has degree > {max_degree}',
      )

    # 3. Local optimality: no example can be added to the selection.
    for ex in range(data.num_examples):
      users = data.csr.users_of(ex)
      if len(users) <= max_degree:
        if not np.any(np.isin(users, selected_users)):
          self.assertIn(
              ex,
              selected,
              msg=f'Example {ex} could have been selected but was not.',
          )


class MultiOwnerMinSepSamplingTest(parameterized.TestCase):

  def test_basic(self):
    data = MultiOwnerGraph.from_owners_per_example(
        [[2 * i, 2 * i + 1] for i in range(15)]
    )
    strategy = MultiOwnerMinSepSampling(
        attribution=data,
        batch_size=3,
        iterations=5,
        min_sep=2,
    )
    batches = list(strategy.batch_iterator(data.num_examples))
    self.assertLen(batches, 5)
    for batch in batches:
      self.assertLessEqual(len(batch), 3)
    _check_slot_level_min_sep(batches, data, 2)

  def test_single_owner_only(self):
    data = MultiOwnerGraph.from_owners_per_example([[i] for i in range(20)])
    strategy = MultiOwnerMinSepSampling(
        attribution=data,
        batch_size=4,
        iterations=5,
        min_sep=2,
    )
    batches = list(strategy.batch_iterator(data.num_examples))
    self.assertLen(batches, 5)
    _check_slot_level_min_sep(batches, data, 2)

  def test_min_sep_one(self):
    """min_sep=1 allows user in consecutive batches."""
    data = MultiOwnerGraph.from_owners_per_example([[i] for i in range(6)])
    strategy = MultiOwnerMinSepSampling(
        attribution=data,
        batch_size=2,
        iterations=3,
        min_sep=1,
    )
    batches = list(strategy.batch_iterator(data.num_examples))
    self.assertLen(batches, 3)
    _check_slot_level_min_sep(batches, data, 1)

  @parameterized.parameters(2, 3, 5)
  def test_various_min_sep(self, min_sep):
    data = MultiOwnerGraph.from_owners_per_example([[i] for i in range(50)])
    strategy = MultiOwnerMinSepSampling(
        attribution=data,
        batch_size=5,
        iterations=10,
        min_sep=min_sep,
    )
    batches = list(strategy.batch_iterator(data.num_examples))
    self.assertLen(batches, 10)
    _check_slot_level_min_sep(batches, data, min_sep)

  def test_valid_indices(self):
    data = MultiOwnerGraph.from_owners_per_example(
        [[0, 1], [2, 3], [4, 5], [6, 7]]
    )
    strategy = MultiOwnerMinSepSampling(
        attribution=data,
        batch_size=2,
        iterations=2,
        min_sep=2,
    )
    batches = list(strategy.batch_iterator(data.num_examples))
    for batch in batches:
      for ex in batch:
        self.assertGreaterEqual(ex, 0)
        self.assertLess(ex, data.num_examples)

  def test_zero_iterations(self):
    data = MultiOwnerGraph.from_owners_per_example([[0], [1]])
    with self.assertRaises(ValueError):
      MultiOwnerMinSepSampling(
          attribution=data,
          batch_size=2,
          iterations=0,
          min_sep=1,
      )

  def test_invalid_params(self):
    data = MultiOwnerGraph.from_owners_per_example([[0], [1]])
    with self.assertRaises(ValueError):
      MultiOwnerMinSepSampling(
          attribution=data, batch_size=2, iterations=3, min_sep=0
      )
    with self.assertRaises(ValueError):
      MultiOwnerMinSepSampling(
          attribution=data, batch_size=0, iterations=3, min_sep=1
      )
    with self.assertRaises(ValueError):
      MultiOwnerMinSepSampling(
          attribution=data, batch_size=2, iterations=-1, min_sep=1
      )

  def test_batch_count_equals_iterations(self):
    data = MultiOwnerGraph.from_owners_per_example([[i] for i in range(20)])
    for iters in [1, 5, 10]:
      strategy = MultiOwnerMinSepSampling(
          attribution=data,
          batch_size=2,
          iterations=iters,
          min_sep=1,
      )
      batches = list(strategy.batch_iterator(data.num_examples))
      self.assertLen(batches, iters)

  @parameterized.parameters(
      (200, 100, 3, 3, 5, 2),
      (400, 200, 4, 5, 8, 3),
      (500, 300, 2, 10, 10, 4),
  )
  def test_random_multi_owner(
      self, n_ex, n_users, max_owners, bs, iters, min_sep
  ):
    rng = np.random.default_rng(0)
    data = _random_multi_owner_graph(rng, n_ex, n_users, max_owners)
    strategy = MultiOwnerMinSepSampling(
        attribution=data,
        batch_size=bs,
        iterations=iters,
        min_sep=min_sep,
    )
    batches = list(strategy.batch_iterator(data.num_examples))
    self.assertLen(batches, iters)
    for batch in batches:
      self.assertLessEqual(len(batch), bs)
    _check_slot_level_min_sep(batches, data, min_sep)

  def test_deterministic_across_calls(self):
    """Same input always produces the same output."""
    data = _random_multi_owner_graph(np.random.default_rng(42), 300, 200, 2.0)
    strategy = MultiOwnerMinSepSampling(
        attribution=data,
        batch_size=5,
        iterations=10,
        min_sep=2,
    )
    batches1 = [b.copy() for b in strategy.batch_iterator(data.num_examples)]
    batches2 = list(strategy.batch_iterator(data.num_examples))
    for b1, b2 in zip(batches1, batches2):
      np.testing.assert_array_equal(b1, b2)

  def test_user_maxpart_matches_actual(self):
    """user_maxpart equals the actual maximum user participation count."""
    data = MultiOwnerGraph.from_owners_per_example([[i] for i in range(100)])
    strategy = MultiOwnerMinSepSampling(
        attribution=data,
        batch_size=5,
        iterations=20,
        min_sep=2,
    )
    batches = list(strategy.batch_iterator(data.num_examples))
    user_counts = _get_user_participation_counts(batches, data)
    actual_max = max(user_counts.values())
    self.assertEqual(strategy.user_maxpart, actual_max)
    _check_slot_level_min_sep(batches, data, 2)

  def test_example_maxpart_computed(self):
    """example_maxpart reflects actual maximum example participation."""
    data = MultiOwnerGraph.from_owners_per_example([[i] for i in range(50)])
    strategy = MultiOwnerMinSepSampling(
        attribution=data,
        batch_size=5,
        iterations=10,
        min_sep=2,
    )
    batches = list(strategy.batch_iterator(data.num_examples))
    all_selected = np.concatenate(batches)
    _, counts = np.unique(all_selected, return_counts=True)
    self.assertEqual(strategy.example_maxpart, int(counts.max()))
    _check_slot_level_min_sep(batches, data, 2)

  def test_shuffle_produces_different_results(self):
    """shuffle_seed gives different assignments but still satisfies min-sep."""
    data = _random_multi_owner_graph(np.random.default_rng(0), 500, 300, 2)
    strategy_no_shuffle = MultiOwnerMinSepSampling(
        attribution=data,
        batch_size=10,
        iterations=10,
        min_sep=2,
    )
    strategy_shuffle = MultiOwnerMinSepSampling(
        attribution=data,
        batch_size=10,
        iterations=10,
        min_sep=2,
        shuffle_seed=42,
    )
    batches_no = list(strategy_no_shuffle.batch_iterator(data.num_examples))
    batches_yes = list(strategy_shuffle.batch_iterator(data.num_examples))
    # Both should satisfy min-sep.
    _check_slot_level_min_sep(batches_no, data, 2)
    _check_slot_level_min_sep(batches_yes, data, 2)
    # Results should differ.
    any_different = any(
        not np.array_equal(a, b) for a, b in zip(batches_no, batches_yes)
    )
    self.assertTrue(any_different, 'Shuffle did not change any batch.')

  def test_user_example_ratio(self):
    """user_example_ratio limits example duplication relative to user cap."""
    data = MultiOwnerGraph.from_owners_per_example([[i] for i in range(100)])
    strategy = MultiOwnerMinSepSampling(
        attribution=data,
        batch_size=5,
        iterations=20,
        min_sep=2,
        user_example_ratio=2,
    )
    batches = list(strategy.batch_iterator(data.num_examples))
    self.assertLessEqual(
        strategy.example_maxpart,
        math.ceil(strategy.user_maxpart / 2),
    )
    _check_slot_level_min_sep(batches, data, 2)

  def test_high_ratio_limits_example_to_one(self):
    """A high user_example_ratio forces each example to appear at most once."""
    data = MultiOwnerGraph.from_owners_per_example([[i] for i in range(50)])
    strategy = MultiOwnerMinSepSampling(
        attribution=data,
        batch_size=5,
        iterations=10,
        min_sep=2,
        user_example_ratio=100,
    )
    batches = list(strategy.batch_iterator(data.num_examples))
    self.assertEqual(strategy.example_maxpart, 1)
    _check_slot_level_min_sep(batches, data, 2)

  @parameterized.parameters(
      # (batch_size, iterations, min_sep); 12 examples, so every case asks for
      # more slots than there are examples.
      (2, 10, 1),
      (2, 10, 2),
      (3, 8, 2),
      (4, 4, 2),
  )
  def test_more_slots_than_examples(self, batch_size, iterations, min_sep):
    """Examples are reused when the schedule is longer than the dataset."""
    data = MultiOwnerGraph.from_owners_per_example([[i] for i in range(12)])
    total_slots = batch_size * iterations
    self.assertGreater(total_slots, data.num_examples)
    strategy = MultiOwnerMinSepSampling(
        attribution=data,
        batch_size=batch_size,
        iterations=iterations,
        min_sep=min_sep,
    )
    batches = list(strategy.batch_iterator(data.num_examples))
    self.assertLen(batches, iterations)
    self.assertEqual(sum(len(batch) for batch in batches), total_slots)
    self.assertGreater(strategy.example_maxpart, 1)
    _check_slot_level_min_sep(batches, data, min_sep)
    # Ganesh et al. (2025) bound the post-hoc contribution bound of their
    # Algorithm 4 by ceil(T / b).
    self.assertLessEqual(strategy.user_maxpart, math.ceil(iterations / min_sep))

  def test_more_slots_than_examples_multi_owner(self):
    """Example reuse also works when examples have several owners."""
    data = MultiOwnerGraph.from_owners_per_example(
        [[2 * i, 2 * i + 1] for i in range(10)]
    )
    strategy = MultiOwnerMinSepSampling(
        attribution=data,
        batch_size=4,
        iterations=9,
        min_sep=2,
    )
    batches = list(strategy.batch_iterator(data.num_examples))
    self.assertLen(batches, 9)
    self.assertEqual(sum(len(batch) for batch in batches), 36)
    _check_slot_level_min_sep(batches, data, 2)

  def test_no_example_repeated_within_a_batch(self):
    """min_sep >= 1 forbids the same example twice in one batch."""
    data = MultiOwnerGraph.from_owners_per_example([[i] for i in range(12)])
    strategy = MultiOwnerMinSepSampling(
        attribution=data,
        batch_size=3,
        iterations=10,
        min_sep=1,
    )
    for batch in strategy.batch_iterator(data.num_examples):
      self.assertLen(set(batch.tolist()), len(batch))

  # 255 and 256 straddle the uint8 / uint16 boundary of the reuse counter.
  @parameterized.parameters(255, 256, 1000)
  def test_heavy_reuse_does_not_overflow_counters(self, iterations):
    """Reuse counts are not limited by the dtype used for example indices."""
    # Three examples fit in an int8 index, but each one is used in every batch.
    data = MultiOwnerGraph.from_owners_per_example([[i] for i in range(3)])
    strategy = MultiOwnerMinSepSampling(
        attribution=data,
        batch_size=3,
        iterations=iterations,
        min_sep=1,
    )
    batches = list(strategy.batch_iterator(data.num_examples))
    _, counts = np.unique(np.concatenate(batches), return_counts=True)
    self.assertEqual(int(counts.max()), iterations)
    self.assertEqual(strategy.example_maxpart, iterations)
    _check_slot_level_min_sep(batches, data, 1)

  def test_examples_without_users_are_used_once(self):
    """Min-sep cannot space out an example with no users, so it isn't reused."""
    # Examples 1 and 4 have no users (example_ids has gaps).
    data = MultiOwnerGraph(
        example_ids=np.array([0, 2, 3, 5, 6, 7]),
        user_ids=np.array([0, 1, 2, 3, 4, 5]),
    )
    strategy = MultiOwnerMinSepSampling(
        attribution=data,
        batch_size=4,
        iterations=8,
        min_sep=1,
    )
    batches = list(strategy.batch_iterator(data.num_examples))
    for batch in batches:
      self.assertLen(set(batch.tolist()), len(batch))
    all_selected = np.concatenate(batches)
    self.assertLessEqual(int(np.sum(all_selected == 1)), 1)
    self.assertLessEqual(int(np.sum(all_selected == 4)), 1)
    _check_slot_level_min_sep(batches, data, 1)

    # Here the three users can fill at most 13 of 30 slots, so the strategy
    # must fail rather than pad batches by repeating the example without users.
    data = MultiOwnerGraph(
        example_ids=np.array([0, 2, 3]), user_ids=np.array([0, 1, 2])
    )
    with self.assertRaisesRegex(ValueError, 'Could not fill'):
      MultiOwnerMinSepSampling(
          attribution=data,
          batch_size=3,
          iterations=10,
          min_sep=3,
      )

  def test_example_maxpart_matches_actual_with_reuse(self):
    """example_maxpart is exact when examples are used more than once."""
    data = MultiOwnerGraph.from_owners_per_example([[i] for i in range(20)])
    strategy = MultiOwnerMinSepSampling(
        attribution=data,
        batch_size=4,
        iterations=15,
        min_sep=2,
    )
    batches = list(strategy.batch_iterator(data.num_examples))
    _, counts = np.unique(np.concatenate(batches), return_counts=True)
    self.assertGreater(int(counts.max()), 1)
    self.assertEqual(strategy.example_maxpart, int(counts.max()))
    user_counts = _get_user_participation_counts(batches, data)
    self.assertEqual(strategy.user_maxpart, max(user_counts.values()))
    _check_slot_level_min_sep(batches, data, 2)

  def test_infeasible_raises(self):
    """ValueError when the greedy algorithm cannot fill all slots."""
    # 2 examples owned by the same user: with min_sep=2 and batch_size=1,
    # only 2 of the 10 requested slots can be filled.
    data = MultiOwnerGraph.from_owners_per_example([[0], [0]])
    with self.assertRaisesRegex(ValueError, 'Try reducing min_sep'):
      MultiOwnerMinSepSampling(
          attribution=data,
          batch_size=1,
          iterations=10,
          min_sep=2,
      )

  def test_infeasible_multi_owner_raises(self):
    """ValueError for multi-owner graph where min_sep makes it infeasible."""
    # All examples share user 0, so at most 1 example per min_sep window.
    data = MultiOwnerGraph.from_owners_per_example([[0, 1], [0, 2], [0, 3]])
    with self.assertRaisesRegex(ValueError, 'Try reducing min_sep'):
      MultiOwnerMinSepSampling(
          attribution=data,
          batch_size=2,
          iterations=10,
          min_sep=3,
      )


def _create_random_graph(n_examples, n_users, max_owners, seed):
  """Generates a random MultiOwnerGraph for testing."""
  rng = np.random.RandomState(seed)
  owners_per_example = []
  for _ in range(n_examples):
    n_owners = rng.randint(1, max_owners + 1)
    owners = rng.choice(n_users, size=n_owners, replace=False)
    owners_per_example.append(owners.tolist())
  return MultiOwnerGraph.from_owners_per_example(owners_per_example)


def _random_multi_owner_graph(rng, n_examples, n_users, max_owners):
  """Generates a random MultiOwnerGraph using a numpy Generator."""
  owners_per_example = []
  for _ in range(n_examples):
    n_owners = rng.integers(1, int(max_owners) + 1)
    owners = rng.choice(n_users, size=n_owners, replace=False)
    owners_per_example.append(owners.tolist())
  return MultiOwnerGraph.from_owners_per_example(owners_per_example)


if __name__ == '__main__':
  absltest.main()

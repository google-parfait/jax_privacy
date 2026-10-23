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

"""Multi-owner differential privacy support for batch selection.

Provides data structures and algorithms for the multi-owner model of
differential privacy, where examples may be attributed to multiple users.

References:
  * `Ganesh et. al (2025) <https://arxiv.org/abs/2503.03622>`_
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import dataclasses
import functools
import math

import numpy as np

from . import _validate
from . import batch_selection


@dataclasses.dataclass(frozen=True)
class MultiOwnerGraph:
  """Bipartite edge list of (example, user) attribution pairs.

  Entry ``(example_ids[i], user_ids[i])`` means that example
  ``example_ids[i]`` is attributed to user ``user_ids[i]``.

  The attribution graph is considered **public information**: it is not
  protected by differential privacy and may be used freely by the batch
  selection algorithm.

  For smaller datasets, use ``from_owners_per_example`` or
  ``from_user_to_examples``.

  Example Usage:
    >>> data = MultiOwnerGraph.from_owners_per_example([[0, 1], [1, 2], [2]])
    >>> data.num_examples
    3
    >>> data.num_users
    3
    >>> data.num_edges
    5

  References:
    * `Ganesh et al. (2025) <https://arxiv.org/abs/2503.03622>`_

  Attributes:
    example_ids: 1D integer array of example indices in ``[0, num_examples)``.
    user_ids: 1D integer array of user indices in ``[0, num_users)``, aligned
      with ``example_ids``.
    validate: If ``False``, skip edge-list validation. Use for large graphs.
  """

  example_ids: np.ndarray
  user_ids: np.ndarray
  validate: bool = True

  def __post_init__(self):
    if self.validate:
      _validate.multi_owner(self.example_ids, self.user_ids)

  @property
  def num_examples(self) -> int:
    """Number of distinct examples (``max(example_ids) + 1``)."""
    return int(self.example_ids.max()) + 1

  @property
  def num_users(self) -> int:
    """Number of distinct users (``max(user_ids) + 1``)."""
    return int(self.user_ids.max()) + 1

  @property
  def num_edges(self) -> int:
    """Total number of (example, user) attribution edges."""
    return len(self.example_ids)

  @classmethod
  def from_owners_per_example(
      cls, owners_per_example: Sequence[np.typing.ArrayLike]
  ) -> MultiOwnerGraph:
    """Constructs from a sequence of user sets, one per example.

    Example IDs are assigned sequentially (0, 1, ...). User IDs are
    renumbered to ``[0, num_users)``.

    Args:
      owners_per_example: ``owners_per_example[i]`` is the array-like of user
        IDs attributed to example ``i``. Every example must have at least one
        owner.

    Returns:
      A new MultiOwnerGraph instance.

    Raises:
      ValueError: If any example has zero owners.
    """
    arrays = [np.asarray(o, dtype=np.int64) for o in owners_per_example]
    sizes = np.array([len(a) for a in arrays])
    if len(sizes) > 0 and int(sizes.min()) == 0:
      raise ValueError(
          'Every example must have at least one owner. '
          f'Example(s) at index {np.where(sizes == 0)[0].tolist()} have none.'
      )
    example_ids = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)
    user_ids = np.concatenate(arrays)
    _, user_ids = np.unique(user_ids, return_inverse=True)
    return cls(example_ids=example_ids, user_ids=user_ids)

  @classmethod
  def from_user_to_examples(
      cls, user_to_examples: Mapping[int, np.typing.ArrayLike]
  ) -> MultiOwnerGraph:
    """Constructs from a mapping of user IDs to their example sets.

    Both example and user IDs are renumbered to contiguous ranges.
    Users mapped to empty example lists are silently skipped.

    Args:
      user_to_examples: ``user_to_examples[u]`` is the array-like of example IDs
        attributed to user ``u``.

    Returns:
      A new MultiOwnerGraph instance with renumbered IDs.
    """
    ex_arrays, u_arrays = [], []
    for uid, examples in user_to_examples.items():
      ex_arr = np.asarray(examples, dtype=np.int64)
      if len(ex_arr) == 0:
        continue
      ex_arrays.append(ex_arr)
      u_arrays.append(np.full(len(ex_arr), uid, dtype=np.int64))
    if not ex_arrays:
      return cls(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    example_ids = np.concatenate(ex_arrays)
    user_ids = np.concatenate(u_arrays)
    _, example_ids = np.unique(example_ids, return_inverse=True)
    _, user_ids = np.unique(user_ids, return_inverse=True)
    return cls(example_ids=example_ids, user_ids=user_ids)

  @functools.cached_property
  def csr(self) -> _ExampleUserCSR:
    """Compressed sparse row view for per-example user lookup."""
    n = self.num_examples
    degree = np.bincount(self.example_ids, minlength=n)
    order = np.argsort(self.example_ids, kind='stable')
    users = self.user_ids[order]
    indptr = np.empty(n + 1, dtype=np.intp)
    indptr[0] = 0
    np.cumsum(degree, out=indptr[1:])
    return _ExampleUserCSR(degree=degree, indptr=indptr, sorted_users=users)


@dataclasses.dataclass(frozen=True)
class _ExampleUserCSR:
  """Compressed sparse row layout indexing users by example.

  Stores the user adjacency list for each example in a flat array
  (``sorted_users``) with ``indptr`` marking where each example's users
  begin and end, i.e. the users of example ``e`` are
  ``sorted_users[indptr[e]:indptr[e+1]]``.

  Attributes:
    degree: Number of users per example, shape ``(num_examples,)``.
    indptr: Index pointers into ``sorted_users``, shape ``(num_examples + 1,)``.
    sorted_users: User IDs ordered by example.
  """

  degree: np.ndarray
  indptr: np.ndarray
  sorted_users: np.ndarray

  def users_of(self, ex: int) -> np.ndarray:
    """Returns the user IDs attributed to example ``ex``."""
    return self.sorted_users[self.indptr[ex] : self.indptr[ex + 1]]


def greedy_contribution_bound(
    attribution: MultiOwnerGraph,
    *,
    max_degree: int = 8,
) -> np.ndarray:
  """Selects examples so each user appears at most once.

  Specializes Algorithm 2 of [1] to k=1, processing examples in ascending
  degree order.  Because each user is attributed to at most one selected
  example, the result induces a user-to-example mapping that can be composed
  with any example-level batch selection strategy.


  References:
    * [1] `Ganesh et al. (2025) <https://arxiv.org/abs/2503.03622>`_

  Args:
    attribution: Multi-owner attribution data.
    max_degree: Maximum number of owners per example to consider. Must be >= 1.

  Returns:
    1D array of selected example IDs.
  """
  _validate.positive(max_degree=max_degree)
  n_ex = attribution.num_examples
  degree = attribution.csr.degree
  ex_ids = attribution.example_ids
  u_ids = attribution.user_ids

  # Priority key: (degree, example_id). Lower selects first.
  priority = degree[ex_ids].astype(np.int64) * n_ex + ex_ids
  user_active = np.ones(attribution.num_users, dtype=bool)
  best = np.empty(attribution.num_users, dtype=np.int64)
  selected = []
  low_degree = degree <= max_degree

  # Round-based vectorized greedy: each round, every unclaimed user votes for
  # its lowest-priority example, and examples with unanimous votes are selected.
  for _ in range(max_degree):
    # An example is selectable if all of it's users are eligible to participate.
    active_counts = np.bincount(ex_ids[user_active[u_ids]], minlength=n_ex)
    selectable = (active_counts == degree) & low_degree
    active = selectable[ex_ids]
    ae, au, ap = ex_ids[active], u_ids[active], priority[active]

    # Each user votes for their lowest-priority active example.
    best.fill(np.iinfo(np.int64).max)
    np.minimum.at(best, au, ap)

    # An example is selected iff every one of its users voted for it.
    vote_count = np.bincount(ae[best[au] % n_ex == ae], minlength=n_ex)
    new = np.where(vote_count == degree)[0]

    new_edge_mask = np.isin(ex_ids, new)
    user_active[u_ids[new_edge_mask]] = False
    selected.append(new)

  return np.concatenate(selected)


class MultiOwnerMinSepSampling(batch_selection.BatchSelectionStrategy):
  """Batch selection with b-min-sep for multi-owner attributed data.

  WARNING: Initializing this class kicks off an example ordering greedy
  algorithm which can be time consuming.

  Note that both the attribution graph and the resulting batch assignments are
  considered **public information** and are not protected by differential
  privacy. The example ordering can be accessed directly via the
  ``example_order``
  field of the dataclass, which is precomputed at initialization.
  As a result, the max_part values are not specified as inputs, but
  rather computed post-hoc after identifying a valid example ordering.

  Formal Guarantees:

  - There is at most one example from each user across any ``min_sep``
    consecutive batches.
  - No user appears in more than ``user_maxpart`` batches (computed post-hoc).
  - No example appears in more than ``example_maxpart`` batches (computed
    post-hoc).

  References:
    * [1] `Ganesh et al. (2025) <https://arxiv.org/abs/2503.03622>`_
  """

  def __init__(
      self,
      attribution: MultiOwnerGraph,
      batch_size: int,
      iterations: int,
      min_sep: int,
      user_example_ratio: float = 4,
      shuffle_seed: int | None = None,
  ):
    """Initializes the MultiOwnerMinSepSampling strategy.

    Args:
      attribution: Multi-owner attribution data for the dataset.
      batch_size: Number of examples per batch.
      iterations: Number of batches to produce.
      min_sep: Min batch separation between same-user appearances.
      user_example_ratio: Ratio of per-user to per-example participation cap.
        Controls per-example duplication relative to the per-user cap. Both caps
        start at 1 and grow dynamically: when progress stalls due to cap
        constraints, ``user_maxpart_cap`` increases by 1 and
        ``example_maxpart_cap`` is recomputed as ``ceil(user_maxpart_cap /
        user_example_ratio)``.
      shuffle_seed: Seed for shuffling within degree classes each pass. ``None``
        (default) disables shuffling.
    """
    _validate.positive(
        min_sep=min_sep,
        batch_size=batch_size,
        iterations=iterations,
        user_example_ratio=user_example_ratio,
    )
    self._batch_size = batch_size
    self._iterations = iterations
    self._min_sep = min_sep
    self._user_example_ratio = user_example_ratio
    self._num_examples = attribution.num_examples
    example_order, user_maxpart, example_maxpart = self._precompute_order(
        attribution, shuffle_seed
    )
    self._example_order = example_order
    self._user_maxpart = user_maxpart
    self._example_maxpart = example_maxpart

  @property
  def example_maxpart(self) -> int:
    """Maximum number of batches an example appears in."""
    return self._example_maxpart

  @property
  def user_maxpart(self) -> int:
    """Maximum number of batches a user appears in."""
    return self._user_maxpart

  def _precompute_order(self, graph, seed):
    """Runs the greedy algorithm once, caching batches and maxpart stats."""
    num_examples = graph.num_examples
    total_slots = self._batch_size * self._iterations
    dtype = np.min_scalar_type(-num_examples)
    user_count = np.zeros(graph.num_users, dtype=np.int32)
    # Initialize last_batch so that every user is eligible in the first batch:
    # current_batch(0) - last_batch(-min_sep) = min_sep >= min_sep.
    last_batch = np.full(graph.num_users, -self._min_sep, dtype=np.int32)
    # ``dtype`` is sized for example *indices*, not participation counts. Each
    # selection of an example stamps ``last_batch`` for all of its users, so
    # repeat selections are at least ``min_sep`` batches apart and an example
    # is selected at most ceil(iterations / min_sep) times (examples without
    # users are selected at most once, see ``_greedy_pass``). Size the counter
    # from that bound: it is one byte per example in typical settings.
    max_example_count = -(-self._iterations // self._min_sep)
    example_count = np.zeros(
        num_examples, dtype=np.min_scalar_type(max_example_count)
    )
    slots: list[int] = []

    user_maxpart_cap = 1
    ex_maxpart_cap = 1
    rng = np.random.default_rng(seed)
    eligible = np.argsort(graph.csr.degree, kind='stable')
    # Examples set aside by the caps, accumulated across passes so that none is
    # lost: a pass that makes progress still reports cap-blocked examples, and
    # they must stay available for when the caps are next relaxed.
    capped = np.empty(0, dtype=eligible.dtype)
    while len(slots) < total_slots:
      if seed is not None:
        _shuffle_in_place(eligible, graph.csr.degree, rng)
      slots_before = len(slots)
      eligible, newly_capped = self._greedy_pass(
          graph,
          eligible,
          slots,
          user_count,
          last_batch,
          example_count,
          user_maxpart_cap,
          ex_maxpart_cap,
      )
      capped = np.concatenate([capped, newly_capped])
      if len(slots) > slots_before:
        continue
      # No progress — relax caps if examples were blocked by them.
      if not capped.size:
        break
      user_maxpart_cap += 1
      ex_maxpart_cap = math.ceil(user_maxpart_cap / self._user_example_ratio)
      eligible = np.concatenate([eligible, capped])
      capped = np.empty(0, dtype=eligible.dtype)

    if len(slots) < total_slots:
      raise ValueError(
          f'Could not fill all {total_slots} slots '
          f'(batch_size={self._batch_size} * iterations={self._iterations}). '
          f'Only {len(slots)} slots were filled. '
          f'Try reducing min_sep (currently {self._min_sep}).'
      )
    example_order = np.array(slots, dtype=dtype)
    actual_user_maxpart = int(user_count.max())
    actual_ex_maxpart = int(example_count.max())
    return example_order, actual_user_maxpart, actual_ex_maxpart

  def batch_iterator(
      self, num_examples: int, rng: np.random.Generator | int | None = None
  ) -> Iterator[np.ndarray]:
    """Yields batches satisfying the b-min-sep property.

    All batches are predetermined at initialization. Therefore, ``num_examples``
    is only used for validation, and ``rng`` is unused. See the class docstring
    for more details.

    Args:
      num_examples: Must match ``attribution.num_examples``.
      rng: Unused. Randomness is controlled by ``shuffle_seed``.

    Yields:
      1D arrays of example indices, each of length exactly ``batch_size``.
    """
    del rng  # Randomness controlled by shuffle_seed.
    _validate.equal(self._num_examples, num_examples=num_examples)
    for t in range(self._iterations):
      yield self._example_order[
          t * self._batch_size : (t + 1) * self._batch_size
      ]

  def _greedy_pass(
      self,
      attribution_graph,
      eligible,
      slots,
      user_count,
      last_batch,
      example_count,
      user_maxpart_cap,
      ex_maxpart_cap,
  ):
    """Runs one greedy pass, returning (still_eligible, cap_blocked).

    ``still_eligible`` holds the examples that remain candidates under the
    current caps: those blocked by min-sep, plus those selected in this pass
    that have at least one user. Selected examples are kept because an example
    may be placed in several batches, up to ``ex_maxpart_cap`` times; dropping
    them here would limit the total number of slots to the number of examples.
    ``cap_blocked`` holds the
    examples that were blocked by the per-example or per-user caps, and only
    become candidates again once those caps are relaxed.
    """
    still_eligible = []
    cap_blocked = []
    for ex in eligible.tolist():
      if len(slots) >= self._batch_size * self._iterations:
        break
      current_batch = len(slots) // self._batch_size
      users = attribution_graph.csr.users_of(ex)
      # Per-example cap.
      if example_count[ex] >= ex_maxpart_cap:
        cap_blocked.append(ex)
        continue
      # Per-user cap.
      if np.any(user_count[users] >= user_maxpart_cap):
        cap_blocked.append(ex)
        continue
      # Batch-level min-sep.
      if np.any(current_batch - last_batch[users] < self._min_sep):
        still_eligible.append(ex)
        continue
      # Select.
      slots.append(ex)
      example_count[ex] += 1
      user_count[users] += 1
      last_batch[users] = current_batch
      # Min-sep is enforced through users, so an example without users would
      # be free to repeat within a batch; such examples are used only once.
      if users.size:
        still_eligible.append(ex)
    return np.array(still_eligible, dtype=eligible.dtype), np.array(
        cap_blocked, dtype=eligible.dtype
    )


def _shuffle_in_place(order, degree, rng):
  """Shuffles elements of ``order`` in-place within each degree class."""
  degrees = degree[order]
  boundaries = np.r_[0, np.where(np.diff(degrees))[0] + 1, len(order)]
  for i in range(len(boundaries) - 1):
    rng.shuffle(order[boundaries[i] : boundaries[i + 1]])

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

"""API and implementations for batch selection strategies.

The BatchSelectionStrategy API specifies how batches should be formed from
examples in a framework-agnostic manner. It produces indices into a dataset,
not example elements themselves, and hence relies on having a way to access
individual examples by index efficiently, such as with an in-memory list, array,
or pytree of arrays. For datasets which do not fit in memory, we recommend using
pygrain (https://github.com/google/grain), or using an offline job to reorder
the data on disk before loading into your training pipeline.

The implementations in this file generally materialize a vector of indices of
size ``num_examples`` and hence it requires that this object fits in memory,
i.e., roughly that ``num_examples < 1e9``.

The examples below demonstrate how to use the BatchSelectionStrategy API.
via the CyclicPoissonSampling implementation (all with expected batch size 3):

Example Usage (fixed order + multi-epoch) [1]:
  >>> rng = np.random.default_rng(0)
  >>> b = CyclicPoissonSampling(sampling_prob=1, iterations=8, cycle_length=4)
  >>> print(*b.batch_iterator(12, rng=rng), sep=' ')
  [9 2 7] [ 4  5 11] [0 3 6] [10  8  1] [9 2 7] [ 4  5 11] [0 3 6] [10  8  1]

Example Usage (standard Poisson sampling) [2]:
  >>> b = CyclicPoissonSampling(sampling_prob=0.25, iterations=8)
  >>> print(*b.batch_iterator(12, rng=rng), sep=' ')
  [5 6 7] [5 8 3 7 2] [ 1  5 11] [0 3] [ 5  1  3  4 10] [2] [4 5 1 3] [6]

Example Usage (BandMF-style sampling) [3]:
  >>> p = 0.5
  >>> b = CyclicPoissonSampling(sampling_prob=p, iterations=6, cycle_length=2)
  >>> print(*b.batch_iterator(12, rng=rng), sep=' ')
  [2 4] [1 8 9] [2 7 5 4] [11  1  3] [10  2  5  0  4] [ 1 11  6]

This module also provides multi-owner differential privacy support via
``MultiOwnerGraph``, ``MultiOwnerMinSepSampling``, and
``greedy_contribution_bound``, which handle settings where examples may be
attributed to multiple users [4].

References:
  * [1] `Choquette-Choo et al. (2022) <https://arxiv.org/abs/2211.06530>`_
  * [2] `Abadi et al. (2016) <https://arxiv.org/abs/1607.00133>`_
  * [3] `Choquette-Choo et al. (2023) <https://arxiv.org/abs/2306.08153>`_
  * [4] `Ganesh et al. (2025) <https://arxiv.org/abs/2503.03622>`_

Design Note (public vs. non-public information):
  Batch selection strategy implementations are typically defined as frozen
  dataclasses. Any fields defined on these dataclasses are generally considered
  **public information** — they describe the sampling strategy itself and can
  be shared freely (e.g., as part of a privacy analysis or hyperparameter
  report).

  In contrast, when constructing a concrete iterator over batches via
  ``batch_iterator``, you must pass in potentially **non-public information**
  such as ``num_examples`` (the dataset size) and ``rng`` (the random number
  generator state). These are supplied as arguments rather than fields
  precisely because they may be sensitive.

  Some strategy dataclasses do include an optional ``num_examples`` field. In
  those cases, if ``num_examples`` is set to a value other than ``None``, it
  is considered public information for that strategy instance.
"""

import abc
import dataclasses
import enum
import itertools
import math
from typing import Iterator

from absl import logging
import numpy as np

from . import _validate
from . import sharding_utils

RngType = np.random.Generator | int | None

__all__ = [
    'BatchSelectionStrategy',
    'BallsInBinsSampling',
    'BMinSepSampling',
    'CyclicPoissonSampling',
    'FixedBatchSampling',
    'MinSepRandomAllocation',
    'MultiOwnerGraph',
    'MultiOwnerMinSepSampling',
    'PartitionType',
    'RandomAllocationSampling',
    'UserSelectionStrategy',
    'greedy_contribution_bound',
    'pad_to_multiple_of',
    'split_and_pad_global_batch',
]


class PartitionType(enum.Enum):
  """An enum specifying how examples should be assigned to groups."""

  INDEPENDENT = enum.auto()
  """Each example will be assigned to a group independently at random."""
  EQUAL_SPLIT = enum.auto()
  """Examples will be shuffled and then split into groups of equal size."""


def _get_dtype(num_examples: int) -> np.dtype:
  """Returns the smallest signed integer dtype that fits ``num_examples``."""
  return np.min_scalar_type(-(num_examples + 1))


def _independent_partition(
    num_examples: int,
    num_groups: int,
    rng: np.random.Generator,
    dtype: np.typing.DTypeLike,
) -> list[np.ndarray]:
  sizes = rng.multinomial(num_examples, np.ones(num_groups) / num_groups)
  boundaries = np.cumsum(sizes)[:-1]
  indices = rng.permutation(num_examples).astype(dtype)
  return np.split(indices, boundaries)


def _equal_split_partition(
    num_examples: int,
    num_groups: int,
    rng: np.random.Generator,
    dtype: np.typing.DTypeLike,
) -> list[np.ndarray]:
  indices = rng.permutation(num_examples).astype(dtype)
  group_size = num_examples // num_groups
  groups = np.array_split(indices, num_groups)
  return [g[:group_size] for g in groups]


def _random_compositions(
    rng: np.random.Generator,
    num_examples: int,
    total: int,
    num_parts: int,
) -> np.ndarray:
  """Returns uniform compositions of ``total`` into ``num_parts`` parts."""
  # Draw uniform nonnegative integer compositions summing to `total` via
  # stars and bars: pick `num_parts - 1` distinct bar positions in
  # `[0, total + num_parts - 1)` and take the gaps between sorted bars.
  n = total + num_parts - 1
  perm = rng.random((num_examples, n)).argsort(axis=1)
  bars = np.sort(perm[:, : num_parts - 1], axis=1)
  lo = np.full((num_examples, 1), -1)
  hi = np.full((num_examples, 1), n)
  return np.diff(np.concatenate([lo, bars, hi], axis=1), axis=1) - 1


def split_and_pad_global_batch(
    indices: np.ndarray, minibatch_size: int, microbatch_size: int | None = None
) -> list[np.ndarray]:
  """Splits a global batch of indices into a list of fixed-size minibatches.

  The last minibatch will be padded with ``-1`` indices to make it the right
  size. It is crucial that downstream users correctly account for this by e.g.,
  loading in a dummy example not derived from real data, and explicitly passing
  in ``is_padding_example`` to the clipped gradient function to ensure the
  gradients for these examples are correctly zeroed out.

  Example Usage:
    >>> indices = np.arange(10)
    >>> split_and_pad_global_batch(indices, minibatch_size=4)
    [array([0, 1, 2, 3]), array([4, 5, 6, 7]), array([ 8,  9, -1, -1])]

  Args:
    indices: A 1D or 2D numpy array of indices representing the global batch.
    minibatch_size: The desired size of each minibatch. Minibatches of this size
      will typically be passed into a compiled function that computes and
      accumulates clipped gradients.
    microbatch_size: The size of each microbatch. If set, will reorder the last
      minibatch to ensure that the padding indices appear in the right indices
      to enable early stopping within the last minibatch gradient evaluation.
      See :func:`~jax_privacy.sharding_utils.compute_early_stopping_order` for
      more details on this.

  Returns:
    A list of minibatches of indices, each of size exactly minibatch_size.
    The last minibatch may contain extra ``-1`` indices representing padding
    examples to make it the right size.
  """
  sections = range(minibatch_size, indices.shape[0], minibatch_size)
  minibatches = np.array_split(indices, sections, axis=0)
  minibatch_shape = (minibatch_size,) + indices.shape[1:]
  last_minibatch = np.full(minibatch_shape, -1, dtype=indices.dtype)
  last_minibatch[: minibatches[-1].shape[0]] = minibatches[-1]
  permutation = sharding_utils.compute_early_stopping_order(
      minibatch_size, microbatch_size
  )
  minibatches[-1] = last_minibatch[permutation]
  return minibatches


def pad_to_multiple_of(
    indices: np.ndarray,
    multiple: int,
    microbatch_size: int | None = None,
) -> np.ndarray:
  """Pads the last dimension of indices to a multiple of multiple.

  Creates a new 1D array of indices whose length is a multiple of ``multiple``.
  Padding positions are filled with ``-1``; callers should derive
  ``is_padding_example = (indices == -1)`` and pass it to
  :func:`~jax_privacy.clipping.clipped_grad` or
  :func:`~jax_privacy.clipping.clipped_fun`.

  Example Usage:
    >>> indices = np.arange(10)
    >>> pad_to_multiple_of(indices, multiple=4)
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, -1, -1])

  Args:
    indices: A 1D array of batch indices.
    multiple: A positive integer. The input batch will be padded to a multiple
      of this value.
    microbatch_size: If set, must divide ``multiple``. Reorders the padded
      indices so that padding examples cluster into the last microbatches under
      Fortran-order reshaping. See
      :func:`~jax_privacy.sharding_utils.compute_early_stopping_order`.

  Returns:
    The padded array of indices.
  """
  if indices.ndim > 1:
    raise ValueError('pad_to_multiple_of currently expects 1D indices.')
  if multiple <= 0:
    raise ValueError(f'Padding multiple must be positive, got {multiple}.')
  if microbatch_size is not None and multiple % microbatch_size != 0:
    raise ValueError(f'{multiple=} must be divisible by {microbatch_size=}.')
  curr_size = indices.shape[0]
  pad_size = (multiple - curr_size) % multiple
  new_indices = np.full(curr_size + pad_size, -1, dtype=indices.dtype)
  new_indices[:curr_size] = indices
  permutation = sharding_utils.compute_early_stopping_order(
      new_indices.shape[0], microbatch_size
  )
  return new_indices[permutation]


class BatchSelectionStrategy(abc.ABC):
  """Abstract base class for batch selection strategies.

  A batch selection strategy is a function that takes a random number generator
  and returns an iterator of batches of data indices. The strategy can
  either be deterministic or random, it may produce equal-sized batches or
  variable-sized batches. Note that the batches of indices, which
  specify which examples contribute in which iterations, are generally
  considered sensitive, and should not be inspected directly.

  This API does not prescribe a specific dataset format, but it is expected
  that the format used supports efficient random access to individual examples.
  """

  # TODO: b/545416482 - Investigate stateful iterator instead of plain method.
  @abc.abstractmethod
  def batch_iterator(
      self, num_examples: int, rng: RngType = None
  ) -> Iterator[np.ndarray]:
    """Yields 1D batches of data indices."""


@dataclasses.dataclass(frozen=True)
class CyclicPoissonSampling(BatchSelectionStrategy):
  """Implements Poisson sampling, possibly with cyclic sampling and truncation.

  This generalizes several common sampling strategies [1,2,3,4].

  References:
    * [1] `Choquette-Choo et al. (2022) <https://arxiv.org/abs/2211.06530>`_
    * [2] `Abadi et al. (2016) <https://arxiv.org/abs/1607.00133>`_
    * [3] `Choquette-Choo et al. (2023) <https://arxiv.org/abs/2306.08153>`_
    * [4] `Chua et al. (2024) <https://arxiv.org/abs/2411.04205>`_

  Formal guarantees of the ``batch_iterator``:
    - All batches consist of indices in the range ``[0, num_examples)``.
    - Each example only appears in batches with index ``i`` such that
      ``i % cycle_length == j`` for some fixed ``j`` per example.
    - Without truncation, every index independently appears in each batch
      (where it is eligible to participate subject to the previous
      guarantee) with probability ``sampling_prob``.
    - With truncation, if > ``truncated_batch_size`` examples appear in a batch
      under the previous guarantee, then we select ``truncated_batch_size`` of
      them uniformly at random and discard the rest.
    - If ``partition_type = PartitionType.EQUAL_SPLIT``,
      ``num_examples % cycle_length`` examples are discarded, i.e. never
      sampled.

  Attributes:
    sampling_prob: The probability of sampling an example in rounds when it is
      eligible to participate. To achieve an average batch size of
      ``expected_batch_size``, one should ideally set ``sampling_prob =
      expected_batch_size / (num_examples // cycle_length)``.
    iterations: The number of total iterations / batches to generate.
    truncated_batch_size: If set, after Poisson sampling, if we have more than
      ``truncated_batch_size`` examples in a batch, we uniformly sample
      ``truncated_batch_size`` of them and discard the rest.
    cycle_length: If > 1, we use cyclic Poisson sampling: we partition the
      examples into ``cycle_length`` groups, and do Poisson sampling from the
      groups in a round-robin fashion. ``cycle_length == 1`` retrieves standard
      Poisson sampling.
    partition_type: How to partition the examples into groups for before Poisson
      sampling. ``EQUAL_SPLIT`` is the default, and is only compatible with
      zero-out and replace-one adjacency notions, while ``INDEPENDENT`` is
      compatible with the add-remove adjacency notion.
  """

  sampling_prob: float
  iterations: int
  truncated_batch_size: int | None = None
  cycle_length: int = 1
  partition_type: PartitionType = PartitionType.EQUAL_SPLIT

  def __post_init__(self):
    _validate.non_negative(iterations=self.iterations)
    _validate.in_range(0, 1, sampling_prob=self.sampling_prob)
    _validate.positive(cycle_length=self.cycle_length)
    if self.truncated_batch_size is not None:
      _validate.non_negative(truncated_batch_size=self.truncated_batch_size)

  def batch_iterator(
      self, num_examples: int, rng: RngType = None
  ) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(rng)
    dtype = _get_dtype(num_examples)

    if self.partition_type == PartitionType.INDEPENDENT:
      partition_fn = _independent_partition
    elif self.partition_type == PartitionType.EQUAL_SPLIT:
      partition_fn = _equal_split_partition
    else:
      raise ValueError(f'Unsupported partition type: {self.partition_type}')

    partition = partition_fn(num_examples, self.cycle_length, rng, dtype)

    for i in range(self.iterations):
      current_group = partition[i % self.cycle_length]
      # See Lemma 1 of https://arxiv.org/abs/2406.17298v3 for a proof this
      # is equivalent to Poisson sampling.
      sample_size = rng.binomial(n=len(current_group), p=self.sampling_prob)
      if self.truncated_batch_size is not None:
        sample_size = min(sample_size, self.truncated_batch_size)
      yield rng.choice(
          current_group, size=sample_size, replace=False, shuffle=False
      )


@dataclasses.dataclass(frozen=True)
class BallsInBinsSampling(BatchSelectionStrategy):
  """Implements balls-in-bins sampling.

  In balls-in-bins, each example is independently assigned a 'bin' from 0 to
  ``cycle_length - 1`` uniformly at random, and then appears in all rounds ``i``
  such that ``i % cycle_length == bin``. See https://arxiv.org/abs/2410.06266
  and https://arxiv.org/abs/2412.16802 for more details.

  Formal guarantees of the ``batch_iterator``:
    - All batches consist of indices in the range ``[0, num_examples)``.
    - Each example appears in all batches with index ``i`` such that
      ``i % cycle_length == j``, with ``j`` chosen uniformly at random
      independently for each example, and in no other batches.

  Attributes:
    iterations: The number of total iterations / batches to generate.
    cycle_length: The number of batches in a cycle, equivalently the separation
      between two consecutive appearances of the same example.
  """

  iterations: int
  cycle_length: int

  def __post_init__(self):
    _validate.non_negative(iterations=self.iterations)
    _validate.positive(cycle_length=self.cycle_length)

  def batch_iterator(
      self, num_examples: int, rng: RngType = None
  ) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(rng)
    dtype = _get_dtype(num_examples)
    groups = _independent_partition(num_examples, self.cycle_length, rng, dtype)

    for i in range(self.iterations):
      yield groups[i % self.cycle_length]


@dataclasses.dataclass(frozen=True)
class RandomAllocationSampling(BatchSelectionStrategy):
  """Implements k-out-of-t random allocation (aka balanced-iteration sampling).

  Each example independently selects exactly :math:`k` steps (out of the steps
  it is eligible for) to participate in, uniformly at random. For :math:`k=1`
  and ``cycle_length=1``, this participation pattern is equivalent to
  BallsInBinsSampling, and for ``total_participations = ceil(iterations /
  cycle_length)`` and ``sampling_prob < 1``, it recovers CyclicPoissonSampling
  (without batch truncation).

  If ``cycle_length > 1``, each example is first independently assigned to one
  of ``cycle_length`` groups uniformly at random, and group :math:`j` is only
  eligible to participate in iterations :math:`i` with ``i % cycle_length ==
  j``.
  Two consecutive participations of an example are then always at least
  ``cycle_length`` iterations apart, which satisfies the min-separation
  requirement of a ``cycle_length``-banded matrix mechanism.
  ``cycle_length == 1`` retrieves standard random allocation.

  References:
    * https://arxiv.org/abs/2206.03151 (k=1 only)
    * https://arxiv.org/abs/2306.08153 (cycle_length > 1)
    * https://arxiv.org/abs/2410.06266 (k=1 only)
    * https://arxiv.org/abs/2412.16802 (k=1 only)
    * https://arxiv.org/abs/2502.08202 (k>1)
    * https://arxiv.org/abs/2503.03043 (k>1)
    * https://arxiv.org/abs/2601.21636 (k>1)
    * https://arxiv.org/abs/2602.17284 (k>1)
    * https://arxiv.org/abs/2605.07072 (k>1)
    * https://arxiv.org/abs/2610.12712 (cycle_length > 1, k > 1)

  Formal guarantees of the ``batch_iterator``:
    - All batches consist of indices in the range ``[0, num_examples)``.
    - Each example only appears in batches with index ``i`` such that
      ``i % cycle_length == j`` for some fixed ``j`` chosen uniformly at random
      independently for each example.
    - Each example appears in *exactly* :math:`k` of the batches it is eligible
      for, chosen uniformly at random without replacement.
    - The allocation for each example is independent of all other examples.

  Accounting:
    For DP-SGD with Gaussian noise (``cycle_length = 1``), and for amplified
    BandMF with a ``cycle_length``-banded strategy matrix, exact analytical
    privacy accounting under the add-or-remove-one adjacency notion is supported
    via :func:`~jax_privacy.accounting.random_allocation_dpsgd_event` and
    :func:`~jax_privacy.accounting.random_allocation_bandmf_event`
    respectively.

  Attributes:
    total_participations: The number of eligible steps each example participates
      in (k). If ``iterations`` is not a multiple of ``cycle_length``, the
      schedule runs as if for the next multiple of ``cycle_length`` and
      truncates after ``iterations`` steps.
    iterations: The total number of iterations / batches to generate (t).
    cycle_length: If > 1, the examples are independently partitioned into
      ``cycle_length`` groups which participate in a round-robin fashion, so
      that each example participates at most once every ``cycle_length``
      iterations. ``cycle_length == 1`` retrieves standard random allocation.
    partition_type: How to partition the examples into groups before random
      allocation. ``INDEPENDENT`` is the default, while ``EQUAL_SPLIT`` shuffles
      and splits into groups of equal size (discarding ``num_examples %
      cycle_length`` examples).
    sampling_prob: Probability of independently keeping each allocated
      participation. Defaults to ``1.0`` (no subsampling).
  """

  total_participations: int
  iterations: int
  cycle_length: int = 1
  partition_type: PartitionType = PartitionType.INDEPENDENT
  sampling_prob: float = 1.0

  def __post_init__(self):
    _validate.non_negative(
        total_participations=self.total_participations,
        iterations=self.iterations,
    )
    _validate.positive(cycle_length=self.cycle_length)
    _validate.in_range(0, 1, sampling_prob=self.sampling_prob)
    num_cycles = math.ceil(self.iterations / self.cycle_length)
    if self.total_participations > num_cycles:
      raise ValueError(
          f'Expected total_participations={self.total_participations} <='
          f' ceil(iterations / cycle_length)={num_cycles}.'
      )
    if self.iterations % self.cycle_length != 0:
      logging.warning(
          'iterations=%d is not a multiple of cycle_length=%d; set iterations '
          'to a multiple of cycle_length for best utility.',
          self.iterations,
          self.cycle_length,
      )

  def batch_iterator(
      self, num_examples: int, rng: RngType = None
  ) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(rng)
    dtype = _get_dtype(num_examples)

    if self.partition_type == PartitionType.INDEPENDENT:
      partition_fn = _independent_partition
    elif self.partition_type == PartitionType.EQUAL_SPLIT:
      partition_fn = _equal_split_partition
    else:
      raise ValueError(f'Unsupported partition type: {self.partition_type}')

    partition = partition_fn(num_examples, self.cycle_length, rng, dtype)
    # If iterations is not a multiple of cycle_length, we pretend we are running
    # for ceil(iterations / cycle_length) * cycle_length iterations so that all
    # groups have the same number of eligible iterations, and then truncate to
    # self.iterations.
    num_cycles = math.ceil(self.iterations / self.cycle_length)
    remaining = [
        np.full(len(group), self.total_participations) for group in partition
    ]

    for i in range(self.iterations):
      group_index = i % self.cycle_length
      group = partition[group_index]
      eligible = num_cycles - i // self.cycle_length
      # At each eligible iteration, an example with r remaining participations
      # and e remaining eligible iterations participates with probability r/e.
      # This is equivalent to each example choosing k of its eligible
      # iterations uniformly without replacement, but uses only O(n) space
      # instead of O(n*k).
      probs = remaining[group_index] / eligible
      mask = rng.random(len(group)) < probs
      if self.sampling_prob < 1.0:
        kept = mask & (rng.random(len(group)) < self.sampling_prob)
      else:
        kept = mask
      yield group[kept]
      # Decrement remaining by the allocated mask, not the kept subset, so the
      # conditional allocation probabilities remain exact under subsampling.
      remaining[group_index] -= mask


@dataclasses.dataclass(frozen=True)
class FixedBatchSampling(BatchSelectionStrategy):
  """Implements fixed-size batch sampling.

  Each batch is sampled uniformly at random from the dataset. By default,
  batches are sampled without replacement within a batch, and with replacement
  across batches (i.e., the same example can appear in multiple iterations).

  References:
    * https://arxiv.org/abs/1807.01647
    * https://arxiv.org/abs/1808.00087 (sampling without replacement)

  Attributes:
    batch_size: The number of examples per batch.
    iterations: The number of total iterations / batches to generate.
    replace: Whether to sample with replacement within each batch.
  """

  batch_size: int
  iterations: int
  replace: bool = False

  def __post_init__(self):
    _validate.non_negative(
        batch_size=self.batch_size, iterations=self.iterations
    )

  def batch_iterator(
      self, num_examples: int, rng: RngType = None
  ) -> Iterator[np.ndarray]:
    if not self.replace and self.batch_size > num_examples:
      raise ValueError(
          'batch_size must be <= num_examples when replace is False.'
      )
    rng = np.random.default_rng(rng)
    dtype = _get_dtype(num_examples)
    for _ in range(self.iterations):
      yield rng.choice(
          num_examples,
          size=self.batch_size,
          replace=self.replace,
      ).astype(dtype, copy=False)


@dataclasses.dataclass(frozen=True)
class BMinSepSampling(BatchSelectionStrategy):
  """Implements b-min-sep Poisson sampling.

  Each batch is sampled using Poisson sampling, ignoring any example that
  participated in the previous ``min_sep - 1`` iterations.

  - While both this and cyclic Poisson sampling enforce the b-min-sep
    property, cyclic Poisson does so by only allowing an example to be eligible
    to participate every :math:`b` iterations, whereas here every example is
    eligible to participate in every iteration as long as it did not participate
    in the previous :math:`b - 1` iterations. However, we can do accounting for
    cyclic Poisson via PLD accounting, whereas we only know how to analyze
    b-min-sep via Monte Carlo accounting.
  - This is a generalization of balls-in-bins, which also enforces b-min-sep
    sampling. In particular, this reduces to balls-in-bins when
    ``warm_start = True`` and ``sampling_prob = 1``.

  References:
    * https://arxiv.org/abs/2602.09338

  Attributes:
    sampling_prob: The probability an example is sampled in a given iteration,
      given that it was not sampled in any of the previous ``min_sep - 1``
      iterations. Note that the expected batch size is ``dataset_size / (min_sep
      - 1 + 1 / sampling_prob)``, not just ``dataset_size * sampling_prob``.
    iterations: The number of total iterations / batches to generate.
    min_sep: The minimum separation between two sampled examples.
    warm_start: If ``True``, we initialize the b-min-sep sampling process at a
      warm start. This ensures the batch size is consistent from the start of
      training.
    truncated_batch_size: If set, we truncate the batch to this size. To
      maintain that the participation of examples is independent prior to
      truncation, examples which were sampled and then truncated are still
      excluded in the next ``min_sep - 1`` iterations.
  """

  sampling_prob: float
  iterations: int
  min_sep: int
  warm_start: bool = True
  truncated_batch_size: int | None = None

  def __post_init__(self):
    _validate.positive(min_sep=self.min_sep)
    _validate.non_negative(iterations=self.iterations)
    _validate.in_range(0, 1, sampling_prob=self.sampling_prob)
    if self.truncated_batch_size is not None:
      _validate.non_negative(truncated_batch_size=self.truncated_batch_size)

  def batch_iterator(
      self, num_examples: int, rng: RngType = None
  ) -> Iterator[np.ndarray]:
    # Aliases for brevity.
    p = self.sampling_prob
    b = self.min_sep
    rng = np.random.default_rng(rng)
    dtype = _get_dtype(num_examples)
    if self.warm_start and b > 1:
      size = rng.binomial(n=num_examples, p=(b - 1) * p / (1 + (b - 1) * p))
      concatenated_history = rng.choice(
          num_examples,
          size=size,
          replace=False,
      )
      partition = _independent_partition(size, b - 1, rng, dtype)
      history = [concatenated_history[idx] for idx in partition]
    else:
      history = []
    for _ in range(self.iterations):
      # Simpler to just sample from all examples and then remove previously seen
      # examples, than to try to determine all examples not in history and
      # sample from just those.

      # See Lemma 1 of https://arxiv.org/abs/2406.17298v3 for a proof this
      # is equivalent to Poisson sampling.
      sample_size = rng.binomial(n=num_examples, p=p)
      sample = rng.choice(
          num_examples,
          size=sample_size,
          replace=False,
      )
      for previous_batch in history:
        sample = np.setdiff1d(sample, previous_batch, assume_unique=True)
      history.append(sample)
      if self.truncated_batch_size is not None:
        sample = rng.choice(
            sample,
            size=min(sample.size, self.truncated_batch_size),
            replace=False,
        )
      if len(history) >= b:
        history.pop(0)
      yield sample


@dataclasses.dataclass(frozen=True)
class MinSepRandomAllocation(BatchSelectionStrategy):
  """Implements k-out-of-t b-min-sep random allocation.

  This implements Algorithm 2 of https://arxiv.org/abs/2610.12712 (and the
  random-rotation variant in Appendix B when ``random_rotation=True``). Each
  example independently selects :math:`k` steps in ``[0, iterations)`` uniformly
  at random over all subsets of size :math:`k` whose consecutive participations
  are separated by at least ``min_sep`` (:math:`b`) iterations.

  References:
    * https://arxiv.org/abs/2610.12712

  Attributes:
    total_participations: The number of steps each example is allocated to
      participate in before subsampling (k).
    iterations: The total number of iterations / batches to generate (t).
    min_sep: The minimum separation between consecutive participations of the
      same example (b). Defaults to ``1``.
    random_rotation: Whether to place participations on a cycle and apply a
      uniform random rotation for a uniform per-iteration marginal. Defaults to
      ``False``.
    sampling_prob: Probability of independently keeping each allocated
      participation. Defaults to ``1.0`` (no subsampling).
  """

  total_participations: int
  iterations: int
  min_sep: int = 1
  random_rotation: bool = False
  sampling_prob: float = 1.0

  def __post_init__(self):
    _validate.positive(
        total_participations=self.total_participations,
        iterations=self.iterations,
        min_sep=self.min_sep,
    )
    _validate.in_range(0, 1, sampling_prob=self.sampling_prob)
    min_iters = self.total_participations * self.min_sep
    if self.iterations < min_iters:
      raise ValueError(f'Expected iterations={self.iterations} >= {min_iters}.')

  def batch_iterator(
      self, num_examples: int, rng: RngType = None
  ) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(rng)
    dtype = _get_dtype(num_examples)

    if self.random_rotation:
      positions = self._cyclic_positions(num_examples, rng)
    else:
      positions = self._linear_positions(num_examples, rng)

    for step in range(self.iterations):
      selected = np.any(positions == step, axis=1)
      if self.sampling_prob < 1.0:
        poisson_mask = rng.random(num_examples) < self.sampling_prob
        kept = selected & poisson_mask
      else:
        kept = selected
      yield np.where(kept)[0].astype(dtype)

  def _linear_positions(
      self, num_examples: int, rng: np.random.Generator
  ) -> np.ndarray:
    """Returns k positions in [0, t) with interior gaps >= min_sep."""
    # Distribute the leftover slack t - 1 - (k - 1) * b across k + 1 parts
    # (k - 1 interior gaps plus two free ends) and offset by b per step.
    k, b, t = self.total_participations, self.min_sep, self.iterations
    parts = _random_compositions(rng, num_examples, t - 1 - (k - 1) * b, k + 1)
    return np.cumsum(parts[:, :k], axis=1) + np.arange(k) * b

  def _cyclic_positions(
      self, num_examples: int, rng: np.random.Generator
  ) -> np.ndarray:
    """Returns k cyclic positions with gaps >= min_sep and uniform rotation."""
    # Distribute the cyclic slack t - k * b across k gaps, offset by b per step,
    # and apply an independent uniform shift mod t to flatten the marginal.
    k, b, t = self.total_participations, self.min_sep, self.iterations
    parts = _random_compositions(rng, num_examples, t - k * b, k)
    starts = np.cumsum(parts, axis=1) + np.arange(k) * b
    rotation = rng.integers(t, size=(num_examples, 1))
    return (starts + rotation) % t


@dataclasses.dataclass(frozen=True)
class UserSelectionStrategy:
  """A strategy that applies a base_strategy at the user level.

  Each batch returned by the ``batch_iterator`` is a 2D array of integer
  indices,
  where all entries in the same row are examples owned by the same user. The
  examples in this ``user-batch`` are chosen in a cyclic fashion (maybe after
  shuffling). For example, if a user owns 3 examples [0, 5, 10], then each
  time this user is selected, the batches will be selected from
  [0, 5, 10, 0, 5, 10, 0, 5, 10, ...]. It is expected that the gradient will be
  evaluated and clipped w.r.t. this ``user-batch`` before being aggregated
  across users.

  Example Usage:
    >>> rng = np.random.default_rng(0)
    >>> base_strategy = CyclicPoissonSampling(sampling_prob=1, iterations=5)
    >>> strategy = UserSelectionStrategy(base_strategy, 2)
    >>> user_ids = np.array([0,0,0,1,1,2])
    >>> iterator = strategy.batch_iterator(user_ids, rng)
    >>> print(next(iterator))
    [[5 5]
     [0 1]
     [3 4]]
    >>> print(next(iterator))
    [[5 5]
     [2 0]
     [3 4]]

  Attributes:
    base_strategy: The base batch selection strategy to apply at the user level.
      Will be used to select batches of users from the set of users.
    examples_per_user_per_batch: The number of examples to select for each user
      in each batch. Determines the number of columns in the returned batches.
    shuffle_per_user: Whether to shuffle the examples for each user before
      selecting examples_per_user_per_batch.
  """

  base_strategy: BatchSelectionStrategy
  examples_per_user_per_batch: int = 1
  shuffle_per_user: bool = False

  def __post_init__(self):
    _validate.positive(
        examples_per_user_per_batch=self.examples_per_user_per_batch
    )

  def batch_iterator(
      self, user_ids: np.ndarray, rng: RngType = None
  ) -> Iterator[np.ndarray]:
    """Yields 2D batches of data indices.

    Args:
      user_ids: A 1D array that maps each example to a user id, where each
        user_id can be an arbitrary integer.
      rng: A random seed or random number generator.

    Yields:
      2D batches of data indices, where users are sampled according to the
      base_strategy and all entries in the same row are examples owned by the
      same selected user.
    """
    rng = np.random.default_rng(rng)
    # inverse contains cleaned ids in the range [0, ..., num_users - 1].
    unique, inverse = np.unique(user_ids, return_inverse=True)
    num_users = unique.size
    num_examples = user_ids.size
    dtype = _get_dtype(num_examples)

    # Group example indices by user once to avoid an O(n) scan per user.
    order = np.argsort(inverse, kind='stable').astype(dtype, copy=False)
    counts = np.bincount(inverse, minlength=num_users)
    grouped_examples = np.split(order, np.cumsum(counts)[:-1])

    def create_user_generator(user_id):
      owned_examples = grouped_examples[user_id]
      if self.shuffle_per_user:
        owned_examples = owned_examples.copy()
        rng.shuffle(owned_examples)
      return itertools.cycle(owned_examples.tolist())

    user_generators = [create_user_generator(i) for i in range(num_users)]

    examples_per_user = self.examples_per_user_per_batch or 1

    def get_user_batch(user_id):
      generator = user_generators[user_id]
      sample = itertools.islice(generator, examples_per_user)
      return np.array(list(sample), dtype=dtype)

    for user_batch in self.base_strategy.batch_iterator(num_users, rng):
      batch = np.array([get_user_batch(uid) for uid in user_batch], dtype=dtype)
      yield batch.reshape((-1, examples_per_user))


# Re-export multi-owner functionality from _multi_owner submodule.
# Imports are at the bottom because _multi_owner imports batch_selection
# (for BatchSelectionStrategy), creating a circular dependency at module level.
# pylint: disable=wrong-import-position,ungrouped-imports,unused-import
# pylint: disable=g-import-not-at-top,g-bad-import-order
from jax_privacy._multi_owner import greedy_contribution_bound  # noqa: E402
from jax_privacy._multi_owner import MultiOwnerGraph  # noqa: E402
from jax_privacy._multi_owner import MultiOwnerMinSepSampling  # noqa: E402

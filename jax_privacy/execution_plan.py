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

"""Module for defining DP Execution Plans.

# Writing General-Purpose DP Training Loops via DPExecutionPlan

This module introduces the :class:`~jax_privacy.execution_plan.DPExecutionPlan`,
an object designed to encapsulate the core components of a differentially
private (DP) mechanism.
The primary aim is to simplify the process of constructing and applying DP
mechanisms by packaging these components cohesively. A key benefit is the
assurance that, when used correctly, the combination of these components will
achieve the stated DP properties.

The design is framework-agnostic, specifying the essential pillars of a DP
mechanism—such as batch selection and noise addition—without tightly coupling
them to a specific training loop. Each component is exposed through a simple,
well-documented API, allowing for flexible integration into various frameworks
or direct use with JAX.

By programming against our DPExecutionPlan interface, it is easy to swap out
different components or entire mechanisms without changing the core training
loop logic. While the components are designed to work together, they can also be
used selectively. For instance, a researcher might choose to use only the noise
addition component if their dataset doesn't support the efficient random access
required by the batch selection strategy. Though this might invalidate the
formal DP guarantee, it can still be valuable for research or when a heuristic
quantification of privacy is acceptable. Ultimately, ``DPExecutionPlan`` aims to
free users to concentrate on their training pipeline setup, rather than on the
intricacies of correctly assembling DP components to achieve a desired privacy
guarantee.

# Selecting and using a DPExecutionPlan

Constructors for these plans are highly configurable, offering access to the
full capabilities of the underlying components while also providing sensible
defaults. Our primary entry point is currently
:class:`~jax_privacy.execution_plan.BandMFConfig`, although more will become
available in the future.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import math
from typing import Callable, Protocol

from absl import logging
import dp_accounting
import jax
import numpy as np
import optax

from . import _validate
from . import accounting
from . import batch_selection
from . import clipping
from . import noise_addition
from . import sharding_utils
from .matrix_factorization import toeplitz

NeighboringRelation = dp_accounting.NeighboringRelation
AccountantFn = Callable[[NeighboringRelation], dp_accounting.PrivacyAccountant]


class CyclicInnerStrategy(enum.StrEnum):
  """Specifies the "inner" sampling strategy for cyclic sampling strategies.

  Both options assign each example to one of ``num_bands`` groups, and only let
  a group participate in iterations matching its index modulo ``num_bands``.
  They differ in how participations are chosen within the iterations a group is
  eligible for.
  """

  POISSON = enum.auto()
  """Each example participates in each eligible iteration independently with a
  fixed probability, so the number of participations is random. See
  :class:`~jax_privacy.batch_selection.CyclicPoissonSampling`."""

  RANDOM_ALLOCATION = enum.auto()
  """Each example participates in exactly ``expected_participations`` of its
  eligible iterations, chosen uniformly at random. See
  :class:`~jax_privacy.batch_selection.RandomAllocationSampling`."""


@dataclasses.dataclass(frozen=True)
class PerformanceFlags:
  """Performance-only flags that do not affect mechanism behavior or privacy.

  These flags control implementation details such as numerical precision,
  sharding strategy, and memory/compute trade-offs. Changing them should
  not alter the privacy properties or the mathematical definition of the
  DP mechanism.

  Attributes:
    dtype: The dtype to use for noise generation and gradient aggregation.
    noise_seed: Deprecated; pass ``noise_seed`` to
      :meth:`~jax_privacy.execution_plan.ExecutionPlanConfig.make` or
      :class:`~jax_privacy.training.DPTrainer` instead.
    intermediate_strategy: Strategy for generating intermediate noise, controls
      sharding behavior for noise addition.
    microbatch_size: If set, per-example gradient computation is broken into
      sequential microbatches to reduce peak memory at the cost of compute.
    batch_axis_name: Mesh axis name (or tuple of axis names) along which the
      batch dimension is sharded in SPMD settings.
    spmd_axis_name: Deprecated alias for ``batch_axis_name``.
    keep_batch_dim: Whether to keep the batch dimension when computing
      per-example gradients.
    param_specs: Optional PyTree of ``jax.sharding.PartitionSpec`` (or prefix
      thereof) matching the parameter PyTree structure. When set, wraps the
      noise addition transform with
      :func:`~jax_privacy.sharding_utils.with_sharding_specs`.
  """

  dtype: jax.typing.DTypeLike = np.float32
  # Deprecated; pass noise_seed to config.make() or DPTrainer instead.
  noise_seed: int | None = None
  intermediate_strategy: noise_addition.ShardingStrategies = (
      noise_addition.ShardingStrategies.DEFAULT
  )
  microbatch_size: int | None = None
  batch_axis_name: str | tuple[str, ...] | None = None
  # Deprecated alias for batch_axis_name; remove in next release.
  spmd_axis_name: str | tuple[str, ...] | None = None
  keep_batch_dim: bool = True
  param_specs: sharding_utils.PartitionSpecPyTree | None = None

  def __post_init__(self):
    if self.noise_seed is not None:
      logging.warning('PerformanceFlags.noise_seed is deprecated.')
    if self.spmd_axis_name is not None:
      if self.batch_axis_name is not None:
        raise ValueError('Cannot set both batch_axis_name and spmd_axis_name.')
      object.__setattr__(self, 'batch_axis_name', self.spmd_axis_name)


@dataclasses.dataclass(frozen=True)
class DPExecutionPlan:
  """Class for defining a DP execution plan.

  A DP execution plan consists of a collection of components which when used
  together in the expected manner determine the DP guarantee, along with a
  DpEvent which precisely quantifies it. If constructed via one of the
  ExecutionPlanConfig classes defined in this module, then the ``dp_event`` can
  be trusted as having been formally verified by the JAX Privacy authors.

  In pseudo-code, the components of this dataclass should roughly be used as
  follows:

  .. code-block:: python

    plan = ... # Plan depending on the flavor of DP training you want
    noise_state = plan.noise_addition_transform.init(...)
    batch_sampler = plan.batch_selection_strategy
    for indices in batch_sampler.batch_iterator(num_examples):
      batch = data.select(indices)
      grad_fn = plan.clipped_grad(loss_fn)
      clipped_grad_sum = grad_fn(params, batch, ...)

      dp_grad, noise_state = plan.noise_addition_transform.update(
          clipped_grad_sum, noise_state
      )
      # Sensitive, discard immediately after use.
      del indices, batch, clipped_grad_sum
      # Arbitrary post-processing of dp_grad.

  If possible, we recommend coupling the batch selection, clipped aggregation,
  and noise addition components as tightly as possible to ensure sensitive
  objects are not intercepted and used unintentionally. For example, it is
  critical that no modification is applied to the ``clipped_grad_sum`` (such as
  scaling) before the noise_addition_transform is applied, as such a
  modification could invalidate the DP guarantee because the noise is
  calibrated based on the sensitivity of the clipped_grad_sum.

  Attributes:
    clipped_grad: A function with a similar signature to
      :func:`jax.value_and_grad`, but computes a sum of per-example clipped
      gradients.
    batch_selection_strategy: Determines how batches are formed in each
      iteration.
    noise_addition_transform: Stateful transformation that adds noise to clipped
      and aggregated gradients after each iteration.
    dp_event: Characterizes the mechanism in terms of primitive building blocks
      that dp_accounting knows how to analyze.
    neighboring_relation: The DP neighboring relation assumed by this mechanism.
  """

  clipped_grad: Callable[..., clipping.BoundedSensitivityCallable]
  batch_selection_strategy: batch_selection.BatchSelectionStrategy
  noise_addition_transform: optax.GradientTransformation
  dp_event: dp_accounting.DpEvent
  neighboring_relation: NeighboringRelation


class ExecutionPlanConfig(Protocol):
  """Protocol for configs that build a ``DPExecutionPlan``.

  Any config that produces a
  :class:`~jax_privacy.execution_plan.DPExecutionPlan` from
  :class:`~jax_privacy.execution_plan.PerformanceFlags` satisfies this protocol
  (e.g.
  :class:`~jax_privacy.execution_plan.BandMFConfig`).
  Consuming a config rather than a pre-built plan lets callers read
  performance-only settings (such as the aggregation ``dtype``) directly from
  the :class:`~jax_privacy.execution_plan.PerformanceFlags` without
  materializing the plan.
  """

  def make(
      self,
      performance_flags: PerformanceFlags | None = None,
      noise_seed: int | None = None,
  ) -> DPExecutionPlan:
    """Returns a ``DPExecutionPlan`` configured with ``performance_flags``.

    Args:
      performance_flags: Optional performance flags controlling implementation
        details such as ``dtype``, sharding, and microbatching.
      noise_seed: Optional seed for the noise-addition RNG. If ``None``, a fresh
        random seed is drawn; pass an explicit seed for multi-host coordination
        or reproducible tests (and keep it secret in production deployments).
    """


@dataclasses.dataclass(frozen=True, kw_only=True)
class BandMFConfig:
  """Configuration for an Amplified BandMF-based DPExecutionPlan.

  This config is designed to be fully serializable, defined in terms of simple
  types. The config can be created with or without a ``noise_multiplier``. If
  created without one, call ``calibrate()`` to obtain a new config with a
  ``noise_multiplier`` calibrated to a target (epsilon, delta) guarantee.

  Example Usage (Calibrate from epsilon/delta):
    >>> config = BandMFConfig.default(  # doctest: +SKIP
    ...   num_bands=1, iterations=1000, expected_participations=100,
    ... ).calibrate(epsilon=1.0, delta=1e-5)

  Example Usage (Direct noise_multiplier):
    >>> config = BandMFConfig.default(
    ...   num_bands=1, iterations=1000, expected_participations=100,
    ...   noise_multiplier=1.0,
    ... )

  Example Usage (BandMF with custom strategy):
    >>> config = BandMFConfig(  # doctest: +SKIP
    ...   strategy=np.array([1.0, 0.5, 0.2]),
    ...   iterations=1000, expected_participations=400,
    ... ).calibrate(epsilon=1.0, delta=1e-5)

  Example Usage (DP-SGD with random allocation instead of Poisson sampling):
    >>> config = BandMFConfig.default(  # doctest: +SKIP
    ...   num_bands=1, iterations=1000, expected_participations=100,
    ...   sub_strategy=CyclicInnerStrategy.RANDOM_ALLOCATION,
    ... ).calibrate(epsilon=1.0, delta=1e-5)

  References: `Choquette-Choo et al. (2023) <https://arxiv.org/abs/2306.08153>`_
  and `McKenna (2024) <https://arxiv.org/abs/2405.15913>`_.

  Attributes:
    iterations: The number of iterations the mechanism is defined for. Tip: Set
      this to be a multiple of ``num_bands`` for the best utility.
    expected_participations: The number of times each example participates
      across all iterations. With
      :attr:`~jax_privacy.execution_plan.CyclicInnerStrategy.POISSON` this is an
      expectation, and the Poisson sampling probability is derived as
      ``expected_participations * num_bands / iterations``. With
      :attr:`~jax_privacy.execution_plan.CyclicInnerStrategy.RANDOM_ALLOCATION`
      this is the exact number of participations, and must be an integer.
    strategy: The Toeplitz coefficients of the BandMF strategy matrix.
    noise_multiplier: The ratio of noise standard deviation to the query
      sensitivity. The actual noise stddev is determined by this value, the
      query sensitivity, and the strategy matrix column norm. If not set, use
      ``calibrate()`` to automatically determine it from target (epsilon, delta)
      privacy parameters.
    l2_clip_norm: The maximum L2 norm of the per-example gradients.
    rescale_to_unit_norm: Divide the clipped gradient by the ``l2_clip_norm``.
    normalize_by: Divide the sum-of-clipped gradients by this value.
    truncated_batch_size: If using truncated Poisson sampling, the maximum batch
      size to truncate to. If set, the ``plan.batch_selection_strategy`` will
      always return batches of size at most ``truncated_batch_size``, and
      accounting will be based on truncated Poisson sampling (`Choquette-Choo et
      al. (2025) <https://arxiv.org/abs/2508.15089>`_). Only supported with
      :attr:`~jax_privacy.execution_plan.CyclicInnerStrategy.POISSON`.
    num_examples: The number of examples in the dataset. Required when
      ``truncated_batch_size`` is set. Only set when the dataset size is
      considered public, non-sensitive information (e.g., when using zero-out
      adjacency rather than add-remove). If specified, the dataset will be
      partitioned using
      :attr:`~jax_privacy.batch_selection.PartitionType.EQUAL_SPLIT`, and
      otherwise it will be partitioned using
      :attr:`~jax_privacy.batch_selection.PartitionType.INDEPENDENT`. Only
      supported with
      :attr:`~jax_privacy.execution_plan.CyclicInnerStrategy.POISSON`, because
      accounting for random allocation assumes the add-or-remove-one adjacency
      notion.
    column_normalize: Whether to column-normalize the strategy matrix. If True,
      each column of the realized strategy matrix is rescaled to have L2 norm
      1.0, which makes the mechanism invariant to the scale of `strategy`.
    sub_strategy: How examples are assigned to the iterations they are eligible
      for. Defaults to
      :attr:`~jax_privacy.execution_plan.CyclicInnerStrategy.POISSON`. Use
      :attr:`~jax_privacy.execution_plan.CyclicInnerStrategy.RANDOM_ALLOCATION`
      for (cyclic) random allocation, which gives each example exactly
      ``expected_participations`` participations.
  """

  iterations: int
  expected_participations: float
  strategy: np.typing.ArrayLike
  noise_multiplier: float | None = None
  l2_clip_norm: float = 1.0
  rescale_to_unit_norm: bool = True
  normalize_by: float = 1.0
  truncated_batch_size: int | None = None
  num_examples: int | None = None
  column_normalize: bool = False
  sub_strategy: CyclicInnerStrategy = CyclicInnerStrategy.POISSON

  def __post_init__(self):
    _validate.non_negative(
        iterations=self.iterations,
        l2_clip_norm=self.l2_clip_norm,
        normalize_by=self.normalize_by,
    )
    _validate.strategy(self.strategy, self.iterations)
    if self.sub_strategy == CyclicInnerStrategy.RANDOM_ALLOCATION:
      max_participations = math.ceil(self.iterations / self.num_bands)
    else:
      max_participations = self.iterations // self.num_bands
    _validate.in_range(
        0,
        max_participations,
        expected_participations=self.expected_participations,
    )
    if self.noise_multiplier is not None:
      _validate.non_negative(noise_multiplier=self.noise_multiplier)
    if self.truncated_batch_size is not None:
      _validate.non_negative(truncated_batch_size=self.truncated_batch_size)
    if self.num_examples is not None:
      _validate.non_negative(num_examples=self.num_examples)
    if self.truncated_batch_size is not None and self.num_examples is None:
      raise ValueError('truncated_batch_size requires num_examples to be set.')
    if self.sub_strategy == CyclicInnerStrategy.RANDOM_ALLOCATION:
      _validate.is_int(
          context=(
              'If using RANDOM_ALLOCATION, expected_participations must be an'
              ' integer.'
          ),
          expected_participations=self.expected_participations,
      )
      _validate.is_none(
          context=(
              'If using RANDOM_ALLOCATION, truncated_batch_size and '
              'num_examples must be None.'
          ),
          truncated_batch_size=self.truncated_batch_size,
          num_examples=self.num_examples,
      )

  @property
  def _max_column_norm(self) -> float:
    """Maximum L2 column norm of the realized strategy matrix."""
    if self.column_normalize:
      return 1.0
    return float(np.linalg.norm(self.strategy))

  @property
  def num_bands(self) -> int:
    """The number of bands in the strategy matrix."""
    return len(self.strategy)

  @property
  def rmse(self) -> float:
    """Root mean squared error of the mechanism on the prefix-sum workload.

    Requires the config to be calibrated (``noise_multiplier`` must be set).
    The actual noise stddev is noise_multiplier * max_column_norm (see
    `make`), so the error includes the same column norm factor and is
    invariant to the scale of the strategy. This method normalizes by
    ``expected_participations`` so that you can fairly compare RMSE across
    instances with different expected participations.

    Returns:
      The RMSE per query, in units of the clipped gradient.

    Raises:
      ValueError: If ``noise_multiplier`` has not been set.
      NotImplementedError: If ``column_normalize`` is True.
    """
    self._check_calibrated()
    if self.column_normalize:
      raise NotImplementedError(
          'Exact RMSE calculation with column_normalize=True is not supported.'
      )
    strategy = np.asarray(self.strategy)
    errors = toeplitz.per_query_error(strategy_coef=strategy, n=self.iterations)
    coefficient = (
        self.noise_multiplier
        * self._max_column_norm
        / self.expected_participations
    )
    return float(coefficient * np.sqrt(np.mean(errors)))

  @property
  def _neighboring_relation(self) -> NeighboringRelation:
    """Returns the neighboring relation for the config."""
    if self.num_examples is not None:
      return NeighboringRelation.REPLACE_SPECIAL
    return NeighboringRelation.ADD_OR_REMOVE_ONE

  @property
  def _partition_type(self) -> batch_selection.PartitionType:
    """Returns the partition type for the config."""
    if self.num_examples is not None:
      return batch_selection.PartitionType.EQUAL_SPLIT
    return batch_selection.PartitionType.INDEPENDENT

  def _get_dp_event(self, sigma: float) -> dp_accounting.DpEvent:
    """Returns a DpEvent for the BandMF mechanism."""
    if self.sub_strategy == CyclicInnerStrategy.RANDOM_ALLOCATION:
      return accounting.random_allocation_bandmf_event(
          noise_multiplier=sigma,
          iterations=self.iterations,
          num_bands=self.num_bands,
          total_participations=int(self.expected_participations),
      )
    sampling_prob = (
        self.expected_participations * self.num_bands / self.iterations
    )
    if self.truncated_batch_size:
      group_size = self.num_examples // self.num_bands
      return accounting.truncated_amplified_bandmf_event(
          noise_multiplier=sigma,
          iterations=self.iterations,
          num_bands=self.num_bands,
          largest_group_size=group_size,
          sampling_prob=sampling_prob,
          truncated_batch_size=self.truncated_batch_size,
      )
    else:
      return accounting.amplified_bandmf_event(
          noise_multiplier=sigma,
          iterations=self.iterations,
          num_bands=self.num_bands,
          sampling_prob=sampling_prob,
      )

  def _check_calibrated(self) -> None:
    """Raises ValueError if noise_multiplier has not been set."""
    if self.noise_multiplier is None:
      raise ValueError(
          'noise_multiplier is not set. Call calibrate() or provide'
          ' noise_multiplier directly.'
      )

  def calibrate(
      self,
      *,
      epsilon: float,
      delta: float,
      tol: float | None = None,
      accountant_fn: AccountantFn = dp_accounting.pld.PLDAccountant,
  ) -> BandMFConfig:
    """Returns a new config with a calibrated ``noise_multiplier``.

    Args:
      epsilon: The target privacy budget.
      delta: The target privacy failure probability.
      tol: The tolerance in ``noise_multiplier`` space for the calibration
        binary search. Defaults to 1e-6 if not specified.
      accountant_fn: A function that returns a fresh privacy accountant used for
        calibration given a neighboring relation. Defaults to
        :class:`~dp_accounting.pld.PLDAccountant`.

    Returns:
      A new BandMFConfig with calibrated ``noise_multiplier``.
    """
    noise_multiplier = dp_accounting.calibrate_dp_mechanism(
        make_fresh_accountant=lambda: accountant_fn(self._neighboring_relation),
        make_event_from_param=self._get_dp_event,
        target_epsilon=epsilon,
        target_delta=delta,
        tol=tol,
    )
    return dataclasses.replace(self, noise_multiplier=noise_multiplier)

  @classmethod
  def default(
      cls,
      num_bands: int,
      iterations: int,
      expected_participations: float,
      strategy_optimization_steps: int = 500,
      **kwargs,
  ) -> BandMFConfig:
    """Returns a BandMFConfig with an RMSE-optimized strategy.

    See ``BandMFConfig`` for the full list of keyword arguments.

    Args:
      num_bands: The number of bands in the strategy matrix.
      iterations: The number of iterations the mechanism is defined for.
      expected_participations: The expected number of times each example
        participates across all iterations.
      strategy_optimization_steps: The number of optimization steps to use for
        the strategy matrix.
      **kwargs: Keyword arguments to pass to ``BandMFConfig``.

    Returns:
      A BandMFConfig with an RMSE-optimized strategy.
    """
    strategy = toeplitz.optimize_banded_toeplitz(
        n=iterations,
        bands=num_bands,
        max_optimizer_steps=strategy_optimization_steps,
    )
    return BandMFConfig(
        iterations=iterations,
        expected_participations=expected_participations,
        strategy=strategy,
        **kwargs,
    )

  def make(
      self,
      performance_flags: PerformanceFlags | None = None,
      noise_seed: int | None = None,
  ) -> DPExecutionPlan:
    """Returns a DP execution plan for the given BandMF mechanism config.

    Args:
      performance_flags: Optional performance flags that control implementation
        details such as ``dtype``, ``sharding``, and ``microbatching``. If
        ``None``, default values are used for all performance flags.
      noise_seed: Optional seed for the random number generator used for noise
        addition. If ``None``, a fresh random seed is drawn; pass an explicit
        seed for multi-host coordination or reproducible tests (and keep it
        secret in production deployments).

    Returns:
      A DPExecutionPlan configured from this config and the given performance
      flags.

    Raises:
      ValueError: If ``noise_multiplier`` has not been set.
    """
    self._check_calibrated()
    if performance_flags is None:
      performance_flags = PerformanceFlags()
    if noise_seed is None:
      noise_seed = performance_flags.noise_seed

    @functools.wraps(clipping.clipped_grad)
    def clipped_grad_transform(*args, **kwargs):
      return clipping.clipped_grad(
          *args,
          **kwargs,
          l2_clip_norm=self.l2_clip_norm,
          normalize_by=self.normalize_by,
          rescale_to_unit_norm=self.rescale_to_unit_norm,
          dtype=performance_flags.dtype,
          microbatch_size=performance_flags.microbatch_size,
          spmd_axis_name=performance_flags.batch_axis_name,
          keep_batch_dim=performance_flags.keep_batch_dim,
      )

    if self.sub_strategy == CyclicInnerStrategy.RANDOM_ALLOCATION:
      batch_selection_strategy = batch_selection.RandomAllocationSampling(
          total_participations=int(self.expected_participations),
          iterations=self.iterations,
          cycle_length=self.num_bands,
      )
    else:
      sampling_prob = (
          self.expected_participations * self.num_bands / self.iterations
      )
      batch_selection_strategy = batch_selection.CyclicPoissonSampling(
          sampling_prob=sampling_prob,
          iterations=self.iterations,
          cycle_length=self.num_bands,
          truncated_batch_size=self.truncated_batch_size,
          partition_type=self._partition_type,
      )

    max_column_norm = self._max_column_norm
    column_normalize_for_n = self.iterations if self.column_normalize else None
    # Keep closed-over Toeplitz coefficients uncommitted so shard_map in
    # with_sharding_specs does not see AxisType.Auto constants; np.asarray
    # strips any mesh sharding already attached to self.strategy.
    with jax.set_mesh(None):
      noising_matrix = toeplitz.inverse_as_streaming_matrix(
          np.asarray(self.strategy), column_normalize_for_n
      )

    query_sensitivity = clipped_grad_transform(lambda: None).sensitivity()

    dp_event = self._get_dp_event(self.noise_multiplier)

    privatizer = noise_addition.matrix_factorization_privatizer(
        noising_matrix,
        stddev=float(
            self.noise_multiplier * query_sensitivity * max_column_norm
        ),
        prng_key=noise_seed,
        dtype=performance_flags.dtype,
        intermediate_strategy=performance_flags.intermediate_strategy,
    )
    if performance_flags.param_specs is not None:
      privatizer = sharding_utils.with_sharding_specs(
          privatizer, performance_flags.param_specs
      )

    return DPExecutionPlan(
        clipped_grad=clipped_grad_transform,
        batch_selection_strategy=batch_selection_strategy,
        noise_addition_transform=privatizer,
        dp_event=dp_event,
        neighboring_relation=self._neighboring_relation,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class NonPrivateConfig:
  """Configuration for a non-private execution plan (no clipping, no noise).

  This config is designed for standard training, bypassing privacy safeguards
  (clipping norm is set to infinity, and noise stddev is zero).

  Attributes:
    iterations: The number of iterations to train for.
    batch_size: The batch size to use for training.
  """

  iterations: int
  batch_size: int

  def make(
      self,
      performance_flags: PerformanceFlags | None = None,
      noise_seed: int | None = None,
  ) -> DPExecutionPlan:
    """Returns a DPExecutionPlan configured for non-private training."""
    del noise_seed
    if performance_flags is None:
      performance_flags = PerformanceFlags()

    @functools.wraps(clipping.clipped_grad)
    def clipped_grad_transform(*args, **kwargs):
      return clipping.clipped_grad(
          *args,
          **kwargs,
          l2_clip_norm=float('inf'),
          normalize_by=float(self.batch_size),
          rescale_to_unit_norm=False,
          dtype=performance_flags.dtype,
          microbatch_size=performance_flags.microbatch_size,
          spmd_axis_name=performance_flags.batch_axis_name,
          keep_batch_dim=performance_flags.keep_batch_dim,
      )

    batch_selection_strategy = batch_selection.FixedBatchSampling(
        batch_size=self.batch_size,
        iterations=self.iterations,
    )

    # Use optax.identity to avoid adding any noise
    noise_addition_transform = optax.identity()

    return DPExecutionPlan(
        clipped_grad=clipped_grad_transform,
        batch_selection_strategy=batch_selection_strategy,
        noise_addition_transform=noise_addition_transform,
        dp_event=dp_accounting.NonPrivateDpEvent(),
        neighboring_relation=NeighboringRelation.ADD_OR_REMOVE_ONE,
    )

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

"""Functions for creating DpEvent objects for common JAX Privacy mechanisms.

Example Usage (Calculating Epsilon for DP-SGD):

  >>> import dp_accounting
  >>> event = dpsgd_event(noise_multiplier=3, iterations=512, sampling_prob=0.1)
  >>> accountant = dp_accounting.pld.PLDAccountant()
  >>> epsilon = accountant.compose(event).get_epsilon(target_delta=1e-6)
  >>> round(epsilon, 2)
  3.77

Example Usage (Calibrating Noise Multiplier for DP-SGD):

  >>> make_event = lambda sigma: dpsgd_event(sigma, 512, sampling_prob=0.1)
  >>> noise_multiplier = dp_accounting.calibrate_dp_mechanism(
  ...     dp_accounting.pld.PLDAccountant,
  ...     make_event,
  ...     target_epsilon=1.0,
  ...     target_delta=1e-6
  ... )
  >>> round(noise_multiplier, 2)
  9.66

Example Usage (Calibrating Number of Iterations for DP-SGD):

  >>> make_event = lambda t: dpsgd_event(3.0, t, sampling_prob=0.1)
  >>> dp_accounting.calibrate_dp_mechanism(
  ...     dp_accounting.pld.PLDAccountant,
  ...     make_event,
  ...     target_epsilon=2.0,
  ...     target_delta=1e-6,
  ...     discrete=True,
  ...     bracket_interval=dp_accounting.LowerEndpointAndGuess(1, 128)
  ... )
  155
"""

import math

import dp_accounting

from . import _validate


def _validate_poisson_args(noise_multiplier, iterations, sampling_prob):
  """Validates common arguments for Poisson-sampled DP events."""
  _validate.non_negative(
      noise_multiplier=noise_multiplier, iterations=iterations
  )
  _validate.in_range(0, 1, sampling_prob=sampling_prob)


def _validate_fixed_args(
    noise_multiplier: float,
    iterations: int,
    dataset_size: int,
    batch_size: int,
    replace: bool,
) -> None:
  """Validates the arguments for fixed-size sampling DP events.

  Args:
    noise_multiplier: Noise multiplier of the Gaussian mechanism.
    iterations: Number of iterations the mechanism is run for.
    dataset_size: Number of examples in the dataset.
    batch_size: Batch size per iteration.
    replace: Whether sampling is done with replacement.
  """
  _validate.non_negative(
      noise_multiplier=noise_multiplier,
      iterations=iterations,
      dataset_size=dataset_size,
      batch_size=batch_size,
  )
  if not replace and batch_size > dataset_size:
    raise ValueError(
        f'Expected {batch_size=} <= {dataset_size=} for replace=False.'
    )
  if replace and dataset_size == 0 and batch_size > 0:
    raise ValueError(
        f'Expected {batch_size=} == 0 for {dataset_size=} with replace=True.'
    )


def dpsgd_event(
    noise_multiplier: float,
    iterations: int,
    *,
    sampling_prob: float,
) -> dp_accounting.DpEvent:
  """Returns the DpEvent for DP-SGD with the given training parameters.

  This mechanism is a composition of Poisson-sampled Gaussian mechanisms. See
  this paper for more details:

  * "Deep Learning with Differential Privacy"
    `Abadi et al. (2016) <https://arxiv.org/abs/1607.00133>`_

  Args:
    noise_multiplier: The noise multiplier of the mechanism.
    iterations: The number of iterations to run the mechanism for.
    sampling_prob: The Poisson sampling probability of the mechanism, i.e., the
      probability an example will be included in each batch.

  Returns:
    A DpEvent object.
  """
  _validate_poisson_args(noise_multiplier, iterations, sampling_prob)
  gaussian = dp_accounting.GaussianDpEvent(noise_multiplier)
  sampled = dp_accounting.PoissonSampledDpEvent(sampling_prob, gaussian)
  return dp_accounting.SelfComposedDpEvent(sampled, iterations)


def fixed_dpsgd_event(
    noise_multiplier: float,
    iterations: int,
    *,
    dataset_size: int,
    batch_size: int,
    replace: bool = False,
) -> dp_accounting.DpEvent:
  """Returns the DpEvent for DP-SGD with fixed-size sampling.

  This mechanism samples a fixed-size batch at each iteration and applies a
  Gaussian mechanism to the aggregated gradients. It is equivalent to DP-SGD
  with fixed-size sampling.

  Note: The without-replacement event is compatible with the RDP accountant
  under the REPLACE_ONE neighboring relation. Sampled-with-replacement events
  may not be supported by all accountants.

  Args:
    noise_multiplier: The noise multiplier of the mechanism.
    iterations: The number of iterations to run the mechanism for.
    dataset_size: The number of examples in the dataset.
    batch_size: The fixed batch size per iteration.
    replace: Whether to sample with replacement.

  Returns:
    A DpEvent object.
  """
  _validate_fixed_args(
      noise_multiplier,
      iterations,
      dataset_size,
      batch_size,
      replace,
  )
  gaussian = dp_accounting.GaussianDpEvent(noise_multiplier)
  if replace:
    sampled = dp_accounting.dp_event.SampledWithReplacementDpEvent(
        dataset_size,
        batch_size,
        gaussian,
    )
  else:
    sampled = dp_accounting.dp_event.SampledWithoutReplacementDpEvent(
        dataset_size,
        batch_size,
        gaussian,
    )
  return dp_accounting.SelfComposedDpEvent(sampled, iterations)


def truncated_dpsgd_event(
    noise_multiplier: float,
    iterations: int,
    *,
    sampling_prob: float,
    num_examples: int,
    truncated_batch_size: int,
) -> dp_accounting.DpEvent:
  """Returns the DpEvent for truncated DP-SGD with the given training params.

  This mechanism is like DP-SGD, but batches larger than
  ``truncated_batch_size`` are truncated to size ``truncated_batch_size``. See
  these references for more information about this mechanism:

  * "Scalable DP-SGD: Shuffling vs. Poisson Subsampling"
    `Chua et al. (2024) <https://arxiv.org/abs/2411.04205>`_
  * "Tighter Privacy Analysis for Truncated Poisson Sampling"
    `Ganesh et al. (2025) <https://arxiv.org/abs/2508.15089>`_

  Args:
    noise_multiplier: The noise multiplier of the mechanism.
    iterations: The number of iterations to run the mechanism for.
    sampling_prob: The Poisson sampling probability of the mechanism, i.e., the
      probability an example will be included in each batch before truncation.
    num_examples: The number of examples in the dataset.
    truncated_batch_size: The maximum batch size.

  Returns:
    A DpEvent object.
  """
  _validate_poisson_args(noise_multiplier, iterations, sampling_prob)
  sampled_gaussian = dp_accounting.TruncatedSubsampledGaussianDpEvent(
      dataset_size=num_examples,
      sampling_probability=sampling_prob,
      truncated_batch_size=truncated_batch_size,
      noise_multiplier=noise_multiplier,
  )
  return dp_accounting.SelfComposedDpEvent(sampled_gaussian, iterations)


def amplified_bandmf_event(
    noise_multiplier: float,
    iterations: int,
    *,
    num_bands: int,
    sampling_prob: float,
) -> dp_accounting.DpEvent:
  """Returns the DpEvent for DP-BandMF with the given training parameters.

  The examples will be split up into ``num_bands`` groups, and then minibatches
  will be formed by sampling within each group. See "(Amplified) Banded Matrix
  Factorization: A unified approach to private training"
  `Choquette-Choo et al. (2023) <https://arxiv.org/abs/2306.08153>`_ for
  details.

  Args:
    noise_multiplier: The noise multiplier to use, assuming the strategy matrix
      has maximum L2 column norm of one.
    iterations: The number of iterations to run the mechanism for.
    num_bands: The number of bands to use.
    sampling_prob: The Poisson sampling probability of the mechanism, i.e., the
      probability an example will be included in each batch. Note that because
      the examples are partitioned into ``num_bands`` groups, the expected batch
      size is actually ``dataset_size * sampling_prob / num_bands`` (i.e., a
      factor of ``num_bands`` smaller than DP-SGD).

  Returns:
    A DpEvent object.
  """
  _validate_poisson_args(noise_multiplier, iterations, sampling_prob)
  _validate.positive(num_bands=num_bands)
  rounds = math.ceil(iterations / num_bands)
  return dpsgd_event(
      noise_multiplier=noise_multiplier,
      sampling_prob=sampling_prob,
      iterations=rounds,
  )


def truncated_amplified_bandmf_event(
    noise_multiplier: float,
    iterations: int,
    *,
    num_bands: int,
    sampling_prob: float,
    largest_group_size: int,
    truncated_batch_size: int,
) -> dp_accounting.DpEvent:
  """Returns the DpEvent for truncated DP-BandMF with the given parameters.

  This mechanism is like BandMF, but batches larger than
  ``truncated_batch_size`` are truncated to size ``truncated_batch_size``. This
  mechanism is compatible with the ``zero-out`` adjacency notion, and requires
  knowledge of the total number of examples.

  Args:
    noise_multiplier: The noise multiplier to use, assuming the strategy matrix
      has maximum L2 column norm of one.
    iterations: The number of iterations to run the mechanism for.
    num_bands: The number of bands to use.
    sampling_prob: The Poisson sampling probability of the mechanism, i.e., the
      probability an example will be included in each batch before truncation.
      Note that because the examples are partitioned into ``num_bands`` groups,
      the expected batch size (before truncation) is actually ``dataset_size *
      sampling_prob / num_bands`` (i.e., a factor of ``num_bands`` smaller than
      DP-SGD).
    largest_group_size: The number of examples in the largest partition group.
      For EQUAL_SPLIT this is ``num_examples // num_bands`` (floor). Use
      ``math.ceil(num_examples / num_bands)`` when groups may be uneven.
    truncated_batch_size: The maximum batch size.

  Returns:
    A DpEvent object.
  """
  _validate_poisson_args(noise_multiplier, iterations, sampling_prob)
  _validate.positive(num_bands=num_bands)
  return truncated_dpsgd_event(
      noise_multiplier=noise_multiplier,
      sampling_prob=sampling_prob,
      iterations=math.ceil(iterations / num_bands),
      num_examples=largest_group_size,
      truncated_batch_size=truncated_batch_size,
  )


def random_allocation_dpsgd_event(
    noise_multiplier: float,
    iterations: int,
    *,
    total_participations: int,
) -> dp_accounting.DpEvent:
  """Returns the DpEvent for DP-SGD with random allocation (k-out-of-t).

  In random allocation (also known as balanced-iteration sampling), each example
  independently participates in exactly ``total_participations`` (:math:`k`)
  iterations chosen uniformly at random without replacement from ``iterations``
  (:math:`t`) total steps. For :math:`k=1`, this participation pattern is
  equivalent to single-epoch balls-in-bins sampling.

  References:
    * "Privacy Amplification by Random Allocation"
      `Feldman & Shenfeld (2025) <https://arxiv.org/abs/2502.08202>`_
    * "Leveraging Randomness in Model and Data Partitioning for Privacy
      Amplification" `Dong, Chen, & Ozgur (2025)
      <https://arxiv.org/abs/2503.03043>`_
    * "Efficient Privacy Loss Accounting for Subsampling and Random Allocation"
      `Feldman & Shenfeld (2026) <https://arxiv.org/abs/2602.17284>`_
    * "Less Random, More Private: What is the Optimal Subsampling Scheme for
      DP-SGD?" `Dong & Ozgur (2026) <https://arxiv.org/abs/2605.07072>`_

  Args:
    noise_multiplier: The noise multiplier of the Gaussian mechanism.
    iterations: The total number of iterations/batches (t).
    total_participations: The number of steps each example participates in (k).

  Returns:
    A DpEvent object.
  """
  _validate.non_negative(
      noise_multiplier=noise_multiplier,
      iterations=iterations,
      total_participations=total_participations,
  )
  if total_participations > iterations:
    raise ValueError(f'Expected {total_participations=} <= {iterations=}.')
  gaussian = dp_accounting.GaussianDpEvent(noise_multiplier)
  return dp_accounting.RandomAllocationDpEvent(
      event=gaussian,
      num_selected=total_participations,
      num_steps=iterations,
  )


def random_allocation_bandmf_event(
    noise_multiplier: float,
    iterations: int,
    *,
    num_bands: int,
    total_participations: int,
) -> dp_accounting.DpEvent:
  """Returns the DpEvent for DP-BandMF with cyclic random allocation.

  The examples are partitioned into ``num_bands`` groups, and each group only
  participates in iterations matching its index modulo ``num_bands``. Within the
  iterations it is eligible for, each example participates in exactly
  ``total_participations`` (:math:`k`) of them, chosen uniformly at random. This
  guarantees a minimum separation of ``num_bands`` between two consecutive
  participations, so with a ``num_bands``-banded strategy matrix the mechanism
  reduces to random allocation of a single Gaussian mechanism over the
  iterations of one group. See
  :class:`~jax_privacy.batch_selection.RandomAllocationSampling` with
  ``cycle_length = num_bands`` for the corresponding batch selection strategy.

  If ``iterations`` is not a multiple of ``num_bands``, the batch selection
  strategy runs as if for ``ceil(iterations / num_bands) * num_bands``
  iterations and truncates to ``iterations`` (which is post-processing). Every
  group therefore has ``ceil(iterations / num_bands)`` eligible iterations and
  the exact same privacy guarantee (prior to post-processing).

  References:
    * "(Amplified) Banded Matrix Factorization: A unified approach to private
      training" `Choquette-Choo et al. (2023)
      <https://arxiv.org/abs/2306.08153>`_
    * "Privacy Amplification by Random Allocation"
      `Feldman & Shenfeld (2025) <https://arxiv.org/abs/2502.08202>`_
    * "Leveraging Randomness in Model and Data Partitioning for Privacy
      Amplification" `Dong, Chen, & Ozgur (2025)
      <https://arxiv.org/abs/2503.03043>`_
    * "Efficient Privacy Loss Accounting for Subsampling and Random Allocation"
      `Feldman & Shenfeld (2026) <https://arxiv.org/abs/2602.17284>`_
    * "Less Random, More Private: What is the Optimal Subsampling Scheme for
      DP-SGD?" `Dong & Ozgur (2026) <https://arxiv.org/abs/2605.07072>`_

  Args:
    noise_multiplier: The noise multiplier to use, assuming the strategy matrix
      has maximum L2 column norm of one.
    iterations: The number of iterations to run the mechanism for.
    num_bands: The number of bands to use.
    total_participations: The number of iterations each example participates in
      (k). Must be at most ``ceil(iterations / num_bands)``.

  Returns:
    A DpEvent object.
  """
  _validate.positive(num_bands=num_bands)
  return random_allocation_dpsgd_event(
      noise_multiplier=noise_multiplier,
      iterations=math.ceil(iterations / num_bands),
      total_participations=total_participations,
  )

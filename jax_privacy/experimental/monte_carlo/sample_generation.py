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

"""Methods for sampling from various PLDs arising from DP-BandMF.

For convenience, all methods assume C in DP-MF is a banded Toeplitz matrix,
which can be specified by the non-zero entries of its first column. The main
benefit of this assumption is that we can compute Cx efficiently via
convolution. The analysis behind the methods here can be extended to other
families of C, but that is beyond the current scope of this library.

Docstrings will refer to an example being included or excluded from the dataset.
The intended adjacency to use this library with is the zero-out adjacency, so
excluded really means the sensitive example is zeroed out. When truncation is
not used, zero-out is equivalent to add-remove. Otherwise add-remove is
generally not compatible with truncation due to the need for the dataset size to
be public.
"""

import functools

import numpy as np
import scipy as sp

from ... import _validate
from ... import batch_selection

Seed = int | np.random.Generator | None
SupportedStrategies = (
    batch_selection.BallsInBinsSampling
    | batch_selection.BMinSepSampling
    | batch_selection.CyclicPoissonSampling
)


def _validate_c_col(c_col: np.ndarray) -> None:
  """Validates c_col for DP-BandMF."""
  if c_col.ndim != 1:
    raise ValueError('c_col must be a 1D array.')
  if np.any(c_col < 0):
    raise ValueError('c_col must be non-negative.')


def _banded_c_times_x(c_col: np.ndarray, x: np.ndarray) -> np.ndarray:
  """Multiplies the LT Toeplitz matrix whose non-zeros are c_col by x."""

  assert c_col.size <= x.size
  if c_col.size == 0:
    return np.zeros_like(x, dtype=x.dtype)
  if c_col.size == 1:
    return c_col[0] * x
  top_prod = sp.signal.convolve(c_col, x[: -c_col.size + 1])
  bot_block = sp.linalg.toeplitz(c_col[:-1], np.zeros(c_col.size - 1))
  bot_prod = np.matmul(bot_block, x[-c_col.size + 1 :])
  top_prod[-c_col.size + 1 :] += bot_prod
  return top_prod


def _add_banded_c_times_sparse_x(
    mode: np.ndarray,
    c_col: np.ndarray,
    r_idx: np.ndarray,
    c_idx: np.ndarray | None = None,
    x_vals: np.ndarray | float | None = None,
    cols: np.ndarray | None = None,
) -> None:
  """Adds C * x to mode in-place for sparse non-zero entries of x.

  Args:
    mode: 2D array of shape ``(iterations, num_samples)`` modified in-place.
    c_col: 1D array of non-zero entries in the first column of banded C.
    r_idx: 1D array of iteration (row) indices where x is non-zero.
    c_idx: 1D array of sample (column) indices where x is non-zero. Only needed
      if ``cols`` is ``None``.
    x_vals: Optional 1D array (matching ``r_idx``/``c_idx`` length) or scalar of
      values of x at (``r_idx``, ``c_idx``). Defaults to 1.0.
    cols: Optional precomputed 2D array of column indices of shape
      ``(r_idx.size, c_col.size)``.
  """
  if r_idx.size == 0 or c_col.size == 0:
    return
  iterations = mode.shape[0]
  rows = r_idx[:, None] + np.arange(c_col.size)
  if cols is None:
    if c_idx is None:
      raise ValueError('Either c_idx or cols must be provided.')
    cols = np.broadcast_to(c_idx[:, None], rows.shape)
  if x_vals is None:
    vals = np.broadcast_to(c_col[None, :], rows.shape)
  else:
    x_vals = np.asarray(x_vals)
    if x_vals.ndim == 1:
      vals = c_col[None, :] * x_vals[:, None]
    else:
      vals = np.broadcast_to(c_col[None, :] * x_vals, rows.shape)
  mask = (rows >= 0) & (rows < iterations)
  np.add.at(mode, (rows[mask], cols[mask]), vals[mask])


def _normal_llrs(
    dot_products: np.ndarray,
    squared_norms: np.ndarray,
    sigma: float,
    mode_scale: float = 1.0,
) -> np.ndarray:
  """Log of N(s * c, sigma^2 I)(y) / N(0, sigma^2 I)(y), given <c, y>, |c|^2."""
  return (2 * mode_scale * dot_products - mode_scale**2 * squared_norms) / (
      2 * sigma**2
  )


def _sliding_dot_products_and_squared_norms(
    samples: np.ndarray, kernel: np.ndarray, num_shifts: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
  """Returns <k_i, y> for each sample y and |k_i|^2, for shifts i < num_shifts.

  Here k_i is ``kernel`` shifted down by i rounds and cut off at the last round:
  column i of the banded Toeplitz matrix C when ``kernel`` is ``c_col``, or the
  i-th balls-in-bins mode when ``kernel`` is the first mode. The dot products
  are a sliding window of the sample against the kernel, and the squared norms
  are prefix sums of the squared kernel. Computing all dot products at once via
  FFT costs O(m log m) per sample with m = ``kernel.size + num_shifts``, versus
  O(kernel.size * num_shifts) directly.

  Args:
    samples: 2D array of shape ``(iterations, num_samples)``.
    kernel: 1D array.
    num_shifts: The number of shifts, defaulting to ``iterations``. Kernels
      shifted past the last round are zero.

  Returns:
    A tuple of (i) the dot products, of shape ``(num_shifts, num_samples)``,
    and (ii) the squared norms, of shape ``(num_shifts,)``.
  """
  iterations = samples.shape[0]
  if num_shifts is None:
    num_shifts = iterations
  if kernel.size == 0:
    return np.zeros((num_shifts, samples.shape[1])), np.zeros(num_shifts)
  # Rows past the last window are irrelevant, and zero padding lets the window
  # slide past the last round.
  num_rows = kernel.size + num_shifts - 1
  padded = np.pad(samples, ((0, max(num_rows - iterations, 0)), (0, 0)))
  dot_products = sp.signal.fftconvolve(
      padded[:num_rows], kernel[::-1, np.newaxis], mode='valid', axes=0
  )
  # |k_i|^2 = |kernel[:iterations - i]|^2.
  prefix_norms = np.concatenate([[0.0], np.cumsum(kernel**2)])
  squared_norms = prefix_norms[
      np.clip(iterations - np.arange(num_shifts), 0, kernel.size)
  ]
  return dot_products, squared_norms


@functools.lru_cache(maxsize=1)
def _balls_in_bins_first_mode(
    iterations: int,
    cycle_length: int,
    c_col_tuple: tuple[float, ...],
) -> np.ndarray:
  """Computes and caches Cx for x = 1 in every cycle_length-th round from 0.

  Since C is lower-triangular Toeplitz, the mode for the pattern starting at
  round j is this mode shifted down by j rows (and truncated).
  """
  # Need a tuple input for caching.
  c_col = np.array(c_col_tuple)
  x = (np.arange(iterations) % cycle_length == 0).astype(float)
  return _banded_c_times_x(c_col, x)


@functools.lru_cache(maxsize=1)
def _all_balls_in_bins_modes(
    iterations: int,
    cycle_length: int,
    c_col_tuple: tuple[float, ...],
) -> np.ndarray:
  """Computes and caches all modes induced by balls-in-bins."""
  first_mode = _balls_in_bins_first_mode(iterations, cycle_length, c_col_tuple)
  return sp.linalg.toeplitz(first_mode, np.zeros(cycle_length))


def balls_in_bins_modes(
    strategy: batch_selection.BallsInBinsSampling, c_col: np.ndarray
) -> np.ndarray:
  """Returns all modes of the mixture induced by balls-in-bins sampling.

  Args:
    strategy: The balls-in-bins sampling strategy.
    c_col: The non-zero entries in the first column of C. Should be non-negative
      and 1D.

  Returns:
    An array of shape ``(strategy.iterations, strategy.cycle_length)`` whose
    columns are the equally likely means of the positive distribution in the
    dominating pair.
  """
  _validate_c_col(c_col)
  return _all_balls_in_bins_modes(
      strategy.iterations,
      strategy.cycle_length,
      tuple(c_col[: strategy.iterations]),
  )


def _sample_balls_in_bins_modes(
    iterations: int,
    cycle_length: int,
    c_col: np.ndarray,
    rng: np.random.Generator,
    positive_sample: bool = True,
    num_samples: int = 1,
) -> np.ndarray:
  """Samples Cx for the distribution on x induced by balls-in-bins sampling.

  See `Choquette-Choo et al. (2024) <https://arxiv.org/abs/2410.06266>`_ for
  details.

  Args:
    iterations: The number of iterations of DP-MF.
    cycle_length: The length of each cycle of balls-in-bins sampling.
    c_col: The non-zero entries in the first column of C. Should be non-negative
      and 1D.
    rng: The rng to use for sampling.
    positive_sample: If ``True``, we sample from the distribution in the
      dominating pair corresponding to the case where the sensitive example is
      included. Otherwise, we sample from the other case in the dominating pair,
      where the sensitive example is not included.
    num_samples: The number of samples to generate.

  Returns:
    Modes of shape ``(iterations, num_samples)``.
  """
  _validate.positive(iterations=iterations)
  _validate.positive(cycle_length=cycle_length)
  if c_col.size > iterations:
    c_col = c_col[:iterations]
  if not positive_sample:
    return np.zeros((iterations, num_samples))
  # Cx, where x is a vector which is 1 in every b-th coordinate starting at a
  # random position in {0, 1, ..., b-1} and 0 otherwise.
  possible_modes = _all_balls_in_bins_modes(
      iterations, cycle_length, tuple(c_col)
  )
  counts = rng.multinomial(
      n=num_samples, pvals=np.full(cycle_length, 1.0 / cycle_length)
  )
  return np.repeat(possible_modes, repeats=counts, axis=1)


def _sample_cyclic_poisson_modes(
    iterations: int,
    cycle_length: int,
    c_col: np.ndarray,
    rng: np.random.Generator,
    sampling_prob: float = 1.0,
    positive_sample: bool = True,
    num_samples: int = 1,
) -> np.ndarray:
  """Samples Cx for the distribution on x induced by cyclic Poisson sampling.

  See `Choquette-Choo et al. (2024) <https://arxiv.org/abs/2410.06266>`__ for
  details.

  Args:
    iterations: The number of iterations of DP-MF.
    cycle_length: The cycle length of cyclic Poisson sampling.
    c_col: The non-zero entries in the first column of C. Should be non-negative
      and 1D.
    rng: The rng to use for sampling.
    sampling_prob: The probability an example is sampled in the iterations it is
      eligible to be sampled in.
    positive_sample: If ``True``, we sample from the distribution in the
      dominating pair corresponding to the case where the sensitive example is
      included. Otherwise, we sample from the other case in the dominating pair,
      where the sensitive example is not included.
    num_samples: The number of samples to generate.

  Returns:
    Modes of shape ``(iterations, num_samples)``.
  """
  _validate.positive(iterations=iterations)
  _validate.positive(cycle_length=cycle_length)
  _validate.in_range(0.0, 1.0, sampling_prob=sampling_prob)
  if c_col.size > cycle_length:
    raise ValueError('c_col must have length <= cycle_length.')
  mode = np.zeros((iterations, num_samples))
  if positive_sample and c_col.size > 0:
    shifts = rng.integers(cycle_length, size=num_samples)
    # x is a matrix of shape (iterations, num_samples) where entry (j, i) is 1
    # if iteration j is sampled for sample i, and 0 otherwise.
    two_d_shifts = shifts[None, :]
    eligible = (np.arange(iterations)[:, None] % cycle_length) == two_d_shifts
    sampled = rng.uniform(size=(iterations, num_samples)) < sampling_prob
    x = eligible & sampled
    r_idx, c_idx = np.where(x)
    _add_banded_c_times_sparse_x(mode, c_col, r_idx, c_idx)
  return mode


def _sample_b_min_sep_positive_modes_no_truncation(
    strategy: batch_selection.BMinSepSampling,
    c_col: np.ndarray,
    rng: np.random.Generator,
    num_samples: int,
) -> np.ndarray:
  """Samples Cx for distribution on x induced by b-min-sep sampling."""
  # Rename for brevity and alignment with paper.
  b = strategy.min_sep
  p = strategy.sampling_prob
  n = strategy.iterations
  mode = np.zeros((n, num_samples))
  if strategy.warm_start:
    warm_start_probs = np.zeros(b)
    warm_start_probs[0] = 1.0 / (1.0 + (b - 1) * p)
    warm_start_probs[1:] = p * warm_start_probs[0]
    last_part = rng.choice(b, p=warm_start_probs, size=num_samples) - b
  else:
    last_part = -b * np.ones(num_samples, dtype=np.int32)
  cols = np.broadcast_to(
      np.arange(num_samples)[:, None], (num_samples, c_col.size)
  )
  while np.min(last_part) < n:
    last_part = last_part + b - 1 + rng.geometric(p, size=num_samples)
    _add_banded_c_times_sparse_x(mode, c_col, last_part, cols=cols)
  return mode


def _sample_b_min_sep_modes_with_truncation(
    strategy: batch_selection.BMinSepSampling,
    c_col: np.ndarray,
    rng: np.random.Generator,
    num_samples: int,
    positive_sample: bool,
    dataset_size: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Samples Cx from distribution induced by truncated b-min-sep sampling.

  See Appendix D of `Dong and Ganesh (2026) <https://arxiv.org/abs/2602.09338>`_
  for a derivation.

  Args:
    strategy: The b-min-sep sampling strategy to use.
    c_col: The non-zero entries in the first column of C. Should be non-negative
      and 1D.
    rng: The rng or seed to use for sampling.
    num_samples: The number of samples to generate.
    positive_sample: If ``True``, we sample from the distribution in the
      dominating pair corresponding to the case where the sensitive example is
      included. Otherwise, we sample from the other case in the dominating pair,
      where the sensitive example is not included.
    dataset_size: The size of the dataset. This method is only called when
      ``strategy.truncated_batch_size`` is not ``None``.

  Returns:
    Sample(s) from the distribution induced by truncated b-min-sep sampling.
    Also returns an extra array stating the pre-truncation batch sizes excluding
    the sensitive example.
  """

  # Rename for brevity and alignment with paper.
  b = strategy.min_sep
  p = strategy.sampling_prob
  n = strategy.iterations
  B_max = strategy.truncated_batch_size  # pylint: disable=invalid-name
  mode = np.zeros((n, num_samples))

  # Track the batch size only counting non-sensitive examples, prior to
  # truncation. If we have a warm start, we simulate the warm-start by
  # initializing batch sizes for rounds -b to -1. So, row i of rest_batch_sizes
  # corresponds to round i - b for now.
  rest_batch_sizes = np.zeros((n + b, num_samples), dtype=np.int32)

  # There is a discrepancy in the convention we use for initializing
  # rest_batch_sizes_total depending on whether warm_start is enabled (and thus
  # whether we need to account for a history of b rounds). This discrepancy
  # is resolved by the rest_batch_sizes_total -= rest_batch_sizes[0, :] line
  # in the first iteration of the loop below.
  if strategy.warm_start:
    warm_start_probs = np.zeros(b)
    warm_start_probs[0] = 1.0 / (1.0 + (b - 1) * p)
    warm_start_probs[1:] = p * warm_start_probs[0]
    last_part = rng.choice(b, p=warm_start_probs, size=num_samples) - b
    rest_batch_sizes[:b, :] = rng.multinomial(
        n=dataset_size - 1, pvals=warm_start_probs, size=num_samples
    ).transpose()
    rest_batch_sizes_total = np.full(
        num_samples, dataset_size - 1, dtype=np.int32
    )
  else:
    last_part = np.full(num_samples, -b, dtype=np.int32)
    rest_batch_sizes_total = np.zeros(num_samples, dtype=np.int32)
  # We only actually care about rest_batch_sizes for rounds where the sensitive
  # example is included. However, we are not aware of a trick which allows us to
  # only sample those values without computing the full history of
  # rest_batch_sizes. So, we just compute the full history for simplicity. Later
  # we will do a while loop which only considers the rounds where the sensitive
  # example is included, and index into rest_batch_sizes with those rounds.
  for i in range(n):
    rest_batch_sizes_total -= rest_batch_sizes[i, :]
    rest_batch_sizes[i + b, :] = rng.binomial(
        n=dataset_size - 1 - rest_batch_sizes_total,
        p=p,
    )
    rest_batch_sizes_total += rest_batch_sizes[i + b, :]

  # We no longer need the rows corresponding to negative rounds.
  rest_batch_sizes = rest_batch_sizes[b:, :]

  x_truncated = 2.0 if positive_sample else -1.0
  x_non_truncated = 1.0 if positive_sample else 0.0

  # See the comment above about why we track the full history of
  # rest_batch_sizes, but this loop only needs to consider the rounds where the
  # sensitive example is included.
  # Note that rest_batch_sizes counts the number of non-sensitive examples in
  # each batch (prior to truncation), whereas last_part tracks the participation
  # of the sensitive example (ignoring truncation). We also prevent an example
  # that is sampled and then truncated from participating in the next b-1
  # rounds. So rest_batch_sizes and last_part are independent.
  while np.min(last_part) < n:
    last_part = last_part + b - 1 + rng.geometric(p, size=num_samples)
    active_mask = last_part < n
    num_active = np.count_nonzero(active_mask)
    if num_active > 0:
      active_indices = np.arange(num_samples)[active_mask]
      active_last_part = last_part[active_mask]
      indicator = rest_batch_sizes[active_last_part, active_indices] >= B_max
      # If no truncation, x = x_non_truncated by default.
      x_active = np.full(num_active, x_non_truncated)
      if np.any(indicator):
        active_rest_batch_sizes = rest_batch_sizes[
            active_last_part[indicator], active_indices[indicator]
        ]
        retention_prob = B_max / (active_rest_batch_sizes + 1)
        size = np.count_nonzero(indicator)
        retained = rng.uniform(size=size) < retention_prob
        # If truncation and retained, x = x_truncated.
        # If truncation and not retained, x = 0.
        x_active[indicator] = np.where(retained, x_truncated, 0.0)
      _add_banded_c_times_sparse_x(
          mode, c_col, active_last_part, active_indices, x_active
      )
  return mode, rest_batch_sizes


def _sample_b_min_sep_modes(
    strategy: batch_selection.BMinSepSampling,
    c_col: np.ndarray,
    rng: np.random.Generator,
    positive_sample: bool = True,
    num_samples: int = 1,
    dataset_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
  """Samples Cx for the distribution on x induced by b-min-sep sampling.

  See `Dong and Ganesh (2026) <https://arxiv.org/abs/2602.09338>`__ for details.

  Args:
    strategy: The b-min-sep sampling strategy to use.
    c_col: The non-zero entries in the first column of C. Should be non-negative
      and 1D. It is assumed that the length of ``c_col`` is the same as the
      minimum separation parameter in the sampling scheme.
    rng: The rng to use for sampling.
    positive_sample: If ``True``, we sample from the distribution in the
      dominating pair corresponding to the case where the sensitive example is
      included. Otherwise, we sample from the other case in the dominating pair,
      where the sensitive example is not included.
    num_samples: The number of samples to generate.
    dataset_size: The size of the dataset. Only used if
      ``strategy.truncated_batch_size`` is not ``None``.

  Returns:
    A tuple of (i) modes of shape ``(iterations, num_samples)`` and (ii) if
    ``strategy.truncated_batch_size`` is not ``None``, an extra array stating
    the pre-truncation batch sizes excluding the sensitive example, otherwise
    ``None``.
  """
  if c_col.size > strategy.min_sep:
    raise ValueError('c_col must have length less than or equal to min_sep.')
  if c_col.size > strategy.iterations:
    c_col = c_col[: strategy.iterations]
  if strategy.truncated_batch_size:
    return _sample_b_min_sep_modes_with_truncation(
        strategy, c_col, rng, num_samples, positive_sample, dataset_size
    )
  if positive_sample:
    mode = _sample_b_min_sep_positive_modes_no_truncation(
        strategy, c_col, rng, num_samples
    )
  else:
    mode = np.zeros((strategy.iterations, num_samples))
  return mode, None


def generate_sample(
    strategy: SupportedStrategies,
    noise_multiplier: float,
    c_col: np.ndarray,
    seed: Seed = None,
    positive_sample: bool = True,
    num_samples: int = 1,
    dataset_size: int | None = None,
    mode_scale: float | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
  """Generates a sample from the dominating pair for amplified DP-BandMF.

  TODO: Explore benefits of Jax implementation.

  Args:
    strategy: The batch selection strategy to use. Currently,
      :class:`~jax_privacy.batch_selection.BallsInBinsSampling`,
      :class:`~jax_privacy.batch_selection.CyclicPoissonSampling` (using
      ``INDEPENDENT`` partition type), and
      :class:`~jax_privacy.batch_selection.BMinSepSampling` are supported. See
      `Choquette-Choo et al. (2024) <https://arxiv.org/abs/2410.06266>`__ and
      `Dong and Ganesh (2026) <https://arxiv.org/abs/2602.09338>`__ for
      technical details.
    noise_multiplier: The noise multiplier of DP-MF. This is multiplied by the
      clip norm, not accounting for the norm of ``c_col``.
    c_col: The non-zero entries in the first column of C. Should be non-negative
      and 1D.
    seed: The rng or seed to use for sampling.
    positive_sample: If ``True``, we sample from the distribution in the
      dominating pair corresponding to the case where the sensitive example is
      included. Otherwise, we sample from the other case in the dominating pair,
      where the sensitive example is not included. Ignored if ``mode_scale`` is
      not ``None``.
    num_samples: The number of samples to generate. The default is 1, but it is
      typically much more efficient to generate multiple samples in a single
      call to benefit from vectorization.
    dataset_size: The size of the dataset. Should only be set if accounting for
      the strategy supports truncation, and ``strategy.truncated_batch_size`` is
      not ``None``.
    mode_scale: If ``None``, we sample from the dominating pair according to
      ``positive_sample``. Otherwise, the modes of the positive mixture are
      multiplied by ``mode_scale`` before noise is added, which samples from a
      stretched mixture that is useful as a proposal distribution for importance
      sampling (``0`` and ``1`` recover the dominating pair). Only supported for
      balls-in-bins sampling.

  Returns:
    Sample(s) from the dominating PLD for DP-BandMF using balls-in-bins
    sampling. The output is 2D with dimension ``(strategy.iterations,
    num_samples)``. Potentially also returns a second array containing auxiliary
    information needed to evaluate the privacy loss.
  """
  _validate.in_range(0, np.inf, noise_multiplier=noise_multiplier)
  _validate_c_col(c_col)
  if mode_scale is None:
    mode_scale = 1.0
  elif isinstance(strategy, batch_selection.BallsInBinsSampling):
    positive_sample = True  # The modes of the positive mixture are scaled.
  else:
    raise ValueError('mode_scale is only supported for balls-in-bins sampling.')
  rng = np.random.default_rng(seed)
  aux = None
  if isinstance(strategy, batch_selection.BallsInBinsSampling):
    if dataset_size is not None:
      raise ValueError(
          'Monte Carlo accounting for balls-in-bins sampling does not support '
          'truncation (yet), so dataset_size should not be set.'
      )
    mode = _sample_balls_in_bins_modes(
        iterations=strategy.iterations,
        cycle_length=strategy.cycle_length,
        c_col=c_col,
        rng=rng,
        positive_sample=positive_sample,
        num_samples=num_samples,
    )
  elif isinstance(strategy, batch_selection.BMinSepSampling):
    if dataset_size is None and strategy.truncated_batch_size is not None:
      raise ValueError(
          'dataset_size must be set if '
          'strategy.truncated_batch_size is not None.'
      )
    if dataset_size is not None:
      _validate.positive(dataset_size=dataset_size)
    if strategy.truncated_batch_size is not None:
      _validate.positive(truncated_batch_size=strategy.truncated_batch_size)
    mode, aux = _sample_b_min_sep_modes(
        strategy=strategy,
        c_col=c_col,
        rng=rng,
        positive_sample=positive_sample,
        num_samples=num_samples,
        dataset_size=dataset_size,
    )
  elif isinstance(strategy, batch_selection.CyclicPoissonSampling):
    if dataset_size is not None:
      raise ValueError(
          'Monte Carlo accounting for cyclic Poisson sampling does not support '
          'truncation (yet), so dataset_size should not be set.'
      )
    if strategy.partition_type != batch_selection.PartitionType.INDEPENDENT:
      raise ValueError(
          'Monte Carlo accounting for cyclic Poisson sampling does not support '
          'non-cyclic partition types.'
      )
    mode = _sample_cyclic_poisson_modes(
        iterations=strategy.iterations,
        cycle_length=strategy.cycle_length,
        c_col=c_col,
        rng=rng,
        sampling_prob=strategy.sampling_prob,
        positive_sample=positive_sample,
        num_samples=num_samples,
    )
  else:
    raise ValueError(f'Unsupported batch selection strategy: {type(strategy)}')
  sample = rng.normal(loc=mode_scale * mode, scale=noise_multiplier)
  return sample if aux is None else (sample, aux)


def _compute_balls_in_bins_privacy_loss(
    epoch_length: int,
    sample: np.ndarray,
    sigma: float,
    c_col: np.ndarray,
    mode_scale: float = 1.0,
) -> np.ndarray:
  """Computes the privacy loss for a sample from balls-in-bins sampling.

  Args:
    epoch_length: The length of each epoch (number of bins) for balls-in-bins
      sampling.
    sample: The sample(s), generated by :func:`generate_sample`.
    sigma: The noise multiplier of DP-MF. This is multiplied by the clip norm,
      not accounting for the norm of ``c_col``.
    c_col: The non-zero entries in the first column of C. Should be non-negative
      and 1D. Assumed to match the epoch length of balls-in-bins.
    mode_scale: A scalar by which the modes of the mixture are multiplied. See
      :func:`compute_privacy_loss`.

  Returns:
    The privacy loss of the sample, assuming we sample from the distribution in
    the dominating pair where the sensitive example is included.
  """
  _validate.positive(epoch_length=epoch_length)
  _validate.positive(sigma=sigma)
  _validate.equal(expected=2, sample_dimension=sample.ndim)
  iterations = sample.shape[0]
  _validate_c_col(c_col)
  if c_col.size > iterations:
    c_col = c_col[:iterations]
  # Mode j is the first mode shifted down by j rounds, so this is far cheaper
  # than multiplying by the dense mode matrix, which is mostly zeros for banded
  # C.
  first_mode = _balls_in_bins_first_mode(iterations, epoch_length, tuple(c_col))
  dot_products, squared_mode_norms = _sliding_dot_products_and_squared_norms(
      sample, first_mode, num_shifts=epoch_length
  )
  llrs = _normal_llrs(
      dot_products, squared_mode_norms[:, np.newaxis], sigma, mode_scale
  )
  return sp.special.logsumexp(llrs, axis=0) - np.log(epoch_length)


def _compute_cyclic_poisson_privacy_loss(
    cycle_length: int,
    sampling_prob: float,
    sample: np.ndarray,
    sigma: float,
    c_col: np.ndarray,
) -> np.ndarray:
  """Computes the privacy loss for a sample from cyclic Poisson sampling.

  Args:
    cycle_length: The length of each epoch (number of bins) for cyclic Poisson
      sampling.
    sampling_prob: The probability an example is sampled in a given iteration.
    sample: The sample(s), generated by :func:`generate_sample`.
    sigma: The noise multiplier of DP-MF. This is multiplied by the clip norm,
      not accounting for the norm of ``c_col``.
    c_col: The non-zero entries in the first column of C. Should be non-negative
      and 1D. Assumed to match the epoch length of balls-in-bins.

  Returns:
    The privacy loss of the sample, assuming we sample from the distribution in
    the dominating pair where the sensitive example is included.
  """
  _validate.positive(cycle_length=cycle_length)
  _validate.positive(sigma=sigma)
  _validate.in_range(0, 1, sampling_prob=sampling_prob)
  _validate.equal(expected=2, sample_dimension=sample.ndim)
  iterations = sample.shape[0]
  _validate_c_col(c_col)
  if c_col.size > cycle_length:
    raise ValueError(
        'c_col must have length less than or equal to cycle_length.'
    )
  dot_products, squared_mode_norms = _sliding_dot_products_and_squared_norms(
      sample, c_col
  )
  unsampled_llrs = _normal_llrs(
      dot_products, squared_mode_norms[:, np.newaxis], sigma
  )
  sampled_llrs = np.logaddexp(
      unsampled_llrs + np.log(sampling_prob), np.log1p(-sampling_prob)
  )
  per_shift_llrs = np.zeros((cycle_length, sample.shape[1]))
  np.add.at(per_shift_llrs, np.arange(iterations) % cycle_length, sampled_llrs)
  privacy_loss = sp.special.logsumexp(per_shift_llrs, axis=0) - np.log(
      cycle_length
  )
  return privacy_loss


def _compute_b_min_sep_privacy_loss_no_truncation(
    strategy: batch_selection.BMinSepSampling,
    samples: np.ndarray,
    sigma: float,
    c_col: np.ndarray,
) -> np.ndarray:
  """Computes the privacy loss for a sample from b-min-sep sampling.

  See `Dong and Ganesh (2026) <https://arxiv.org/abs/2602.09338>`__ for details.
  Note that the dynamic program in the paper returns the likelihood ratio, while
  we return the privacy loss which is a log-likelihood ratio.

  Args:
    strategy: The probability an example is sampled in a given iteration, given
      that it was not sampled in any of the previous b-1 iterations. Note that
      it will participate in 1 / (b - 1 + 1 / sampling_prob) fraction of the
      iterations on average, not sampling_prob fraction of the iterations as in
      Poisson sampling.
    samples: The samples, generated by :func:`generate_sample`.
    sigma: The noise multiplier of DP-MF. This is multiplied by the clip norm,
      not accounting for the norm of ``c_col``.
    c_col: The non-zero entries in the first column of C. Should be non-negative
      and 1D.

  Returns:
    The privacy loss of the sample(s), assuming we sample from the distribution
    in the dominating pair where the sensitive example is included.
  """
  # Aliases to make the math more readable / terse.
  b = strategy.min_sep
  n = samples.shape[0]
  p = strategy.sampling_prob

  # This evaluates the recurrence relation for the privacy loss in Equation (2)
  # of https://arxiv.org/abs/2602.09338. Note that we use log probabilities
  # here, while the paper uses probabilities.
  dot_products, squared_norms = _sliding_dot_products_and_squared_norms(
      samples, c_col
  )
  # The (log of) N(c_i, sigma^2 I) / N(0, sigma^2 I)(y[i:i+b-1]) for each i,
  # which are the coefficients appearing in the recurrence relation.
  normal_llrs = _normal_llrs(dot_products, squared_norms[:, None], sigma)

  # We use a circular buffer to only store b suffix losses and avoid having to
  # shift the dynamic program table around. This also allows us to handle
  # boundary conditions more easily; accessing index i % b when i >= n will
  # give us zero, the right value for the boundary condition.
  suffix_losses_buffer = np.zeros((b, samples.shape[1]))
  log1p_neg_p = np.log1p(-p)
  log_p = np.log(p)

  for i in range(n - 1, -1, -1):
    li_plus_b = suffix_losses_buffer[i % b]
    li_plus_1 = suffix_losses_buffer[(i + 1) % b]
    term1 = log1p_neg_p + li_plus_1
    term2 = log_p + normal_llrs[i] + li_plus_b
    suffix_losses_buffer[i % b] = np.logaddexp(term1, term2)

  if strategy.warm_start:
    suffix_losses_buffer[1:b] += log_p
    log_divisor = np.log1p((b - 1) * p)
    return sp.special.logsumexp(suffix_losses_buffer, axis=0) - log_divisor
  else:
    return suffix_losses_buffer[0]


def _compute_b_min_sep_privacy_loss(
    strategy: batch_selection.BMinSepSampling,
    samples: np.ndarray,
    sigma: float,
    c_col: np.ndarray,
    rest_batch_sizes: np.ndarray | None = None,
) -> np.ndarray:
  """Computes the privacy loss for a sample from b-min-sep sampling.

  See `Dong and Ganesh (2026) <https://arxiv.org/abs/2602.09338>`__ for details.
  Note that the dynamic program in the paper returns the likelihood ratio, while
  we return the privacy loss which is a log-likelihood ratio.

  Args:
    strategy: The probability an example is sampled in a given iteration, given
      that it was not sampled in any of the previous b-1 iterations. Note that
      it will participate in 1 / (b - 1 + 1 / sampling_prob) fraction of the
      iterations on average, not sampling_prob fraction of the iterations as in
      Poisson sampling.
    samples: The samples, generated by :func:`generate_sample`.
    sigma: The noise multiplier of DP-MF. This is multiplied by the clip norm,
      not accounting for the norm of ``c_col``.
    c_col: The non-zero entries in the first column of C. Should be non-negative
      and 1D.
    rest_batch_sizes: The pre-truncation batch sizes excluding the sensitive
      example. If ``None``, we assume no truncation.

  Returns:
    The privacy loss of the sample(s), assuming we sample from the distribution
    in the dominating pair where the sensitive example is included.
  """
  # If no truncation, we can use a more efficient computation.
  if rest_batch_sizes is None:
    return _compute_b_min_sep_privacy_loss_no_truncation(
        strategy, samples, sigma, c_col
    )
  # Aliases to make the math more readable / terse.
  b = strategy.min_sep
  n = samples.shape[0]
  p = strategy.sampling_prob

  # This evaluates the recurrence relation for the privacy loss in Equation (2)
  # of https://arxiv.org/abs/2602.09338. Note that we use log probabilities
  # here, while the paper uses probabilities.
  dot_products, squared_norms = _sliding_dot_products_and_squared_norms(
      samples, c_col
  )

  # We use a circular buffer to only store b suffix losses and avoid having to
  # shift the dynamic program table around. This also allows us to handle
  # boundary conditions more easily; accessing index i % b when i >= n will
  # give us zero, the right value for the boundary condition.

  # Directly computing the log-likelihood ratio between the positive
  # and negative distributions in the dominating pair is challenging since
  # both are mixtures now. Instead, we compute the log-likehood ratios for both
  # with the zero-mean Gaussian, and then take the difference of the two.
  suffix_losses_buffer_pos = np.zeros((b, samples.shape[1]))
  suffix_losses_buffer_neg = np.zeros((b, samples.shape[1]))
  log1p_neg_p = np.log1p(-p)
  log_p = np.log(p)

  for i in range(n - 1, -1, -1):
    li_plus_b_pos = suffix_losses_buffer_pos[i % b]
    li_plus_1_pos = suffix_losses_buffer_pos[(i + 1) % b]
    term1_pos = log1p_neg_p + li_plus_1_pos

    li_plus_b_neg = suffix_losses_buffer_neg[i % b]
    li_plus_1_neg = suffix_losses_buffer_neg[(i + 1) % b]
    term1_neg = log1p_neg_p + li_plus_1_neg

    term2_pos = np.empty(samples.shape[1])
    term2_neg = np.empty(samples.shape[1])
    truncated = rest_batch_sizes[i] >= strategy.truncated_batch_size
    not_truncated = ~truncated

    if np.any(not_truncated):
      normal_llrs = _normal_llrs(
          dot_products[i][not_truncated], squared_norms[i], sigma
      )
      term2_pos[not_truncated] = (
          log_p + normal_llrs + li_plus_b_pos[not_truncated]
      )
      # The negative distribution is also a mean-zero distribution when
      # truncation doesn't occur, so normal_llrs = zero.
      term2_neg[not_truncated] = log_p + li_plus_b_neg[not_truncated]

    # When truncation occurs, we instead want to compute log of
    # (tp * N(kc_i, sigma^2 I) + (1 - tp) * N(0, sigma^2 I)) /
    # N(0, sigma^2 I)), where k = 2 for the positive distribution and k = -1 for
    # the negative distribution, and tp is the probability of the sensitive
    # example surviving truncation if it is sampled.
    if np.any(truncated):
      truncation_probs = strategy.truncated_batch_size / (
          rest_batch_sizes[i][truncated] + 1
      )
      double_sens_llrs = _normal_llrs(
          dot_products[i][truncated], squared_norms[i], sigma, mode_scale=2.0
      )
      negative_sens_llrs = _normal_llrs(
          dot_products[i][truncated], squared_norms[i], sigma, mode_scale=-1.0
      )
      log_tp = np.log(truncation_probs)
      log1p_neg_tp = np.log1p(-truncation_probs)
      truncation_llrs_pos = np.logaddexp(
          log_tp + double_sens_llrs,
          log1p_neg_tp,
      )
      truncation_llrs_neg = np.logaddexp(
          log_tp + negative_sens_llrs,
          log1p_neg_tp,
      )
      term2_pos[truncated] = (
          log_p + truncation_llrs_pos + li_plus_b_pos[truncated]
      )
      term2_neg[truncated] = (
          log_p + truncation_llrs_neg + li_plus_b_neg[truncated]
      )
    suffix_losses_buffer_pos[i % b] = np.logaddexp(term1_pos, term2_pos)
    suffix_losses_buffer_neg[i % b] = np.logaddexp(term1_neg, term2_neg)

  if strategy.warm_start:
    suffix_losses_buffer_pos[1:b] += log_p
    suffix_losses_buffer_neg[1:b] += log_p
    return sp.special.logsumexp(
        suffix_losses_buffer_pos, axis=0
    ) - sp.special.logsumexp(suffix_losses_buffer_neg, axis=0)
  else:
    return suffix_losses_buffer_pos[0] - suffix_losses_buffer_neg[0]


def compute_privacy_loss(
    strategy: SupportedStrategies,
    sample: np.ndarray,
    noise_multiplier: float,
    c_col: np.ndarray,
    aux: np.ndarray | None = None,
    mode_scale: float = 1.0,
) -> np.ndarray:
  r"""Computes the privacy loss on a sample from the dominating pair.

  This method reports the privacy loss assuming we sample from the distribution
  in the dominating pair where the sensitive example is included. To get the
  privacy loss for when the example is excluded, we simply negate the privacy
  loss for the positive sample.

  Args:
    strategy: The batch selection strategy used to generate the sample.
      Currently, :class:`~jax_privacy.batch_selection.BallsInBinsSampling`,
      :class:`~jax_privacy.batch_selection.CyclicPoissonSampling` (using
      ``INDEPENDENT`` partition type), and
      :class:`~jax_privacy.batch_selection.BMinSepSampling` are supported. See
      `Choquette-Choo et al. (2024) <https://arxiv.org/abs/2410.06266>`__ and
      `Dong and Ganesh (2026) <https://arxiv.org/abs/2602.09338>`__ for
      technical details.
    sample: The samples, generated by :func:`generate_sample`.
    noise_multiplier: The noise multiplier of DP-MF. This is multiplied by the
      clip norm, not accounting for the norm of ``c_col``.
    c_col: The non-zero entries in the first column of C. Should be non-negative
      and 1D. Assumed to match the epoch length of balls-in-bins.
    aux: Auxiliary information needed to compute the privacy loss.
    mode_scale: A scalar :math:`w` by which the modes of the positive mixture
      :math:`P` are multiplied. The returned value is then :math:`\ln(P_w(y) /
      Q(y))`, where :math:`P_w` is the stretched mixture (with :math:`P_1 = P`)
      and :math:`Q` is the negative distribution. This is used to compute
      importance weights for samples drawn from :math:`P_w`. Currently only
      supported for balls-in-bins sampling.

  Returns:
    The privacy loss of the sample, assuming we sample from the distribution in
    the dominating pair where the sensitive example is included.
  """
  if sample.shape[0] != strategy.iterations:
    raise ValueError(
        "sample's size should be (iterations, num_samples), i.e. the first "
        'dimension must match the number of iterations of the strategy. Got '
        f'{sample.shape[0]} and {strategy.iterations}.'
    )
  if mode_scale != 1.0 and not isinstance(
      strategy, batch_selection.BallsInBinsSampling
  ):
    raise ValueError('mode_scale is only supported for balls-in-bins sampling.')
  if isinstance(strategy, batch_selection.BallsInBinsSampling):
    if aux is not None:
      raise ValueError('aux must be None for balls-in-bins sampling.')
    return _compute_balls_in_bins_privacy_loss(
        epoch_length=strategy.cycle_length,
        sample=sample,
        sigma=noise_multiplier,
        c_col=c_col,
        mode_scale=mode_scale,
    )
  elif isinstance(strategy, batch_selection.BMinSepSampling):
    if (strategy.truncated_batch_size is None) != (aux is None):
      raise ValueError(
          'aux for b-min-sep sampling must be set if and only if '
          'strategy.truncated_batch_size is not None.'
      )
    if aux is not None and aux.shape != sample.shape:
      raise ValueError('aux must have the same shape as sample.')
    return _compute_b_min_sep_privacy_loss(
        strategy=strategy,
        samples=sample,
        sigma=noise_multiplier,
        c_col=c_col,
        rest_batch_sizes=aux,
    )
  elif isinstance(strategy, batch_selection.CyclicPoissonSampling):
    if aux is not None:
      raise ValueError('aux must be None for cyclic Poisson sampling.')
    if strategy.partition_type != batch_selection.PartitionType.INDEPENDENT:
      raise ValueError(
          'Only INDEPENDENT partition type is supported for cyclic '
          'Poisson sampling.'
      )
    return _compute_cyclic_poisson_privacy_loss(
        cycle_length=strategy.cycle_length,
        sampling_prob=strategy.sampling_prob,
        sample=sample,
        sigma=noise_multiplier,
        c_col=c_col,
    )
  else:
    raise ValueError(f'Unsupported batch selection strategy: {type(strategy)}')


def get_privacy_loss_sample(
    strategy: SupportedStrategies,
    noise_multiplier: float,
    c_col: np.ndarray,
    seed: Seed = None,
    positive_sample: bool = True,
    num_samples: int = 1,
    dataset_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray | tuple[np.ndarray, np.ndarray]]:
  """Returns sample(s) from DP-BandMF's dominating privacy loss distribution.

  Args:
    strategy: The batch selection strategy to use. Currently,
      :class:`~jax_privacy.batch_selection.BallsInBinsSampling`,
      :class:`~jax_privacy.batch_selection.CyclicPoissonSampling` (using
      ``INDEPENDENT`` partition type), and
      :class:`~jax_privacy.batch_selection.BMinSepSampling` are supported. See
      `Choquette-Choo et al. (2024) <https://arxiv.org/abs/2410.06266>`__ and
      `Dong and Ganesh (2026) <https://arxiv.org/abs/2602.09338>`__ for
      technical details.
    noise_multiplier: The noise multiplier of DP-MF. This is multiplied by the
      clip norm, not accounting for the norm of ``c_col``.
    c_col: The non-zero entries in the first column of C. Should be non-negative
      and 1D.
    seed: The seed to use for sampling.
    positive_sample: If ``True``, we sample from the distribution in the
      dominating pair corresponding to the case where the sensitive example is
      included. Otherwise, we sample from the other case in the dominating pair,
      where the sensitive example is not included.
    num_samples: The number of samples to generate. The default is 1, but it is
      typically much more efficient to generate multiple samples in a single
      call to benefit from vectorization.
    dataset_size: The size of the dataset. Should only be set if accounting for
      the strategy supports truncation, and ``strategy.truncated_batch_size`` is
      not ``None``.

  Returns:
    A tuple of (i) privacy loss of the sample(s), and (ii) either the sample(s)
    or a tuple of (sample(s), auxiliary information needed to compute the
    privacy loss).
  """
  sample = generate_sample(
      strategy=strategy,
      noise_multiplier=noise_multiplier,
      c_col=c_col,
      seed=seed,
      positive_sample=positive_sample,
      num_samples=num_samples,
      dataset_size=dataset_size,
  )
  if isinstance(sample, tuple):
    sample, aux = sample
  else:
    aux = None
  privacy_loss = compute_privacy_loss(
      strategy=strategy,
      sample=sample,
      noise_multiplier=noise_multiplier,
      c_col=c_col,
      aux=aux,
  )
  if not positive_sample:
    privacy_loss = -privacy_loss
  if aux is None:
    return privacy_loss, sample
  else:
    return privacy_loss, (sample, aux)

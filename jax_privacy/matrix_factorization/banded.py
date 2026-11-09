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

"""Class and instances for expressing and optimizing banded strategies."""

from collections.abc import Callable
import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import optimization
from . import sensitivity
from . import streaming_matrix


# Disabling pylint invalid-name to allow mathematical notation including
# single-capital-letter variables for matrices.
# See README.md for notation conventions.
# pylint:disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class ColumnNormalizedBanded:
  """A column-normalized banded lower triangular n x n matrix.

  This matrix class is parameterized by an arbitrary n x b matrix.
  C(params) is obtained by setting the first b bands of C based on params.
  The matrix is normalized to have sensitivity 1 under a single epoch,
  by dividing each column by its respective norm.

  Below we show how params relates to the matrix (before column normalization):

  ::

    params = [a b c]
             [d e f]
             [g h i]
             [j k -]
             [m - -]

    C = [a        ]
        [b d      ]
        [c e g    ]
        [  f h j  ]
        [    i k m]
  """

  params: jnp.ndarray

  @property
  def n(self) -> int:
    return self.params.shape[0]

  @property
  def bands(self) -> int:
    return self.params.shape[1]

  @classmethod
  def from_banded_toeplitz(
      cls, n: int, coefs: jnp.ndarray
  ) -> 'ColumnNormalizedBanded':
    """Construct an instance of this object from banded toeplitz coefficients.

    Args:
      n: the number of training iterations.
      coefs: an array of b toeplitz coefficients defining the strategy.

    Returns:
      A ColumnNormalizedBanded representation of the banded toeplitz matrix.
    """
    bands = coefs.size
    if bands > n or bands < 1:
      raise ValueError(f'len(coefs) must be in the range [1, n], got {bands}')
    coefs = np.asarray(coefs)
    coefs = coefs / np.linalg.norm(coefs)
    params = jnp.broadcast_to(jnp.array(coefs), (n, bands))
    params = jnp.tril(params[::-1])[::-1]  # set the lower right triangle to 0
    return cls(params=params)

  @classmethod
  def default(cls, n: int, bands: int) -> 'ColumnNormalizedBanded':
    """Construct a default instance of this object given n and bands.

    This object is initialized by using the fixed toeplitz strategy proposed
    in [1; Algorithm 1], truncating to :math:`b` entries, and column
    normalizing. It can act as a useful initialization for further optimization.

    [1] https://proceedings.mlr.press/v202/fichtenberger23a/fichtenberger23a.pdf

    Args:
      n: the number of training iterations.
      bands: the number of bands in the strategy.

    Returns:
      A ColumnNormalizedBanded object.
    """
    k = np.arange(bands)
    ratios = (2 * k - 1) / (2 * k)
    ratios[0] = 1
    coefs = np.cumprod(ratios)
    return ColumnNormalizedBanded.from_banded_toeplitz(n, coefs)

  def materialize(self) -> jnp.ndarray:
    I = jnp.arange(self.n)[:, None]
    J = jnp.arange(self.n)[None]
    D = I - J
    indexer = (D + self.bands * J + 1) * (D >= 0) * (D < self.bands)
    C = jnp.append(0, self.params.flatten())[indexer]
    return C / jnp.linalg.norm(C, axis=0)

  def inverse_as_streaming_matrix(
      self,
  ) -> streaming_matrix.StreamingMatrix:
    r"""Create :math:`C^{-1}` as a StreamingMatrix object."""

    def init_fn(abstract_value):
      dtype = jnp.promote_types(abstract_value.dtype, self.params.dtype)
      # *_like preserves shape, dtype, and (if possible) sharding.
      zero = jnp.zeros_like(abstract_value, dtype=dtype)
      buffers = jnp.broadcast_to(zero, (self.bands,) + zero.shape)
      return jnp.array(0, dtype=jnp.int32), buffers

    def next_fn(value, state):
      index, bufs = state
      if self.bands == 1:
        return value, (index + 1, bufs)

      k = index % self.bands
      r = jnp.arange(self.bands)
      row = self.params[index - r, r]
      # Algorithm 9 from Choquette-Choo et al. (2023)
      # https://arxiv.org/abs/2306.08153
      # Compute xi = (value - row[1:] @ bufs[k-r][1:]) / row[0]
      inner = jnp.tensordot(row[1:], bufs[k - r][1:], axes=((0,), (0,)))
      xi = (value - inner) / row[0]
      col_norm = jnp.linalg.norm(self.params[index])
      updated_state = (index + 1, bufs.at[k].set(xi))
      return xi * col_norm, updated_state

    return streaming_matrix.StreamingMatrix.from_array_implementation(
        init_fn, next_fn
    )


def minsep_sensitivity_squared(
    strategy: ColumnNormalizedBanded,
    min_sep: int,
    max_participations: int | None = None,
    n: int | None = None,
    skip_checks: bool = False,
) -> int:
  """Returns the sensitivity of the ColumnNormalizedBanded strategy.

  With max_participations = 1 (and any min_sep, say min_sep = 1), this is the
  same as single participation.

  Args:
    strategy: The strategy matrix defining the mechanism.
    min_sep: The minimum separation between two participation of a worst-case
      client/sample. Note that we use the definition in `Choquette-Choo et al.
      (2023) <https://arxiv.org/abs/2306.08153>`_. For a user participating on
      iteration :math:`i` and then again on iteration :math:`j`, the separation
      is :math:`j - i`; that is, a ``min_sep`` of 1 allows participation on
      every iteration.
    max_participations: The maximum participation of a worst-case user. The
      default value ``None`` allows the max number of possible participations.
    n: Optional, the size of the matrix :math:`C`. If ``None``, the size of the
      matrix is equal to the number of coefficients in ``strategy``.
    skip_checks: If ``True``, don't perform input verification which may not be
      supported in jitted contexts.

  Returns:
    The sensitivity squared.
  """
  bands = strategy.bands
  n = n or strategy.n
  max_participations = sensitivity.minsep_true_max_participations(
      n, min_sep, max_participations
  )
  if not skip_checks:
    if min_sep < bands:
      raise ValueError(
          f'{min_sep=} must be greater than or equal to {bands=}. This error is'
          ' usually indicative of a mis-configuration of the strategy for the'
          ' participation pattern.  If it is intentional, please use'
          ' sensitivity.get_sensitivity_banded.'
      )
    if n > strategy.n:
      raise ValueError(f'{n=} must be less than or equal to {strategy.n=}.')
  return max_participations


def _checkpoint_count(n: int, bands: int, memory_limit_gb: float = 4) -> int:
  """Largest power-of-two checkpoint count that fits in ``memory_limit_gb``.

  Extra checkpoints beyond ``n`` cannot reduce rematerialization, and
  ``log2(0)`` overflows when the working set exceeds the memory budget.
  """
  used = max(int(bands) * int(n), 1)
  limit = int((2**30) * memory_limit_gb // 8)
  if used >= limit:
    return 1
  max_pow = int(np.log2(limit // used) - 1)
  checkpoints = 2 ** max(max_pow, 0)
  return min(max(checkpoints, 1), int(n))


def _equinox_scan_fn(n: int, bands: int, memory_limit_gb: float = 4):
  """Checkpointed scan function for memory-efficient backpropagation."""
  # We do not want to take a hard dependence on equinox, so we import it here.
  import equinox  # pylint: disable=g-import-not-at-top, import-outside-toplevel

  return functools.partial(
      equinox.internal.scan,
      kind='checkpointed',
      checkpoints=_checkpoint_count(n, bands, memory_limit_gb),
  )


def _dinosaur_scan_fn(n: int, bands: int, memory_limit_gb: float = 4):
  """Checkpointed scan function for memory-efficient backpropagation."""
  # We do not want to take a hard dependence on dinosaur, so we import it here.
  from dinosaur import time_integration  # pylint: disable=g-import-not-at-top, import-outside-toplevel

  max_checkpoints = _checkpoint_count(n, bands, memory_limit_gb)
  candidates = range(max_checkpoints, 0, -1)
  num_checkpoints = next(filter(lambda d: n % d == 0, candidates))
  nested_lengths = [num_checkpoints, n // num_checkpoints]
  return functools.partial(
      time_integration.nested_checkpoint_scan,
      nested_lengths=nested_lengths,
  )


# TODO: b/329444015 - document the definition of per query squared error.
@functools.partial(jax.jit, static_argnums=[1, 2])
def per_query_error(
    C: ColumnNormalizedBanded,
    A: streaming_matrix.StreamingMatrix | None = None,
    scan_fn: Any = jax.lax.scan,
) -> jnp.ndarray:
  r"""Computes expected per-query squared error of a strategy.

  Specifically, this function computes the row-wise :math:`L_2^2` norm of
  :math:`B = A C^{-1}`.

  Since :math:`C` is column normalized, this error function can be used as
  a loss function, since sensitivity is constant for ColumnNormalizedBanded
  strategies for both single-participation and multi-participation settings,
  as long as the number of bands in :math:`C` is less than or equal to the
  (minimum) separation between contributions from the same user.

  If you need to backpropagate through this function, you can use the
  ``'equinox'`` or ``'dinosaur'`` scan functions to make the scan
  checkpointed, which allows the scan to be performed for large ``n`` without
  OOMing the accelerator.

  Args:
    C: the strategy matrix, represented implicitly.
    A: The workload matrix, represented implicitly.
    scan_fn: A function with the same signature as jax.lax.scan.

  Returns:
    The per query expected squared error of the strategy on the workload,
    represented as an array of length ``n``.
  """
  if scan_fn == 'equinox':
    scan_fn = _equinox_scan_fn(C.n, C.bands)
  elif scan_fn == 'dinosaur':
    scan_fn = _dinosaur_scan_fn(C.n, C.bands)

  A = A or streaming_matrix.prefix_sum()
  B = A @ C.inverse_as_streaming_matrix()
  return B.row_norms_squared(C.n, scan_fn=scan_fn)


# TODO: b/329444015 - rethink how the objective should be specified
def optimize(
    n: int,
    *,
    bands: int,
    C: ColumnNormalizedBanded | None = None,
    A: streaming_matrix.StreamingMatrix | None = None,
    max_optimizer_steps: int = 100,
    reduction_fn: Callable[[jnp.ndarray], jnp.ndarray] = jnp.mean,
    scan_fn: Any = jax.lax.scan,
    callback: optimization.CallbackFnType = lambda _: None,
) -> ColumnNormalizedBanded:
  r"""Optimize the strategy using a gradient-based method.

  Note that this function benefits substantially from GPUs.  This function
  is primarily supported to aid in reproducing results from
  `McKenna (2024) <https://arxiv.org/abs/2405.15913>`_  In practice, we
  recommend using a
  banded Toeplitz strategy instead (see
  :func:`~jax_privacy.matrix_factorization.toeplitz.optimize_banded_toeplitz`),
  which are <0.5% suboptimal in the regimes of most interest
  (:math:`n \ge 1000`, :math:`b \le 32`).

  The strategies produced by this procedure can be used in both single- and
  multi-participation settings -- both (k, b)-min-sep and (k, b)-fixed epoch
  order, as described in `Choquette-Choo et al. (2023)
  <https://arxiv.org/abs/2306.08153>`_, as long as the
  number of bands in :math:`C` is less than or equal to the (minimum)
  separation between contributions from the same user.

  Args:
    n: The number of training iterations the strategy is configured for.
    bands: The number of bands in the strategy.
    C: The initial strategy to be optimized.
    A: The target workload.
    max_optimizer_steps: The maximum number of iterations to optimize for.
    reduction_fn: A function that converts per query squared errors to a scalar.
      Use ``jnp.mean`` to optimize mean-squared-error, ``jnp.max`` to optimize
      max squared error, or ``lambda v: v[-1]`` to optimize last iterate squared
      error.
    scan_fn: Either ``'equinox'``, ``'dinosaur'``, or a function with the same
      signature as :func:`jax.lax.scan`.  Using ``'equinox'`` or ``'dinosaur'``
      is helpful for doing strategy optimization on GPUs for large ``n``, since
      it allows the scan function used internally by ``per_query_error`` to be
      checkpointed, avoiding OOM errors during backpropagation.
    callback: A function to call after each optimization iteration. See
      :func:`~jax_privacy.matrix_factorization.optimization.optimize` for
      details.

  Returns:
    An optimized strategy having the same structure as ``C``.
  """

  A = A or streaming_matrix.prefix_sum()
  C = C or ColumnNormalizedBanded.default(n, bands)
  loss_fn = lambda C: reduction_fn(per_query_error(C, A=A, scan_fn=scan_fn))
  return optimization.optimize(
      loss_fn, C, max_optimizer_steps=max_optimizer_steps, callback=callback
  )

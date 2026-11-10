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

"""Library for working with general Toeplitz matrices.

In particular, lower-triangular Toeplitz matrices used in DP matrix
factorization algorithms.  Generally, these functions operate directly in terms
of the `n` Toeplitz coefficients, and hence can be much more efficient than
doing the calculations on the materialized `n**2` matrices.
"""

import concurrent
import dataclasses
import functools
import math
from typing import Any, Callable

import dp_accounting
import jax
import jax.numpy as jnp
import numpy as np

from .. import _validate
from . import optimization
from . import sensitivity
from . import streaming_matrix

# Disabling pylint invalid-name to allow mathematical notation including
# single-capital-letter variables for matrices.
# See README.md for notation conventions.
# pylint:disable=invalid-name


def _l2_norm_squared(x: jax.Array):
  return jnp.inner(x, x)


def _reconcile(coef: jax.Array, n: int | None = None) -> tuple[jax.Array, int]:
  """Reconciles the Toeplitz coefficients with the matrix size.

  Args:
    coef: The nonzero coefficients of a lower-triangular Toeplitz matrix C, that
      is, ``coef`` are the leading nonzero entries of C[:, 0]. C is of size n x
      n; if len(coef) < n, the remaining coefficients are assumed to be zero. If
      len(coef) > n, then only the first n coefficients are used.
    n: Optional, the size of the matrix C (see ``coef`` above). If None, the
      size of the matrix is equal to the number of coefficients.

  Returns:
    A tuple (coef, n) where n >= len(coef). If a non-None ``n`` keyword argument
    was provided, then this function will return the same ``n`` value.
  """
  n = n or len(coef)
  coef = jnp.array(coef)[:n]  # Drop extra coefficients if needed.
  return coef, n


def pad_coefs_to_n(coef: jax.Array, n: int | None = None) -> jax.Array:
  """Materializes length-n Toeplitz coefficients."""
  coef, n = _reconcile(coef, n)
  return jnp.zeros(n).at[0 : len(coef)].set(coef)


def inverse_as_streaming_matrix(
    coef: jax.Array,
    column_normalize_for_n: int | None = None,
) -> streaming_matrix.StreamingMatrix:
  r"""Create :math:`C^{-1}` as a StreamingMatrix object.

  If ``column_normalize_for_n`` is ``None``, the returned object represents
  :math:`C^{-1}`, the inverse of an arbitrarily large banded Toeplitz matrix
  :math:`C` with coefficients ``coef``.

  If ``column_normalize_for_n`` is finite, the returned object represents
  :math:`C^{-1}`, the inverse of a banded matrix :math:`C` of the given size
  :math:`n \times n`, formed by taking the banded Toeplitz matrix with
  coefficients ``coef`` and re-scaling each column so it has :math:`L_2` norm
  1.0. We recommend setting ``column_normalize_for_n`` in centralized training
  scenarios where ``n`` is known in advance.

  Formal Guarantees:
    * The maximum :math:`L_2` norm of the strategy matrix :math:`C` is
      :math:`\|\text{coef}\|_2` if ``column_normalize_for_n`` is ``None``, and
      1.0 if it is not ``None``.
    * The strategy matrix :math:`C` corresponding to the returned noising matrix
      :math:`C^{-1}` is :math:`b`-banded, where :math:`b = \text{coef.size}`.

  This implementation is based on Algorithm 9 from
  `Choquette-Choo et al. (2023) <https://arxiv.org/abs/2306.08153>`_

  Args:
    coef: The Toeplitz coefficients of the strategy.
    column_normalize_for_n: If given, the returned object represents the inverse
      of a *column-normalized* banded Toeplitz matrix of the given size.
      Otherwise, it represents the inverse of an ordinary banded Toeplitz
      matrix. If not ``None``, the supplied ``coef`` must have an :math:`L_2`
      norm of 1.0 (otherwise, column normalization could change the sensitivity
      of the implied :math:`C` matrix).

  Returns:
    A StreamingMatrix object representing :math:`C^{-1}`.
  """
  coef, _ = _reconcile(coef, column_normalize_for_n)
  bands = coef.shape[0]

  def init(abstract_yi):
    dtype = jnp.promote_types(abstract_yi.dtype, coef.dtype)
    # *_like preserves shape, dtype, and (if possible) sharding.
    zero = jnp.zeros_like(abstract_yi, dtype=dtype)
    return jnp.broadcast_to(zero, (bands - 1,) + zero.shape)

  def _next(yi, state):
    if bands == 1:
      return yi / coef[0], state
    inner = jnp.tensordot(coef[1:], state, axes=1)
    xi = (yi - inner) / coef[0]
    return xi, jnp.roll(state, 1, axis=0).at[0].set(xi)

  Cinv = streaming_matrix.StreamingMatrix.from_array_implementation(init, _next)

  if column_normalize_for_n is not None:
    # 1/s scale on the cols of C translates to scale of s on the rows of C^{-1}.
    full_coef = jnp.pad(coef, (0, column_normalize_for_n - bands))
    col_norms = jnp.sqrt(jnp.cumsum(full_coef**2))[::-1]
    Cinv = streaming_matrix.scale_rows_and_columns(Cinv, row_scale=col_norms)

  return Cinv


def optimal_max_error_strategy_coefs(n: int) -> jax.Array:
  # pylint: disable=line-too-long
  r"""Returns the coefs of the optimal Toeplitz strategy matrix :math:`C` for max error.

  These coefficients were introduced by Fichtenberger, Henzinger, and Upadhyay
  in `Constant Matters: Fine-grained Error Bound on Differentially Private
  Continual Observation <https://arxiv.org/abs/2202.11205>`_,
  and proved to be optimal for max error under single participations by
  Dvijotham, McMahan, Pillutla, Steinke, and Thakurta in
  "Efficient and Near-Optimal Noise Generation for Streaming Differential
  Privacy" (`Dvijotham et al. (2024) <https://arxiv.org/abs/2404.16706>`_).

  Args:
    n: The number of coefficients to return.

  Returns:
    The coefficients of the lower-triangular Toeplitz matrix :math:`C` that
    factorizes the prefix sum matrix :math:`A` as :math:`A = C @ C`.
  """
  k = jnp.arange(n)
  return jnp.cumprod(((2 * k - 1) / (2 * k)).at[0].set(1))


def optimal_max_error_noising_coefs(n: int) -> jax.Array:
  # pylint: disable=line-too-long
  r"""Returns the coefs of the optimal Toeplitz noise matrix for max error.

  Args:
    n: The number of coefficients to return.

  Returns:
    The coefficients of the lower-triangular Toeplitz matrix :math:`C^{-1}`
    that is the inverse of the matrix returned by
    :func:`~jax_privacy.matrix_factorization.toeplitz.optimal_max_error_strategy_coefs`.
  """
  # This factorization of A = B C where A is the prefix-sum matrix is symmetric,
  # in that C = B = A C^{-1}, so C^{-1} = A^{-1} C, where A^{-1}
  # computes differences.
  c = optimal_max_error_strategy_coefs(n)
  return c.at[1:n].subtract(c[:-1])


def materialize_lower_triangular(
    coef: jax.Array, n: int | None = None
) -> jax.Array:
  r"""Creates a lower-triangular Toeplitz matrix.

  Example: If `coef = [a, b, c]` and `n = 6`, then this method returns::

    [a 0 0 0 0 0]
    [b a 0 0 0 0]
    [c b a 0 0 0]
    [0 c b a 0 0]
    [0 0 c b a 0]
    [0 0 0 c b a]

  Args:
    coef: The nonzero coefficients of a lower-triangular Toeplitz matrix
      :math:`C`, that is, ``coef`` are the leading nonzero entries of
      :math:`C[:, 0]`. :math:`C` is of size :math:`n \times n`; if ``len(coef) <
      n``, the remaining coefficients are assumed to be zero. If ``len(coef) >
      n``, then only the first ``n`` coefficients are used.
    n: Optional, the size of the matrix :math:`C` (see ``coef`` above). If
      ``None``, the size of the matrix is equal to the number of coefficients.

  Returns:
    The lower-triangular Toeplitz matrix.
  """
  coef = pad_coefs_to_n(coef, n)
  return jax.scipy.linalg.toeplitz(coef, r=jnp.zeros_like(coef))


def solve_banded(coef: jax.Array, rhs: jax.Array) -> jax.Array:
  # pylint: disable=line-too-long
  r"""Solve the linear system :math:`T_{\text{coef}} x = \text{rhs}` for :math:`x` for Toeplitz matrix :math:`T`.

  Specifically, :math:`T_{\text{coef}}` is a lower triangular banded Toeplitz
  matrix.

  Note we want to be able to back-propagate gradients through this function,
  hence we cannot use :func:`scipy.linalg.solve_toeplitz`.

  Example: ``coef = [a, b, c]``, ``rhs = [1, 1, 1, 1, 1, 1]``, we solve the
  following system for x::

    [a 0 0 0 0 0] [x_0]   [1]
    [b a 0 0 0 0] [x_1]   [1]
    [c b a 0 0 0] [x_2] = [1]
    [0 c b a 0 0] [x_3]   [1]
    [0 0 c b a 0] [x_4]   [1]
    [0 0 0 c b a] [x_5]   [1]

  Args:
    coef: The nonzero coefficients of a lower-triangular Toeplitz matrix
      :math:`C`, that is, ``coef`` are the leading nonzero entries of
      :math:`C[:, 0]`. :math:`C` is of size :math:`n \times n` where ``n =
      len(rhs)`` (see below); if ``len(coef) < n``, the remaining coefficients
      are assumed to be zero. If ``len(coef) > n``, then only the first ``n``
      coefficients are used.
    rhs: The right hand side vector, of length ``n``.

  Returns:
    The solution to the linear system Toeplitz(coef, n) x = rhs.
  """
  return inverse_as_streaming_matrix(coef) @ rhs


def multiply(
    lhs_coef: jax.Array,
    rhs_coef: jax.Array,
    n: int | None = None,
) -> jax.Array:
  r"""Computes the matrix product of two lower-triangular Toeplitz matrices.

  Args:
    lhs_coef: The nonzero coefficients of a lower-triangular Toeplitz matrix
      :math:`L`, that is, ``lhs_coef`` are the leading nonzero entries of
      :math:`L[:, 0]`. :math:`L` is of size :math:`n \times n`; if
      ``len(lhs_coef) < n``, the remaining coefficients are assumed to be zero.
      If ``len(lhs_coef) > n``, then only the first ``n`` coefficients are used.
    rhs_coef: The nonzero coefficients of a lower-triangular Toeplitz matrix
      :math:`R`, under the same conventions as ``lhs_coef``.
    n: Optional, the size of the matrices :math:`L` and :math:`R` (see ``coef``
      above). If ``None``, the size of the matrices is equal to the number of
      coefficients.

  Returns:
    The coefficients of the lower-triangular Toeplitz matrix :math:`LR` where
    :math:`L = \text{materialize\_lower\_triangular}(\text{lhs\_coef}, n)` and
    :math:`R = \text{materialize\_lower\_triangular}(\text{rhs\_coef}, n)`.
  """
  if n is None:
    _validate.same_shape(lhs_coef=lhs_coef, rhs_coef=rhs_coef)
  lhs_coef, n = _reconcile(lhs_coef, n)
  rhs_coef, _ = _reconcile(rhs_coef, n)
  return jnp.convolve(
      lhs_coef, rhs_coef, mode='full', precision=jax.lax.Precision.HIGHEST
  )[:n]


def inverse_coef(coef: jax.Array, n: int | None = None) -> jax.Array:
  r"""Finds the inverse coefficients of a lower-triangular Toeplitz matrix.

  If :math:`C` is a lower-triangular Toeplitz matrix, then so is :math:`C^{-1}`;
  this function returns the Toeplitz coefficients of this inverse.

  Args:
    coef: The nonzero coefficients of a lower-triangular Toeplitz matrix
      :math:`C`, that is, ``coef`` are the leading nonzero entries of
      :math:`C[:, 0]`. :math:`C` is of size :math:`n \times n`; if ``len(coef) <
      n``, the remaining coefficients are assumed to be zero. If ``len(coef) >
      n``, then only the first ``n`` coefficients are used.
    n: Optional, the size of the matrix :math:`C` (see ``coef`` above). If
      ``None``, the size of the matrix is equal to the number of coefficients.

  Returns:
    The Toeplitz coefficients of :math:`C^{-1}`, of length n.
  """
  coef, n = _reconcile(coef, n)
  return solve_banded(coef, jnp.zeros(n).at[0].set(1))


def sensitivity_squared(coef: jax.Array, n: int | None = None) -> jax.Array:
  """Sensitivity^2 under single participation."""
  coef, _ = _reconcile(coef, n)
  return _l2_norm_squared(coef)


def minsep_sensitivity_squared(
    strategy_coef: jax.Array,
    min_sep: int,
    *,
    n: int,
    max_participations: int | None = None,
) -> jax.Array:
  r"""Returns the sensitivity of the Toeplitz matrix.

  With max_participations = 1 (and any min_sep, say min_sep = 1), this is the
  same as single participation.

  The result is exact (tight) when the Toeplitz coefficients are non-negative
  and non-increasing. If the coefficients do not satisfy these assumptions, they
  are projected onto the set of non-negative non-increasing sequences (via
  absolute value and reverse cumulative max), and the result is an upper bound
  on the true sensitivity.

  Reference: While this code actually predates the paper, this result is
  published in https://arxiv.org/pdf/2405.13763, Theorem 2.

  Args:
    strategy_coef: The nonzero coefficients of the Toeplitz matrix :math:`C`
      used in the matrix mechanism with factorization :math:`A = BC`. That is,
      ``coef`` are the leading nonzero entries of :math:`C[:, 0]`. :math:`C` is
      of size :math:`n \times n`; if ``len(coef) < n``, the remaining
      coefficients are assumed to be zero. If ``len(coef) > n``, then only the
      first ``n`` coefficients are used.
    min_sep: The minimum separation between two participation of a worst-case
      client/sample. Note that we use the definition in `Choquette-Choo et al.
      (2023) <https://arxiv.org/abs/2306.08153>`_. For a user participating on
      iteration :math:`i` and then again on iteration :math:`j`, the separation
      is :math:`j - i`; that is, a ``min_sep`` of 1 allows participation on
      every iteration.
    n: The size of the matrix :math:`C` (see ``coef`` above).
    max_participations: The maximum participation of a worst-case user.

  Returns:
    The sensitivity squared. This is exact when ``strategy_coef`` is
    non-negative and non-increasing, and an upper bound otherwise.
  """
  coef, n = _reconcile(strategy_coef, n)
  _validate.positive(min_sep=min_sep)

  # Project onto non-negative non-increasing coefficients. This is a no-op
  # when the coefficients already satisfy these assumptions, and yields an
  # upper bound on sensitivity otherwise.
  coef = jax.lax.cummax(jnp.abs(coef)[::-1])[::-1]

  k = sensitivity.minsep_true_max_participations(
      n=n, min_sep=min_sep, max_participations=max_participations
  )
  # When the Toeplitz coefficients are non-negative and non-increasing, the
  # worst-case for sensitivity is (up to) k participations separated by
  # exactly b.  We use a difference of cumsums to do this in O(n) time.

  padding = (min_sep - n) % min_sep
  coef = jnp.pad(coef, (0, n - coef.size + padding))
  vector = coef.reshape(-1, min_sep).cumsum(axis=0).flatten()
  vector = vector.at[min_sep * k :].set(
      vector[min_sep * k :] - vector[: -min_sep * k]
  )
  return vector[:n] @ vector[:n]


def per_query_error(
    *,
    strategy_coef: jax.Array | None = None,
    noising_coef: jax.Array | None = None,
    n: int | None = None,
    workload_coef: jax.Array | None = None,
) -> jax.Array:
  """Expected per-query squared error for a (banded) Toeplitz mechanism.

  This function returns the squared error on a per-iteration basis.  To compute
  the mean-squared error or max squared error, use jnp.mean or jnp.max on the
  output of this function. Note: for toeplitz workloads / strategies, the max
  squared error is equal to the last iterate squared error, and might be more
  efficient to compute under jax transformations.

  Exactly one of ``strategy_coef`` and ``noising_coef`` should be provided.

  Args:
    strategy_coef: Toeplitz coefficients of the strategy matrix.
    noising_coef: Toeplitz coefficients of the noising matrix.
    n: The size of the implied matrices (defaults to the length of the Toeplitz
      coefficient array).
    workload_coef: Toeplitz coefficients of the workload matrix. Defaults to the
      vector of 1s, corresponding to the prefix matrix. If this is longer than
      ``n``, the extra entries are ignored (even if ``n`` is inferred from the
      length of the ``strategy_coef`` or ``noising_coef``).

  Returns:
    The expected per-query squared error, an array of length n.
  """
  if (strategy_coef is None) == (noising_coef is None):
    raise ValueError('Specify exactly one of strategy_coef or noising_coef.')

  if strategy_coef is not None:
    strategy_coef, n = _reconcile(strategy_coef, n)
    if workload_coef is not None:
      workload_coef = pad_coefs_to_n(workload_coef, n)
    else:
      workload_coef = jnp.ones(n)
    B_coef = solve_banded(strategy_coef, workload_coef)
  else:
    assert noising_coef is not None
    noising_coef, n = _reconcile(noising_coef, n)
    noising_coef = jnp.pad(noising_coef, (0, n - noising_coef.shape[0]))
    if workload_coef is None:
      # This is more efficient than explicitly multiplying by the prefix matrix.
      B_coef = jnp.cumsum(noising_coef)
    else:
      B_coef = multiply(workload_coef, noising_coef, n=n)

  return jnp.cumsum(B_coef**2)


def loss(
    strategy_coef: jax.Array,
    n: int | None = None,
    reduction_fn: Callable[[jax.Array], jax.Array] = jnp.mean,
) -> jax.Array:
  r"""Error of :math:`C` on prefix workload under single participation.

  See Scaling up the Amplified Banded Matrix Factorization Mechanism for
  Differentially Private ML (`McKenna (2024)
  <https://arxiv.org/abs/2405.15913>`_) for details.

  Args:
    strategy_coef: The nonzero coefficients of the Toeplitz matrix :math:`C`
      used in the matrix mechanism with factorization :math:`A = BC`. That is,
      ``coef`` are the leading nonzero entries of :math:`C[:, 0]`. :math:`C` is
      of size :math:`n \times n`; if ``len(coef) < n``, the remaining
      coefficients are assumed to be zero. If ``len(coef) > n``, then only the
      first ``n`` coefficients are used.
    n: Optional, the size of the matrix :math:`C` (see ``coef`` above). If
      ``None``, the size of the matrix is equal to the number of coefficients.
    reduction_fn: A function that converts per query squared errors to a scalar.
      Use ``jnp.mean`` to optimize mean-squared-error, ``jnp.max`` to optimize
      max squared error (which is equivalent to ``lambda v: v[-1]`` in this
      case).

  Returns:
    The total squared error times sensitivity of the toeplitz :math:`C` on the
    prefix-sum workload under single participation.
  """
  # Special property of Toeplitz matrices: max error occurs on last iterate.
  if reduction_fn is jnp.max:
    reduction_fn = lambda v: v[-1]
  # This is the maximum column norm of C, i.e., single-epoch sensitivity.
  strategy_coef, n = _reconcile(strategy_coef, n)
  error = reduction_fn(per_query_error(strategy_coef=strategy_coef, n=n))
  sens_squared = sensitivity_squared(strategy_coef, n)
  return error * sens_squared


def optimize_banded_toeplitz(
    n: int,
    bands: int,
    strategy_coef: jax.Array | None = None,
    max_optimizer_steps: int = 250,
    reduction_fn: Callable[[jax.Array], jax.Array] = jnp.mean,
) -> jax.Array:
  """Optimize over the space of banded Toeplitz strategies on a Prefix workload.

  The banded toeplitz strategies produced by this function can be used for
  both the single-participation setting and the multi-participation setting,
  (including both the ``fixed_epoch_order`` and ``min_sep`` participation
  schemas; see README.md) as long as the (minimum) separation between
  contributions from the same user is at least the number of bands provided.
  See `Choquette-Choo et al. (2023) <https://arxiv.org/abs/2306.08153>`_ for
  more details.

  If used with a different participation pattern (e.g., (k, b)-minsep where
  b is less than the number of bands, sensitivity can be computed post-hoc
  using e.g.
  :func:`~jax_privacy.matrix_factorization.toeplitz.minsep_sensitivity_squared`.
  This should not be necessary in centralized training regimes where the exact
  participation pattern should be known in advance, however.

  Args:
    n: the number of iterations that defines the workload.
    bands: The number of bands in the Toeplitz matrix.
    strategy_coef: Optional toeplitz coefficients to initialize optimization.
    max_optimizer_steps: The maximum number of LBFGS iterations.
    reduction_fn: A function that converts per query squared errors to a scalar.
      Use jnp.mean to optimize mean-squared-error, jnp.max to optimize max
      squared error, or lambda v: v[-1] to optimize last iterate squared error.
      Defaults to jnp.mean.

  Returns:
    The coefficients of the optimal banded Toeplitz strategy, guaranteed to
    have L2 norm 1.
  """
  loss_fn = functools.partial(loss, n=n, reduction_fn=reduction_fn)

  if strategy_coef is None:
    strategy_coef = optimal_max_error_strategy_coefs(bands)
  _validate.equal(bands, strategy_coef_len=strategy_coef.shape[0])

  params = optimization.optimize(
      loss_fn, strategy_coef, max_optimizer_steps=max_optimizer_steps
  )
  return params / jnp.linalg.norm(params)


def _factors(n):
  result = functools.reduce(
      list.__add__,
      ([i, n // i] for i in range(1, math.isqrt(n) + 1) if n % i == 0),
  )
  # Return a sorted list, remove duplicates.
  return list(sorted(set(result)))


@dataclasses.dataclass(frozen=True)
class _AmplifiedBandMFHelper:
  """A convenience class for building an amplified BandMF mechanism.

  This class is primarily used to implement
  ``optimize_banded_toeplitz_for_amplifications`` below.
  """

  n: int
  dataset_size: int
  batch_size: int  # The (expected) batch size.

  epsilon: float
  delta: float

  reduction_fn: Callable[[jax.Array], jax.Array] = jnp.max
  make_fresh_accountant: Callable[[], dp_accounting.PrivacyAccountant] = (
      dp_accounting.rdp.RdpAccountant
  )

  @property
  def max_bands(self):
    """The largest number of bands b possible while achieving b-min-sep."""
    return self.dataset_size // self.batch_size

  def sensitivity(self, coef: jax.Array):
    """The sensitivity of the banded Toeplitz matrix C."""
    return float(jnp.sqrt(sensitivity_squared(coef, n=self.n)))

  def total_noise_multiplier(self, bands: int):
    """The total noise multiplier needed to achieve the privacy target."""
    # It is preferable if n % bands == 0.
    max_participations = math.ceil(self.n / bands)

    # It is also preferable if dataset_size % bands == 0.
    subset_size = self.dataset_size // bands
    sampling_probability = self.batch_size / subset_size
    max_noise_multiplier = int(100 * math.sqrt(self.batch_size))

    def dpsgd_event(noise_multiplier):
      one_round_event = dp_accounting.PoissonSampledDpEvent(
          sampling_probability=sampling_probability,
          event=dp_accounting.GaussianDpEvent(
              noise_multiplier=noise_multiplier
          ),
      )
      return dp_accounting.SelfComposedDpEvent(
          event=one_round_event, count=max_participations
      )

    return dp_accounting.calibrate_dp_mechanism(
        # ADD_OR_REMOVE_ONE is the default neighboring relation.
        make_fresh_accountant=self.make_fresh_accountant,
        make_event_from_param=dpsgd_event,
        target_epsilon=self.epsilon,
        target_delta=self.delta,
        # We have to hack things a little bit so RDP accounting doesn't blow up.
        bracket_interval=dp_accounting.ExplicitBracketInterval(
            0.01, max_noise_multiplier
        ),
    )

  def required_stddev(self, coef: jax.Array):
    """The stddev of the uncorrelated noise Z required.

    That is, passing this stddev to
    :func:`~jax_privacy.noise_addition.matrix_factorization_privatizer`
    should achieve the (epsilon, delta)-DP guarantee.

    Args:
      coef: The coefficients of the banded Toeplitz matrix C.

    Returns:
      The stddev of the uncorrelated noise Z required.
    """
    total_nm = self.total_noise_multiplier(bands=len(coef))
    return total_nm * self.sensitivity(coef)

  def amplified_bandmf_loss(self, coef: jax.Array):
    """The loss in the estimate of the *average* prefix sum."""
    error_times_sens = jnp.sqrt(
        loss(strategy_coef=coef, n=self.n, reduction_fn=self.reduction_fn)
    )
    # Note: loss = error * single_participation_sensitivity
    # We would normally take
    #  stddev = total_nm * single_participation_sensitivity
    #  loss = error * stddev / batch_size
    # The following is equivalent.
    total_nm = self.total_noise_multiplier(bands=len(coef))
    return error_times_sens * total_nm / self.batch_size

  def compute_loss_for_bands(
      self,
      bands_list: list[int] | None = None,
      max_workers: int | None = None,
      **optimizer_kwargs,
  ) -> list[dict[str, Any]]:
    # pylint: disable=line-too-long
    """Computes the loss for each value in a list of possible bands.

    Args:
      bands_list: The list of possible bands to compute the loss for. If None,
        all factors of ``n`` less than the maximum number of possible bands are
        considered.
      max_workers: The maximum number of workers to use, passed to the
        ThreadPoolExecutor.
      **optimizer_kwargs: Keyword arguments passed to
        :func:`~jax_privacy.matrix_factorization.toeplitz.optimize_banded_toeplitz`.

    Returns:
      A list of dictionaries, one for each value in ``bands_list``. Each
      dictionary contains the following keys:
        - ``bands``: The number of bands.
        - ``coef``: The coefficients of the banded Toeplitz matrix C.
        - ``loss``: The loss of the estimate of the *average* prefix sum.
    """
    assert 1 <= self.batch_size <= self.dataset_size
    if bands_list is None:
      bands_list = [
          b for b in _factors(self.dataset_size) if b <= self.max_bands
      ]

    if max(bands_list) > self.max_bands:
      raise ValueError(
          f'bands_list={bands_list} contains a value that exceeds '
          f'max_bands={self.max_bands}'
      )

    def run_task(b):
      coef = optimize_banded_toeplitz(
          n=self.n, bands=b, reduction_fn=self.reduction_fn, **optimizer_kwargs
      )
      loss_value = self.amplified_bandmf_loss(coef)
      return {'bands': b, 'coef': coef, 'loss': loss_value}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      return list(executor.map(run_task, bands_list))

  def optimize_bands(self, **optimizer_kwargs) -> dict[str, Any]:
    """Returns best result (in terms of loss) from compute_loss_for_bands."""
    results = self.compute_loss_for_bands(**optimizer_kwargs)
    losses = np.array([d['loss'] for d in results])
    best_idx = np.argmin(losses)
    return results[best_idx]


def optimize_coefs_for_amplifications(
    n: int,
    *,
    dataset_size: int,
    expected_batch_size: int,
    epsilon: float,
    delta: float,
    max_optimizer_steps: int = 250,
    reduction_fn: Callable[[jax.Array], jax.Array] = jnp.mean,
) -> tuple[jax.Array, float]:
  # pylint: disable=line-too-long
  """Select num_bands (and coefs) to minimize loss subject to a privacy target.

  Following Theorem 4 of `Choquette-Choo et al. (2023)
  <https://arxiv.org/abs/2306.08153>`_, this function
  (approximately) minimizes the loss_fn assuming privacy amplification under
  block-cyclic Poisson sampling (Algorithm 2 of
  `Choquette-Choo et al. (2023)
  <https://arxiv.org/abs/2306.08153>`_). A smaller
  number of bands allows more
  benefit from amplification, while a larger number of bands allows more benefit
  from correlated noise.

  Notes:
   - This function only optimizes over numbers of bands that evenly divide
     ``n``, as this is generally preferable. Hence, it is recommended to choose
     ``n`` so it has well spaced factors; powers of 2 are particularly useful.
   - This function delegates to
     :func:`~jax_privacy.matrix_factorization.toeplitz.optimize_banded_toeplitz`
     to actually optimize for the coefficients at a given number of bands.
     Hence, column normalization is not directly supported, but the final
     returned strategy can always be used with column normalization.

  Args:
    n: the number of iterations that defines the workload.
    dataset_size: The size of the dataset.
    expected_batch_size: The target batch size (so for example if we were
      Poisson sampling from the whole dataset, the sampling probability would be
      ``expected_batch_size / dataset_size``).
    epsilon: The privacy target is (epsilon, delta)-DP.
    delta: The privacy target is (epsilon, delta)-DP.
    max_optimizer_steps: The maximum number of LBFGS iterations, passed to
      :func:`~jax_privacy.matrix_factorization.toeplitz.optimize_banded_toeplitz`.
    reduction_fn: A function that converts per query squared errors to a scalar.
      Use jnp.mean to optimize mean-squared-error, jnp.max to optimize max
      squared error, or lambda v: v[-1] to optimize last iterate squared error.

  Returns:
    A tuple ``(coefs, stddev)`` where:
      - ``coefs`` are the coefficients of a banded Toeplitz strategy; the number
         of bands chosen is simply the length of the returned coefficients.
      - ``stddev`` is the stddev of the uncorrelated noise Z required to achieve
        the privacy target. Passing this stddev to
        :func:`~jax_privacy.noise_addition.matrix_factorization_privatizer`
        should achieve the (epsilon, delta)-DP guarantee.
  """
  helper = _AmplifiedBandMFHelper(
      n, dataset_size, expected_batch_size, epsilon, delta, reduction_fn
  )
  coef = helper.optimize_bands(max_optimizer_steps=max_optimizer_steps)['coef']
  stddev = helper.required_stddev(coef)
  return coef, stddev


def banded_inverse_square_root_noising_coefs(
    num_bands: int,
    workload_coef: jax.Array | None = None,
) -> jax.Array:
  r"""Returns Toeplitz noising coefficients for the BISR factorization.

  This computes the first ``num_bands`` coefficients of the lower-triangular
  Toeplitz noising matrix :math:`C^{-1}` for the Banded Inverse Square Root
  (BISR) factorization introduced in https://arxiv.org/pdf/2505.12128. If
  ``workload_coef`` is not provided, this uses the default prefix-sum workload
  with all-ones Toeplitz coefficients. If ``workload_coef`` is provided, then it
  is treated as the Toeplitz coefficients of the workload; this can encode
  workload families such as those induced by SGD with momentum and weight
  decay. In that case, this function computes Toeplitz coefficients of the
  square root of the workload and then returns the first ``num_bands``
  coefficients of its inverse.

  Args:
    num_bands: The number of coefficients to return.
    workload_coef: Optional Toeplitz coefficients of the workload.

  Returns:
    The coefficients of the lower-triangular Toeplitz noising matrix
    :math:`C^{-1}`.
  """
  if workload_coef is None:
    return optimal_max_error_noising_coefs(num_bands)

  workload_coef = pad_coefs_to_n(workload_coef, num_bands)
  sqrt_coefs = jnp.zeros(num_bands, dtype=workload_coef.dtype)
  sqrt_coefs = sqrt_coefs.at[0].set(jnp.sqrt(workload_coef[0]))
  for j in range(1, num_bands):
    inner = jnp.dot(sqrt_coefs[1:j], sqrt_coefs[1:j][::-1])
    sqrt_coefs = sqrt_coefs.at[j].set(
        (workload_coef[j] - inner) / (2 * sqrt_coefs[0])
    )
  return inverse_coef(sqrt_coefs, num_bands)[:num_bands]


def compute_banded_inverse_sensitivity_squared(
    n: int,
    noising_coef: jax.Array,
    min_sep: int,
    max_participations: int | None = None,
    use_matrix_upper_bound: bool = False,
) -> jax.Array:
  r"""Returns squared sensitivity for a banded inverse Toeplitz noising matrix.

  This function takes Toeplitz coefficients of a lower-triangular noising
  matrix :math:`C^{-1}`, computes the implied strategy coefficients for
  :math:`C`, and then estimates the min-separation sensitivity of :math:`C`.

  Tightness depends on the sign and monotonicity of the implied strategy
  coefficients. If the strategy coefficients are positive and
  non-increasing, this uses the closed-form Toeplitz sensitivity computation in
  :func:`~jax_privacy.matrix_factorization.toeplitz.minsep_sensitivity_squared`,
  which is exact.

  Otherwise, the behavior depends on ``use_matrix_upper_bound``:

  - If ``False``, the absolute strategy coefficients are projected onto the
    smallest non-increasing majorant, and the resulting sequence is used to
    compute an upper bound. This bound is exact when the strategy coefficients
    are positive and decreasing, but may be looser when they are not
    non-increasing.

  - If ``True``, the Toeplitz matrix formed from the absolute strategy
    coefficients is materialized, and the generic sensitivity upper bound from
    :mod:`~jax_privacy.matrix_factorization.sensitivity` is used instead. This
    is more computationally expensive, but gives a tighter bound when the
    sequence is not non-increasing, and is exact when the strategy coefficients
    are positive.

  Args:
    n: Size of the implied Toeplitz matrix.
    noising_coef: Toeplitz coefficients of the noising matrix :math:`C^{-1}`.
    min_sep: Minimum separation between participations.
    max_participations: Optional cap on the number of participations.
    use_matrix_upper_bound: Whether to use the generic matrix-based upper bound
      instead of the projected-coefficient upper bound when the absolute
      strategy coefficients are not non-increasing.

  Returns:
    The squared b-min-separated sensitivity of the implied strategy matrix
    :math:`C`.
  """
  strategy_coef = inverse_coef(noising_coef, n)

  if not use_matrix_upper_bound:
    return minsep_sensitivity_squared(
        strategy_coef,
        min_sep=min_sep,
        max_participations=max_participations,
        n=n,
    )

  strategy_matrix = materialize_lower_triangular(jnp.abs(strategy_coef), n)
  return (
      sensitivity.get_min_sep_sensitivity_upper_bound(
          strategy_matrix,
          min_sep=min_sep,
          max_participations=max_participations,
      )
      ** 2
  )


def optimize_banded_inverse_toeplitz(
    n: int,
    min_sep: int,
    num_bands: int,
    *,
    noising_coef: jax.Array | None = None,
    strategy_coef: jax.Array | None = None,
    workload_coef: jax.Array | None = None,
    max_participations: int | None = None,
    max_optimizer_steps: int = 1000,
    reduction_fn: Callable[[jax.Array], jax.Array] = jnp.mean,
) -> jax.Array:
  # pylint: disable=line-too-long
  r"""Optimize over banded inverse Toeplitz noising matrices for BandInvMF.

  This function optimizes directly over the Toeplitz coefficients of the
  lower-triangular noising matrix :math:`C^{-1}` for a Toeplitz workload,
  following the BandInvMF construction introduced in
  https://arxiv.org/pdf/2505.12128. The objective is the reduced per-query
  squared error on the induced workload times the squared ``min_sep``
  sensitivity of the implied strategy matrix :math:`C`.

  Args:
    n: The number of iterations that defines the workload.
    min_sep: The minimum separation between contributions from the same user.
    num_bands: The number of Toeplitz coefficients of the noising matrix to
      optimize, including the diagonal.
    noising_coef: Optional initialization for the noising coefficients. If not
      provided, initializes from ``strategy_coef`` if given, otherwise from
      :func:`~jax_privacy.matrix_factorization.toeplitz.banded_inverse_square_root_noising_coefs`.
      If longer than ``num_bands``, the extra coefficients are ignored.
    strategy_coef: Optional initialization for the strategy coefficients. If
      provided, the corresponding noising coefficients are computed via
      :func:`~jax_privacy.matrix_factorization.toeplitz.inverse_coef`.
    workload_coef: Optional Toeplitz coefficients of the workload. If not
      provided, the default prefix-sum workload of all ones is used.
    max_participations: Optional cap on the number of participations.
    max_optimizer_steps: The maximum number of L-BFGS iterations.
    reduction_fn: A function that converts per query squared errors to a scalar.
      Use ``jnp.mean`` to optimize mean-squared-error, ``jnp.max`` to optimize
      max squared error, or ``lambda v: v[-1]`` to optimize last iterate squared
      error. Defaults to ``jnp.mean``.

  Returns:
    The optimized Toeplitz coefficients of the lower-triangular noising
    matrix :math:`C^{-1}`.
  """
  if workload_coef is None:
    workload_coef = jnp.ones(n)
  else:
    if workload_coef.shape[0] != n:
      raise ValueError(f'{workload_coef.shape[0]=} != {n=}')

  def loss_fn(coef: jax.Array) -> jax.Array:
    error = reduction_fn(
        per_query_error(
            noising_coef=coef,
            n=n,
            workload_coef=workload_coef,
        )
    )
    sens_squared = compute_banded_inverse_sensitivity_squared(
        n=n,
        noising_coef=coef,
        min_sep=min_sep,
        max_participations=max_participations,
        use_matrix_upper_bound=False,
    )

    return error * sens_squared

  if noising_coef is None:
    if strategy_coef is not None:
      noising_coef = inverse_coef(strategy_coef, num_bands)
    else:
      noising_coef = banded_inverse_square_root_noising_coefs(
          num_bands, workload_coef=workload_coef
      )
  noising_coef = pad_coefs_to_n(noising_coef, num_bands)

  params = optimization.optimize(
      loss_fn, noising_coef, max_optimizer_steps=max_optimizer_steps
  )
  return params / params[0]

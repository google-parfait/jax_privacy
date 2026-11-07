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

"""Functions for working with Buffered Linear Toeplitz (BLT) strategy matrices.

:class:`BufferedToeplitz` is the main BLT class, with several helper functions
for error and sensitivity calculation, as well as optimization.
"""

from collections.abc import Callable
import dataclasses
import functools
from typing import Any, TypeAlias

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import optimization
from . import sensitivity
from . import streaming_matrix
from . import toeplitz

# Disabling pylint invalid-name to allow mathematical notation including
# single-capital-letter variables for matrices.
# See README.md for notation conventions.
# pylint:disable=invalid-name


# We want to allow primitive types, lists of primitive types, np.array,
# jnp.array, etc.
# See https://jax.readthedocs.io/en/latest/jep/12049-type-annotations.html


ThetaPairType: TypeAlias = tuple[jax.Array, jax.Array]
ScalarFloat: TypeAlias = jax.typing.ArrayLike


def _gpu_or_cpu_device() -> jax.Device:
  # Use a GPU if possible, and avoid TPUs as we require float64:
  try:
    return jax.local_devices(backend='gpu')[0]
  except RuntimeError:
    return jax.local_devices(backend='cpu')[0]


def check_float64_dtype(blt: 'BufferedToeplitz'):
  if (blt.dtype == jnp.float64) and not jax.config.read('jax_enable_x64'):
    raise ValueError(
        'A BLT with dtype=jnp.float64 requires `jax_enable_x64`.\nMany BLT'
        ' operations require float64 for numerical stability, particularly'
        ' optimization, and buffered_toeplitz.optimize() will return a BLT with'
        ' float64 parameters. If you see this error, you likely need to do one'
        ' of the following:\n'
        '1. Locally, add `with '
        'jax.enable_x64():` around the code that triggered this '
        'error.\n'
        '2. Globally set `jax.config.update("jax_enable_x64", True)`.\n'
        '3. Explicitly cast the parameters of your BufferedToeplitz to '
        ' have dtype=jnp.float32.'
    )


@dataclasses.dataclass(kw_only=True, frozen=True)
class StreamingMatrixBuilder:
  """Builder to convert a BLT to a StreamingMatrix.

  Attributes:
    buf_decay: A numpy array of shape ``(num_buffers,)`` representing the decay
      factors for each buffer.
    output_scale: A numpy array of shape ``(num_buffers,)`` representing the
      scale factors applied to each buffer when reading the output.
  """

  buf_decay: np.ndarray
  output_scale: np.ndarray

  @property
  def dtype(self) -> np.dtype:
    """Returns the dtype of the BLT parameters."""
    assert self.output_scale.dtype == self.buf_decay.dtype
    return self.output_scale.dtype

  def _init(self, abstract_value):
    # Common logic in the StreamingMatrix representation of C and C^{-1}
    # The state is simply the contents of the buffers, each of `shape`.
    assert len(self.buf_decay) == len(self.output_scale)
    num_buffers = self.buf_decay.shape[0]
    # *_like preserves shape, dtype, and (if possible) sharding.
    zero = jnp.zeros_like(abstract_value, dtype=self.dtype)
    return jnp.broadcast_to(zero, (num_buffers,) + zero.shape)

  def _read(self, state):
    # Common logic in the StreamingMatrix representation of C and C^{-1}
    bufs = state
    return jnp.tensordot(
        jnp.asarray(self.output_scale), bufs, axes=((0,), (0,))
    )

  def _update(self, state, next_rhs_value):
    # Common logic in the StreamingMatrix representation of C and C^{-1}
    bufs = state
    buf_decay = jnp.asarray(self.buf_decay)
    # Decay buffers:
    bufs = jnp.tensordot(jnp.diag(buf_decay), bufs, axes=((0,), (0,)))
    # Add xi to each buffer (relies on broadcasting):
    bufs = bufs + next_rhs_value
    return bufs

  def build(
      self,
  ) -> streaming_matrix.StreamingMatrix:
    # pylint: disable=line-too-long
    """Returns a :class:`~jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix` representing C.

    This implements Alg 2 of https://arxiv.org/pdf/2408.08868
    """

    def multiply_next(xi, state):
      yi = xi + self._read(state)
      state = self._update(state, xi)
      return yi, state

    return streaming_matrix.StreamingMatrix.from_array_implementation(
        self._init, multiply_next
    )

  def build_inverse(
      self,
  ) -> streaming_matrix.StreamingMatrix:
    # pylint: disable=line-too-long
    """Returns a :class:`~jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix` representing C^{-1}.

    This implements Alg 3 of https://arxiv.org/pdf/2408.08868
    """

    def inv_multiply_next(yi, state):
      xi = yi - self._read(state)
      state = self._update(state, xi)
      return xi, state

    return streaming_matrix.StreamingMatrix.from_array_implementation(
        self._init, inv_multiply_next
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class BufferedToeplitz:
  # pylint: disable=line-too-long
  """A lower-triangular Toeplitz C parameterized as a BLT.

  :meth:`BufferedToeplitz.build` is the recommended way to construct a BLT.

  For background on Buffered Linear Toeplitz matrices and DP mechanisms, see:
   - `Dvijotham et al. (2024) <https://arxiv.org/abs/2404.16706>`_
   - `McMahan et al. (2024) <https://arxiv.org/abs/2408.08868>`_

  If buf_decay = [d1, d2] and output_scale = [s1, s2], for n = 5
  this class represents
  ::

    1   0  0  0  0             t0 = 1
    t1  1  0  0  0             t1 = s1       + s2
    t2 t1  1  0  0    where    t2 = s1*d1**1 + s2*d2**1
    t3 t2 t1  1  0             t3 = s1*d1**2 + s2*d2**2
    t4 t3 t2 t1  1             t4 = s1*d1**3 + s2*d2**3

  These Toeplitz parameters are returned by ``toeplitz_coefs(n=5)``.
  """

  buf_decay: jax.Array  # Shape(nbuf, )
  output_scale: jax.Array  # Shape (nbuf, )

  def validate(self):
    """Validates basic properties of the BLT parameters."""
    # Note: We do not put these checks in __post_init__ because
    # these properties might not hold if we tree_map over a BLT
    # to hold other values (like gradients or dtypes).
    if not ((self.buf_decay.ndim <= 1) and (self.output_scale.ndim <= 1)):
      raise ValueError(
          'buf_decay and output_scale must be OD or 1D, but :'
          f'buf_decay.shape = {self.buf_decay.shape} and '
          f'output_scale.shape = {self.output_scale.shape}'
      )
    if self.buf_decay.shape != self.output_scale.shape:
      raise ValueError(
          'buf_decay and output_scale must have the same length, but '
          f'buf_decay.shape = {self.buf_decay.shape} !='
          f' output_scale.shape = {self.output_scale.shape}'
      )
    if self.buf_decay.dtype != self.output_scale.dtype:
      raise ValueError(
          'buf_decay and output_scale must have the same dtype, but '
          f'buf_decay.dtype = {self.buf_decay.dtype} !='
          f' output_scale.dtype = {self.output_scale.dtype}'
      )

  @classmethod
  def build(
      cls,
      buf_decay: Any,
      output_scale: Any,
      dtype: jax.typing.DTypeLike = jnp.float64,
  ) -> 'BufferedToeplitz':
    """Helper to convert arguments to jnp.arrays and canonicalize.

    Args:
      buf_decay: The buf_decay parameters of a BLT.
      output_scale: The output_scale parameters of a BLT.
      dtype: The dtype to use for the BLT parameters. The default,
        ``jnp.float64``, is strongly recommended for numerical stability.
        However, this requires either the global option
        ``jax.config.update('jax_enable_x64', True)`` or that build() and
        subsequent computations occur within a ``with jax.enable_x64():``
        context. See also
        :func:`~jax_privacy.matrix_factorization.buffered_toeplitz.check_float64_dtype`.

    Returns:
      A
      :class:`~jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz`
      with buf_decay in decreasing order.
    """
    if (dtype == jnp.float64) and not jax.config.read('jax_enable_x64'):
      raise ValueError(
          'dtype=jnp.float64 requires `jax_enable_x64`; see docstring.'
      )

    blt = cls(
        buf_decay=jnp.array(buf_decay, dtype=dtype),
        output_scale=jnp.array(output_scale, dtype=dtype),
    )
    return blt.canonicalize()  # Also calls validate()

  @classmethod
  def from_rational_approx_to_sqrt_x(
      cls,
      num_buffers: int,
      *,
      max_buf_decay: float = 1.0,
      max_pillutla_score: float | None = None,
      buf_decay_scale: float = 1.6,
      buf_decay_shift: int = -1,
  ) -> 'BufferedToeplitz':
    """Returns a BLT based on a rational approximation of 1/sqrt(1 - x).

    The optimal-for-max-loss Toeplitz coefficients (see
    :func:`~jax_privacy.matrix_factorization.toeplitz.optimal_max_error_strategy_coefs`)
    correspond to the ordinary generating function 1/sqrt(1 - x). Thus, this
    method is used to produce a BLT that approximates these coefficients, but
    allows for a much more memory-efficient implementation of multiplication by
    the noising matrix :math:`C^{-1}`.

    The rational approximation is from `Dvijotham et al. (2024)
    <https://arxiv.org/abs/2404.16706v2>`__,
    see Proposition 4.5 in particular.

    NOTE: The BLTs produced by this method are generally significantly inferior
    to those from
    :func:`~jax_privacy.matrix_factorization.buffered_toeplitz.optimize`, which
    finds a numerically optimal BLT for a specific value of ``n``. Hence, the
    primary use of this method is initializing numerical optimization, as well
    as providing an implementation of the "RA-BLT" method of
    `Dvijotham et al. (2024) <https://arxiv.org/abs/2404.16706v2>`__ for use in
    research comparisons.

    Args:
      num_buffers: The number of buffers to use in the BLT (equivalently, the
        degree of the rational function), must be >= 1.
      max_buf_decay: The maximum value of buf_decay to use. For large numbers of
        buffers, this routine can produce buf_decay values of 1.0 (up to float64
        precision),or higher due to numerical issues. This parameter can be used
        to enforce that the largest buf_decay paramter is strictly less than
        one. This is useful for initializing optimization.
      max_pillutla_score: If not None, the maximum pillutla score to use. This
        is accomplished by first scaling the buf_decay parameters if needed
        based on max_buf_decay, and then scaling the output_scale parameters to
        ensure the pillutla score is less than or equal to this value.
      buf_decay_scale: A factor that scales the dynamic range of the buf_decay
        parameters. Larger values indicate a coarser resolution, and hence the
        largest buf_decay will be closer to 1.0, and the smallest closer to 0.0.
      buf_decay_shift: A shift added to the range of the counter ``k`` in the
        construction of the rational approximation. The buf decay parameters
        come from a discrete set indexed by ``k``, with resolution that depends
        on the ``buf_decay_scale``. A negative ``buf_decay_shift`` shifts the
        selected buf_decay parameters toward 1.0; a positive shift moves the
        selected set closer to 0.0. The default of -1 is recommended.

    Returns:
      A :class:`BufferedToeplitz` matrix generated by a rational function
      approximation of 1/sqrt(1 - x).
    """
    # Note: Following Prop 4.5, we actually construct a BLT for
    # C^{-1} based on the rational function approximation of sqrt(1 - x),
    # and then take the inverse.
    if num_buffers < 1:
      raise ValueError('num_buffers must be >= 1.')

    degree = num_buffers
    d1 = (degree + 1) // 2
    h = buf_decay_scale * np.pi / np.sqrt(2 * (d1 + 1))  # step size.

    # Construct the approximation.
    # The range of the values of the counter `k`
    ks = np.arange(-d1 + 1, degree - d1 + 1, 1) + buf_decay_shift

    buf_decay = 1 / (1 + np.exp(2 * h * ks))  # Aka `c`s

    output_scale = -(buf_decay**2) * np.exp(3 * h * ks)
    constant_term = np.sum(np.exp(h * ks) * buf_decay)

    # Normalize the leading constant to 1.
    output_scale /= constant_term

    # The (cs, scales) define a BLT object for C^{-1}
    inv_blt = cls.build(buf_decay=buf_decay, output_scale=output_scale)
    blt = inv_blt.inverse()
    buf_decay = np.array(blt.buf_decay)
    output_scale = np.array(blt.output_scale)

    # We now scale / project to ensure we are within the
    # max_buf_decay and max_pillutla_socre. First,
    # we scale all the buf decay values, which ensures the ordering
    # isn't change and we don't introduce duplicate values.
    largest_buf_decay = buf_decay[0]
    scale = np.minimum(1.0, max_buf_decay / largest_buf_decay)
    buf_decay *= scale
    assert buf_decay[0] <= max_buf_decay + 1e-12
    score = np.sum(output_scale / buf_decay)
    # If needed, rescale the output parameters to ensure the pillutla score
    # is less than the maximum.
    if max_pillutla_score is not None:
      score_scale = np.minimum(1.0, max_pillutla_score / score)
      output_scale *= score_scale

    blt = cls.build(
        buf_decay=buf_decay,
        output_scale=output_scale,
    )
    if max_pillutla_score is not None:
      assert blt.pillutla_score() <= max_pillutla_score + 1e-12
    return blt

  def canonicalize(self) -> 'BufferedToeplitz':
    """Returns a :class:`BufferedToeplitz` with buf_decay decreasing."""
    self.validate()  # Make sure lengths match.
    idx = jnp.argsort(self.buf_decay)[::-1]
    return BufferedToeplitz(
        buf_decay=self.buf_decay[idx], output_scale=self.output_scale[idx]
    )

  @property
  def dtype(self) -> jax.typing.DTypeLike:
    return self.buf_decay.dtype

  def toeplitz_coefs(self, n: int) -> jax.Array:
    """Returns the Toeplitz coefficients for ``C``."""
    powers = jnp.arange(n - 1)
    tmp = self.buf_decay ** powers[:, None] * self.output_scale
    return jnp.append(1, jnp.sum(tmp, axis=1))

  def materialize(self, n: int) -> jax.Array:
    return toeplitz.materialize_lower_triangular(self.toeplitz_coefs(n))

  def inverse(self, skip_checks: bool = False) -> 'BufferedToeplitz':
    """Compute the BufferedToeplitz parameterization of C^{-1} from C.

    This is an alternative approach to https://arxiv.org/pdf/2404.16706 Lemma
    5.2, along the lines of Proposition 5.6 (Representation of the reciprocal of
    a rational generating function), with slightly different parameterization.


    Args:
      skip_checks: If True, skip error checks on the inputs and results
        (necessary in jitted contexts).

    Returns:
      A Buffered Linear Toeplitz parameterization of the inverse.

    Raises:
      RuntimeError: If skip_checks=False and the inverse calculation encounters
        numerical problems.
    """
    check_float64_dtype(self)
    blt = self  # We only use the public API of the BLT.
    if not skip_checks and len(blt.buf_decay) > 1:
      gap = min_buf_decay_gap(blt.buf_decay)
      if gap < 1e-9:
        # Some quick empirical tests show the closed-form evecs computation
        # returns nans for gaps < 1e-9.
        raise ValueError(
            'The input BLT has buf_decay values that are too close to each'
            f' other.Expected gap >= 1e-9, found {gap} for'
            f' buf_decay={blt.buf_decay}.'
        )
    nbuf = len(blt.buf_decay)
    Theta = jnp.diag(blt.buf_decay)
    omega = blt.output_scale
    alpha = jnp.ones(nbuf)

    # These parameters produce the inverse, but Theta2 is not diagonal,
    # so this is not a BLT:
    Theta2 = Theta - jnp.outer(omega, alpha)
    omega2 = -omega

    # Derive an equivalent BLT by diagonalizing Theta2.
    evals = jnp.linalg.eigvals(Theta2)

    # Ideally we would have a proof that the evals are in fact all real:
    evals = evals.real

    # We have the following closed-form for the eigenvectors:
    evecs = omega[:, jnp.newaxis] / (evals - blt.buf_decay[:, jnp.newaxis])
    einv = jnp.linalg.inv(evecs)

    if not skip_checks:
      Theta2_diagonalized = evecs @ jnp.diag(evals) @ einv
      if not np.allclose(Theta2_diagonalized, Theta2, atol=1e-7):
        raise RuntimeError(
            'Error computing inverse parameters:'
            f' {blt=}\n{evecs=}\n{einv=}\n{evals=}\n'
            f'{Theta2=}\n{Theta2_diagonalized=}'
        )

    omega3 = (einv @ omega2) * (evecs.T @ alpha)
    return BufferedToeplitz.build(
        buf_decay=evals, output_scale=omega3, dtype=blt.dtype
    )

  def pillutla_score(self) -> ScalarFloat:
    """Returns the 'Pillutla Score' of the BLT.

    See Theorem 1 of `An Inversion Theorem for Buffered Linear Toeplitz (BLT)
    Matrices and Applications to Streaming Differential Privacy
    <https://arxiv.org/abs/2504.21413>`_. To avoid a negative buf_decay value
    in the noising matric BLT (which produces an oscillating term), we enforce
    a pillutla_score < 1 during optimization.

    Note that a BLT may have ``buf_decay == 0`` values, which leads to an
    nan or inf pillutla score. (In particular, the inverse of a BLT with
    pillutla_score=0 will have this property).

    Returns:
      The Pillutla Score of the BLT, ``sum_i(output_scale[i] / buf_decay[i])``.
    """
    return jnp.sum(self.output_scale / self.buf_decay)

  @property
  def _num_buffers(self):
    return self.buf_decay.shape[0]

  def _streaming_matrix_builder(self):
    dtype = np.dtype(self.dtype)
    return StreamingMatrixBuilder(
        output_scale=np.array(self.output_scale, dtype=dtype),
        buf_decay=np.array(self.buf_decay, dtype=dtype),
    )

  def as_streaming_matrix(
      self,
  ) -> streaming_matrix.StreamingMatrix:
    # pylint: disable=line-too-long
    """Returns a :class:`~jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix` representing C."""
    check_float64_dtype(self)
    return self._streaming_matrix_builder().build()

  def inverse_as_streaming_matrix(
      self,
  ) -> streaming_matrix.StreamingMatrix:
    # pylint: disable=line-too-long
    """Returns a :class:`~jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix` representing C^{-1}."""
    check_float64_dtype(self)
    return self._streaming_matrix_builder().build_inverse()

  def __str__(self):
    coefs = self.toeplitz_coefs(10)
    coefs_str = ', '.join([f'{c:.5f}' for c in coefs] + ['...'])
    return (
        # Print in a copy-pastable format, with initial coefficients:
        'BufferedToeplitz.build(\n'
        f'    buf_decay={self.buf_decay.tolist()},\n'
        f'    output_scale={self.output_scale.tolist()},\n'
        f'    dtype=jnp.{self.dtype})\n'
        f'Initial Toeplitz coefs=[{coefs_str}])'
    )


def _gt_zero_penalty(x: jax.Array) -> jax.Array:
  """Penalize values to enforce x > 0."""
  return -jnp.log(x).sum()


def _lt_zero_penalty(x: jax.Array) -> jax.Array:
  """Penalize values to enforce x < 0."""
  return -jnp.log(-x).sum()


def _lt_penalty(x: jax.Array, upper_bound: float) -> jax.Array:
  """Penalize values to enforce x < upper_bound."""
  return -jnp.log(upper_bound - x).sum()


def _lt_one_penalty(x: jax.Array) -> jax.Array:
  """Penalize values to enforce x < 1."""
  return -jnp.log(1 - x).sum()


def min_buf_decay_gap(buf_decay: jax.Array) -> jax.Array:
  r"""Returns :math:`\min_{i, j; i \neq j} |\theta_i - \theta_j|`.

  Much of the theory for BLTs, as well as the numerical functions in this file,
  require uniqueness of the buf_decay parameters theta. This function computes
  the smallest gap between two such thetas, without assuming they are sorted.

  Args:
    buf_decay: The buf_decay parameters of a BLT.

  Returns:
    The minimum gap :math:`\min_{i, j; i \neq j} |\theta_i - \theta_j|`.
  """
  theta = jnp.asarray(buf_decay)
  A = theta[:, jnp.newaxis] - theta
  diag_elements = jnp.diag_indices_from(A)
  A = A.at[diag_elements].set(jnp.inf)
  return jnp.min(jnp.abs(A))


@dataclasses.dataclass(frozen=True)
class LossFn:
  # pylint: disable=line-too-long
  """Encapsulates the loss to be optimized for a specific setting.

  This can represent the loss for both single participation and min-sep
  participation (which has single participation as a special case).

  Attributes:
    error_for_inv: Function for computing the error for the BLT representing
      C^{-1}, the noise correlating matrix.
    sensitivity_squared: Function for computing the sensitivity for the BLT
      representing C, the strategy matrix.
    n: The number of iterations the mechanism is optimized for.
    min_sep: The minimum separation of participations.
    max_participations: The effective maximum number of participations, taking
      into account n, min_sep, and max_participations.
    penalty_strength: The multiplier applied to the sum of penalties for the
      loss.
    penalty_multipliers: A dict of multipliers (default 1.0) applied to the
      individual penalties returned by
      :meth:`~jax_privacy.matrix_factorization.buffered_toeplitz.ObjectiveState.compute_penalties`.
    max_second_coef: The maximum value of the second Toeplitz coefficient, which
      is equal to sum(output_scale).
    min_theta_gap: The minimum gap between buf_decay parameters allowed by the
      theta_gap penalty.
  """

  error_for_inv: Callable[[BufferedToeplitz], ScalarFloat]
  sensitivity_squared: Callable[[BufferedToeplitz], ScalarFloat]

  n: int
  min_sep: int
  max_participations: int

  # Usually doesn't need to be changed:
  penalty_strength: float = 1e-8
  penalty_multipliers: dict[str, float] = dataclasses.field(
      default_factory=dict
  )
  max_second_coef: float = 1.0
  min_theta_gap: float = 1e-12

  @classmethod
  def build_closed_form_single_participation(cls, n: int, **kwargs) -> 'LossFn':
    """Construct a :class:`LossFn` for single participation max-error.

    This function utilizes the closed-form calculations for sensitivity and
    error from `Dvijotham et al. (2024) <https://arxiv.org/abs/2404.16706>`__,
    and hence optimization time is essentially independent of ``n``. However,
    particularly for large ``n`` or large numbers of buffers, the optimal BLT
    may have a buf_decay theta very near 1, which leads to numerical issues in
    the closed forms. For max error,this function has been reasonably well
    tested up to n=10**7.  Closed-form optimization of the mean loss closed
    form is possible, but this has not beenwell tested.

    Args:
      n: The number of iterations the mechanism is optimized for.
      **kwargs: Optional additional arguments to pass to the constructor.

    Returns:
      A :class:`LossFn` for single participation.

    Raises:
      ValueError: If ``error`` is not 'max' or 'mean'.
    """
    return cls(
        error_for_inv=functools.partial(max_error, n=n),
        sensitivity_squared=functools.partial(sensitivity_squared, n=n),
        n=n,
        min_sep=1,
        max_participations=1,
        **kwargs,
    )

  @classmethod
  def build_min_sep(
      cls,
      n: int,
      error: str = 'max',
      min_sep: int = 1,
      max_participations: int | None = None,
      **kwargs,
  ) -> 'LossFn':
    """Construct a :class:`LossFn` for min-sep participation.

    This LossFn computes loss and sensitivity by materializing the Toeplitz
    coefficients of C and C^{-1}, and then using the loss functions of
    :mod:`~jax_privacy.matrix_factorization.toeplitz`, as described in
    `McMahan et al. (2024) <https://arxiv.org/abs/2408.08868>`__. This is still
    significantly faster than computing the error directly from the Toeplitz
    coefficients of C, because
    ::

      c_inv_coef = blt.inverse().toeplitz_coefs(n)

    is orders of magnitude faster (on GPUs) than

    ::

      c_inv_coef = toeplitz.inverse_coef(blt.toeplitz_coefs(n))

    Args:
      n: The number of iterations the mechanism is optimized for.
      error: Either 'max' or 'mean', indicating whether to optimize for the
        maximum or mean squared error, respectively.
      min_sep: The minimum separation of participations.
      max_participations: The maximum number of participations.
      **kwargs: Optional additional arguments to pass to the constructor.

    Returns:
      A :class:`LossFn` for min-sep participation.

    Raises:
      ValueError: If ``error`` is not 'max' or 'mean'.
    """

    if error == 'mean':
      error_fn = lambda inv_blt: jnp.mean(
          toeplitz.per_query_error(noising_coef=inv_blt.toeplitz_coefs(n))
      )
    elif error == 'max':
      error_fn = lambda inv_blt: jnp.max(
          toeplitz.per_query_error(noising_coef=inv_blt.toeplitz_coefs(n))
      )

    else:
      raise ValueError(f'Unknown error={error}')

    def minsep_sensitivity_squared(blt):
      return toeplitz.minsep_sensitivity_squared(
          strategy_coef=blt.toeplitz_coefs(n),
          min_sep=min_sep,
          n=n,
          max_participations=max_participations,
      )

    return cls(
        n=n,
        error_for_inv=error_fn,
        sensitivity_squared=minsep_sensitivity_squared,
        min_sep=min_sep,
        max_participations=sensitivity.minsep_true_max_participations(
            n=n,
            min_sep=min_sep,
            max_participations=max_participations,
        ),
        **kwargs,
    )

  def compute_penalties(
      self, blt: BufferedToeplitz, inv_blt: BufferedToeplitz
  ) -> dict[str, ScalarFloat]:
    """Computes penalties that help keep the optimization well-behaved.

    These correspond to the conditions of Theorem 1 (part a) of "An Inversion
    Theorem for Buffered Linear Toeplitz (BLT) Matrices and Applications to
    Streaming Differential Privacy" (https://arxiv.org/abs/2504.21413), which
    restricts the optimization to a class of well-behaved BLTs. Note the
    constraint ``pillutla_score < 1`` of part (a) is not strictly necessary, but
    empirically including it produces better results.

    Args:
      blt: The BLT representing C.
      inv_blt: The BLT representing C^{-1}.

    Returns:
      A dictionary of named penalties.
    """
    check_float64_dtype(blt)
    num_buffers = len(blt.buf_decay)
    # We could also impose the same penalties on inv_blt, but this does not seem
    # to be necessary.
    penalties = {
        # Conditions on `blt` parameters.
        'buf_decay>0': _gt_zero_penalty(blt.buf_decay),
        'buf_decay<1': _lt_one_penalty(blt.buf_decay),
        'output_scale>0': _gt_zero_penalty(blt.output_scale),
        # Conditions on `inv_blt` parameters.
        'inv_buf_decay>0': _gt_zero_penalty(inv_blt.buf_decay),
        'inv_buf_decay<1': _lt_one_penalty(inv_blt.buf_decay),
        'inv_output_scale<0': _lt_zero_penalty(inv_blt.output_scale),
    }

    if num_buffers > 1:
      # Optimizing for multiple participations can cause multiple buf_decay
      # parameters to converge to the same value, which will break
      # `blt.inverse()` numerically. We know "good" solutions should
      # have separated thetas, so we enforce some separation:
      penalties['theta_gap'] = _gt_zero_penalty(
          min_buf_decay_gap(blt.buf_decay) - self.min_theta_gap
      ) + _gt_zero_penalty(
          min_buf_decay_gap(inv_blt.buf_decay) - self.min_theta_gap
      )

    # The 2nd Toeplitz coefficient (after 1.0) is equal to sum(output_scale),
    # so to ensure decreasing coefficients it is also necessary that
    # sum(output_scale) < 1.  When optimizing for single participation,
    # we know the optimal 2nd coefficient is 0.5, so we could also use
    # a value like max_second_coef = 0.5 + (small value).
    second_coef = blt.output_scale.sum()
    penalties['second_coef'] = _lt_penalty(second_coef, self.max_second_coef)

    # The final condition is the 'Pillutla Score' is less than 1.
    penalties['pillutla_score'] = _lt_one_penalty(blt.pillutla_score())

    return penalties

  def penalized_loss(
      self,
      blt: BufferedToeplitz,
      inv_blt: BufferedToeplitz,
      normalize_by_approx_optimal_loss: bool = True,
  ) -> ScalarFloat:
    """Computes the total composite loss to be optimized.

    Args:
      blt: The BLT representing C.
      inv_blt: The BLT representing C^{-1}.
      normalize_by_approx_optimal_loss: If True, the loss is normalized by the
        expected optimal loss, so the relative penalty strength remains somewhat
        consistent across n and k. This is the default, and recommended for
        optimization.

    Returns:
      The total composite loss to be optimized.
    """
    check_float64_dtype(blt)
    error = self.error_for_inv(inv_blt)
    sens_squared = self.sensitivity_squared(blt)
    penalty_dict = self.compute_penalties(blt, inv_blt)

    multipliers = dict(self.penalty_multipliers)  # copy
    total_penalty = 0.0
    for k, v in penalty_dict.items():
      multiplier = multipliers.pop(k, 1.0)
      if multiplier != 0.0:
        # We want a multiplier of 0 to "turn off" the penalty even if
        # the penalty is NaN or inf, so we avoid mulitplication in that case.
        total_penalty += multiplier * v
    total_penalty *= self.penalty_strength
    if multipliers:
      raise ValueError(
          f'Unrecognized penalty multipliers: {multipliers.keys()}'
      )

    loss = error * sens_squared
    if normalize_by_approx_optimal_loss:
      # Optimal sqrt(max_loss) for Toeplitz matrices scales like
      # 1 + ln(n)/pi; sqrt(mean_loss) can be a bit lower.
      # Similarly, sensitivity_squared scales by max_participations.
      # This gives a rough estimate of the expected optimal loss,
      # which we use to normalize the computed loss so the
      # relative penalty strength remains somewhat consistent across n and k.
      approx_optimal_loss = (
          self.max_participations + (1 + jnp.log(self.n) / np.pi) ** 2
      )
      loss /= approx_optimal_loss

    return loss + total_penalty

  def loss(
      self, blt: BufferedToeplitz, skip_checks: bool = False
  ) -> ScalarFloat:
    """Returns the loss (not including penalties).

    This function is not intended to be jitted or used in optimization, but
    only in evaluation of the final BLT.

    Args:
      blt: The BLT to compute the loss of.
      skip_checks: If True, do not check that the BLT is valid for min-sep
        sensitivity.
    """
    check_float64_dtype(blt)
    try:
      inv_blt = blt.inverse(skip_checks=skip_checks)
    except RuntimeError as e:
      logging.warning(
          'During loss computation, error computing inverse for'
          ' BLT\n%s\nReturning jnp.inf. If you really need a finite loss for'
          ' this BLT and n is small, consider computing the loss directly from'
          ' blt.toeplitz_coefs(n). Exception:\n%s',
          str(blt),
          str(e),
      )
      return jnp.inf
    error = self.error_for_inv(inv_blt)
    sens_squared = self.sensitivity_squared(blt)
    return error * sens_squared


@dataclasses.dataclass(frozen=True)
class Parameterization:
  """A parameterization of a BufferedToeplitz for optimization.

  Used by
  :func:`~jax_privacy.matrix_factorization.buffered_toeplitz.optimize_loss` to
  specify how parameters relate to the pair of BLTs representing C and C^{-1}.

  Attributes:
    params_from_blt: Constructs parameters to be optimized initialized from a
      BLT.
    blt_and_inverse_from_params: Constructs a tuple of BLTs representing the
      (strategy_matrix, noising_matrix) from parameters.
    loss_fn: The loss function to optimize.
  """

  params_from_blt: Callable[[BufferedToeplitz], optax.ArrayTree]
  blt_and_inverse_from_params: Callable[
      [Any],
      tuple[BufferedToeplitz, BufferedToeplitz],
  ]

  @classmethod
  def strategy_blt(cls) -> 'Parameterization':
    """A parameterization where the strategy BLT is the parameterization."""
    return cls(
        params_from_blt=lambda blt: blt,
        blt_and_inverse_from_params=lambda blt: (
            blt,
            blt.inverse(skip_checks=True),
        ),
    )

  @classmethod
  def buf_decay_pair(cls) -> 'Parameterization':
    """A parameterization where a pair of buf_decay parameters is optimized.

    This parameterization is generally more numerically stable than the
    ``strategy_blt`` parameterization, as well as being negligibly faster to
    compute (as it does not require a singular-value decomposition). However,
    the current L-BFGS parameters are tuned for the strategy_blt
    parameterization, so this parameterization may not converge as well with the
    default settings.

    Returns:
        A :class:`Parameterization`.
    """

    def params_from_blt(blt: BufferedToeplitz) -> tuple[jax.Array, jax.Array]:
      inv_blt = blt.inverse()
      return (blt.buf_decay, inv_blt.buf_decay)

    def blt_and_inverse_from_params(
        params: tuple[jax.Array, jax.Array],
    ) -> tuple[BufferedToeplitz, BufferedToeplitz]:
      return blt_pair_from_theta_pair(params[0], params[1])

    return cls(
        params_from_blt=params_from_blt,
        blt_and_inverse_from_params=blt_and_inverse_from_params,
    )

  def get_loss_fn(
      self, loss_fn: LossFn
  ) -> Callable[[optax.ArrayTree], ScalarFloat]:
    """Returns a loss function for the parameterization."""
    return lambda params: loss_fn.penalized_loss(
        *self.blt_and_inverse_from_params(params)
    )


def get_init_blt(
    num_buffers: int = 3,
    init_blt: BufferedToeplitz | None = None,
) -> BufferedToeplitz:
  """Returns an initial BufferedToeplitz for initializing optimization.

  Currently, this defaults to
  :meth:`BufferedToeplitz.from_rational_approx_to_sqrt_x`, setting
  max_buf_decay and max_pillutla_score so that the solution is within the
  feasible set imposed by the optimization penalities in
  :meth:`LossFn.compute_penalties`. However, this initialization choice may be
  changed in the future.

  Args:
    num_buffers: The number of buffers to use in the initial BLT, greater than
      or equal to zero.
    init_blt: An initial BLT to use. If ``None``, a default initialization is
      used. This is a convenience for callers who want to handle an optional
      explicit initialization and check that it has the correct number of
      buffers.

  Returns:
    A BufferedToeplitz with the requested number of buffers.
  """
  if not init_blt:
    if num_buffers == 0:
      init_blt = BufferedToeplitz.build(buf_decay=[], output_scale=[])
    else:
      init_blt = BufferedToeplitz.from_rational_approx_to_sqrt_x(
          num_buffers=num_buffers,
          max_buf_decay=1 - 1e-6,
          max_pillutla_score=1 - 1e-6,
      )

  # Check we have the correct number of buffers:
  if len(init_blt.buf_decay) != num_buffers:
    raise ValueError(
        f'{num_buffers=} does not match {len(init_blt.buf_decay)=}'
    )
  return init_blt


@optimization.jax_enable_x64
def optimize_loss(
    loss_fn: LossFn,
    num_buffers: int = 1,
    init_blt: BufferedToeplitz | None = None,
    parameterization: Parameterization | None = None,
    **kwargs,
) -> tuple[BufferedToeplitz, ScalarFloat]:
  """Like :func:`optimize`, but more configurable and a fixed number of buffers.

  Args:
    loss_fn: The loss to optimize.
    num_buffers: The number of buffers to optimize for; the default of 3 is a
      good choice in general; large number of buffers can cause numerical
      instability in the optimization.
    init_blt: An initial BLT to start the optimization from. If ``None``, a
      default initialization is used.
    parameterization: The parameterization to use for optimization. If ``None``,
      the default parameterization is used.
    **kwargs: Optional additional arguments to pass to
      :func:`~jax_privacy.matrix_factorization.optimization.optimize`, such as
      ``max_optimizer_steps``, ``callback``, and ``optimizer``. Note this
      function may supply different defaults for these arguments compared to
      :func:`~jax_privacy.matrix_factorization.optimization.optimize`, so use
      with care.

  Returns:
     A tuple (blt, loss).

  Raises:
    RuntimeError: If the optimization produces an invalid BLT.
  """
  if num_buffers == 0:
    # Construct the identity strategy matrix and directly compute the loss.
    blt = BufferedToeplitz.build(buf_decay=[], output_scale=[])
    return blt, loss_fn.loss(blt)

  if parameterization is None:
    parameterization = Parameterization.buf_decay_pair()

  default_optimizer = optax.lbfgs(
      # Empirical testing shows these parameters improve over the defaults.
      # A larger memory size can speed convergence, but also seems to result
      # in the optimizer getting stuck at a suboptimal solution.
      memory_size=15,
      linesearch=optax.scale_by_zoom_linesearch(
          max_linesearch_steps=100,
          curv_rtol=0.5,
          # This avoids some rare nan cases:
          max_learning_rate=1.0,
          # A small stepsize_precision is necessary when we need values of
          # theta very near 1.
          stepsize_precision=1e-20,
      ),
  )
  optimize_kwargs = {'max_optimizer_steps': 600, 'optimizer': default_optimizer}
  optimize_kwargs.update(kwargs)
  blt = get_init_blt(num_buffers=num_buffers, init_blt=init_blt)

  params = parameterization.params_from_blt(blt)
  # Combine the parameterization with the loss fn:
  loss_fn_to_optimize = parameterization.get_loss_fn(loss_fn)
  params = optimization.optimize(
      # Create
      loss_fn_to_optimize,
      params,
      **optimize_kwargs,
  )
  blt, _ = parameterization.blt_and_inverse_from_params(params)
  blt = blt.canonicalize()

  loss = loss_fn.loss(blt)
  if not jnp.isfinite(loss):
    raise RuntimeError(
        f'Optimization produced BLT with non-finite loss {loss}:\n{blt}'
    )

  if jnp.any(jnp.abs(blt.output_scale) < 1e-8):
    logging.warning(
        'BLT has near-zero output_scale parameters, which '
        'means some buffers are ignored. Consider re-optimizing '
        'with a smaller number of buffers.\n%s',
        blt,
    )
  return blt, loss


def _optimize_increasing_nbuf(
    opt_blt_and_loss_fn: Callable[[int], tuple[Any, float]],
    min_buffers: int = 0,
    max_buffers: int = 10,
    rtol: float = 1.02,
) -> Any:
  """Optimizes w/ increasing num_buffers until the improvement is < rtol."""
  prev_blt, prev_loss = opt_blt_and_loss_fn(min_buffers)
  for nbuf in range(min_buffers + 1, max_buffers + 1):
    try:
      blt, loss = opt_blt_and_loss_fn(nbuf)
    except NotImplementedError as err:
      err.add_note(
          'This error may be caused by trying to run `blt.inverse()` on a GPU.'
      )
      # This can happen if we try to run `jnp.linalg.eigvals` on a GPU,
      # for example. This is a real configuration issue, not
      # an optimization failure.
      raise err
    except RuntimeError as err:
      logging.warning(
          'Optimization failed for %d buffers with:\n%s',
          nbuf,
          str(err),
      )
      blt, loss = None, float('inf')

    if rtol * loss < prev_loss:
      # Sufficient improvement, accept this BLT and maybe try more buffers:
      prev_blt, prev_loss = blt, loss
    else:
      # Improvement was < rtol, return prev_blt
      return prev_blt
  return prev_blt


@optimization.jax_enable_x64
def optimize(
    *,
    n: int,
    min_sep: int = 1,
    max_participations: int | None = 1,
    error: str = 'max',
    min_buffers: int = 0,
    max_buffers: int = 10,
    rtol: float = 1.01,
    **kwargs,
) -> BufferedToeplitz:
  """Computes a good BLT with a dynamically-chosen num_buffers for min-sep.

  Internally this function uses ``jax.jit`` on the key optimization steps, but
  it is not intended to be called from a jitted context itself.

  For single-participation optimization of max error, this function utilizes the
  closed-form calculations for sensitivity and error from
  `Dvijotham et al. (2024) <https://arxiv.org/abs/2404.16706>`__ and hence
  optimization time is essentially
  independent of ``n``. However, particularly for large ``n`` or large numbers
  of buffers, the optimal BLT may have a buf_decay theta very near 1, which
  leads to numerical issues in the closed forms. This function has been
  reasonably well tested for max loss up to n=10**7.

  Otherwise (for multiple-participations or mean loss optimization), this
  function computes loss and sensitivity by materializing the Toeplitz
  coefficients of C and C^{-1}, and then using the loss functions of
  :mod:`~jax_privacy.matrix_factorization.toeplitz`, as described in
  `McMahan et al. (2024) <https://arxiv.org/abs/2408.08868>`__. This is still
  significantly faster than
  directly optimizing Toeplitz mechanisms, because ``blt.inverse()`` is much
  faster than computing the Toeplitz coefficients of C^{-1} directly. This
  optimization benefits from GPUs for large n (say > 1000); We have observed
  some issues using TPUs, so avoid them for now.

  Internally this function uses ``jax.jit`` on the key optimization steps, but
  it is not intended to be called from a jitted context itself.

  Args:
    n: The number of iterations the mechanism is optimized for.
    min_sep: The minimum separation of participations.
    max_participations: The maximum number of participations.
    error: Either 'max' or 'mean', indicating whether to optimize for the
      maximum or mean squared error, respectively.
    min_buffers: The minimum number of buffers to optimize for (inclusive).
    max_buffers: The maximum number of buffers to optimize for (inclusive).
    rtol: The relative tolerance for the loss improvement in order to increase
      the number of buffers.
    **kwargs: Optional additional arguments to pass to
      :func:`~jax_privacy.matrix_factorization.optimization.optimize`, such as
      ``max_optimizer_steps`` and ``optimizer``. Note this function may supply
      different defaults for these arguments compared to
      :func:`~jax_privacy.matrix_factorization.optimization.optimize`, so use
      with care. Further, the same arguments will be passed to each optimization
      (for different ``num_buffers``), so for example a stateful ``callback``
      function may not work as expected.

  Returns:
    A BLT that approximately minimizes the loss.

  Raises:
    RuntimeError: If the optimization produces an invalid BLT.
  """

  if max_buffers > 15:
    raise ValueError(
        'In typical regimes 3 to 7 buffers should give the best utility (and'
        ' memory efficiency); 15+ buffers will almost certainly lead to'
        ' numerical issues in the optimization. Try setting a smaller value for'
        ' `max_buffers`.'
    )

  k = sensitivity.minsep_true_max_participations(
      n=n, min_sep=min_sep, max_participations=max_participations
  )
  if k == 1 and error == 'max':
    # Single participation max loss, used closed forms:
    loss_fn = LossFn.build_closed_form_single_participation(n=n)
  else:
    loss_fn = LossFn.build_min_sep(
        n=n, error=error, min_sep=min_sep, max_participations=max_participations
    )

  with jax.default_device(_gpu_or_cpu_device()):
    return _optimize_increasing_nbuf(
        opt_blt_and_loss_fn=lambda nbuf: (
            optimize_loss(
                loss_fn=loss_fn,
                num_buffers=nbuf,
                parameterization=Parameterization.buf_decay_pair(),
                **kwargs,
            )
        ),
        min_buffers=min_buffers,
        max_buffers=max_buffers,
        rtol=rtol,
    )


def geometric_sum(
    a: jax.Array, r: jax.Array, num: jax.typing.ArrayLike = jnp.inf
) -> jax.Array:
  """Sum a + a*r + a*r**2 + ... + a*r**(num-1) (or limit if num=jnp.inf).

  Args:
    a: Scale factor (or vector of scale factors).
    r: ratio between successive terms. Any value is supported for finite
      ``num``; the limit ``num=jnp.inf`` requires :math:`|r| < 1` for the series
      to converge.
    num: How many terms to add, or jnp.inf for the limit.

  Returns:
    The sum.
  """
  n = jnp.array(num, dtype=jnp.float64)

  def finite_n_geo_sum(a, n, r):
    """Robustly handle the finite-n case, including r=1."""
    # We choose between a Taylor series approx near r=1 and the
    # direct calculation.
    #
    # The direct calculation is numerically fairly stable for the
    # value, but gets unstable earlier (r farther from 1) for the calculation
    # of the gradient (which we need). So this threshold rule was chosen to
    # minimize the difference between the direct computation and the series
    # approximation in terms of the gradient w.r.t. `r`.
    SLOPE = 0.53018965
    INTERCEPT = 3.33503185
    pow_threshold = INTERCEPT + SLOPE * jnp.log(n)

    # The series approximation is only valid in a neighborhood of r = 1, so the
    # test must be two-sided: the direct calculation is exact for every r != 1,
    # including r > 1 (which arises for example as a product theta_j * theta_k
    # of buf_decay parameters of magnitude greater than one).
    use_direct_calc = jnp.abs(r - 1) > 10 ** (-pow_threshold)

    # Quadratic Taylor polynomial approx at r = 1 from sympy:
    x0 = n - 1
    x1 = r - 1
    series_approx = (1 / 6) * a * n * (x0 * x1**2 * (n - 2) + 3 * x0 * x1 + 6)

    # Avoid nans in the untaken branch when r == 1, see
    # https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
    safe_r = jnp.where(use_direct_calc, r, jnp.zeros_like(r))
    return jnp.where(
        use_direct_calc, a * (1 - safe_r**num) / (1 - safe_r), series_approx
    )

  return jax.lax.cond(
      jnp.isinf(n),
      lambda a, n, r: a / (1 - r),  # Infinite n case
      finite_n_geo_sum,
      a,
      num,
      r,
  )


def _inf_if(cond_fn, blt_fn: Callable[..., Any]) -> Any:
  def new_fn(blt, *args, **kwargs):
    return jax.lax.cond(
        cond_fn(blt),
        lambda: jnp.inf,
        lambda: blt_fn(blt, *args, **kwargs),
    )

  return new_fn


def require_buf_decay_less_eq_one(blt_fn: Callable[..., Any]) -> Any:
  """Return inf if blt, the first arg to blt_fn, has buf_decay > 1."""

  return _inf_if(lambda blt: jnp.any(blt.buf_decay > 1), blt_fn)


def require_output_scale_gt_zero(blt_fn: Callable[..., Any]) -> Any:
  """Return inf if blt, the first arg to blt_fn, has output_scale <= 0."""

  return _inf_if(lambda blt: jnp.any(blt.output_scale <= 0), blt_fn)


def _poly_from_theta(theta: jax.Array) -> jax.Array:
  """Returns a polynomial (1 - theta[0]*x)*...*(1-theta[-1]*x).

  Args:
     theta: The multiplicative inverses of the roots of the polynomial.

  Returns:
     A polynomial of degree len(theta) -1  = d represented as an array under the
     ``np.poly1d`` convention. If the returned array is ``c``, then
     p(x) = c[0]*x**d + c[1]*x**(d-1) + ... + c[-2]*x + c[-1]
  """
  theta = jnp.asarray(theta)
  z = jnp.prod(-theta)
  p = jnp.poly(seq_of_zeros=1 / theta) * z
  p = p.at[-1].set(1)  # This should be approximately 1, set to make exact.
  return p


def blt_pair_from_theta_pair(
    theta: jax.Array, theta_hat: jax.Array
) -> tuple[BufferedToeplitz, BufferedToeplitz]:
  """Computes BLTs (C, C_inv) from theta and theta_hat.

  This implements Lemma 5.2 of `Dvijotham et al. (2024)
  <https://arxiv.org/abs/2404.16706>`__ (See also
  Algorithm 5 of `McMahan et al. (2024) <https://arxiv.org/abs/2408.08868>`__).
  The simplified computation
  used in the implementation here appears as Theorem 2 of
  `McMahan and Pillutla (2025) <https://arxiv.org/abs/2504.21413>`_.

  This function uses quantities like 1 /(1/theta[i] - 1/theta[j]), and so can be
  numerically unstable if abs(theta[i] - theta[j]) is too small; hence, by
  default the Parameterization based on ``blt.inverse()`` should be used
  instead.

  Args:
     theta: Array of thetas for the denominator q of ``r(x) = p(x; theta_hat) /
       q(x; theta)``.
     theta_hat: Array of thetas for the numerator of r(x).

  Returns:
    A tuple of BLTs (C, C_inv) where C.buf_decay = theta and C_inv.buf_decay =
    theta_hat,
    with the output_scale (omega) parameters for each computed to make these
    inverses.
  """
  theta = jnp.asarray(theta)
  theta_hat = jnp.asarray(theta_hat)

  def get_omega(theta, theta_hat):
    numerators = jnp.prod(theta[:, jnp.newaxis] - theta_hat, axis=1)
    # Compute denom_i = prod_{j: i != j} (lambda_i - lambda_j)
    A = theta[:, jnp.newaxis] - theta
    A = A.at[jnp.diag_indices_from(A)].set(1)
    denominators = jnp.prod(A, axis=1)
    return numerators / denominators

  return (
      BufferedToeplitz.build(
          output_scale=get_omega(theta, theta_hat), buf_decay=theta
      ),
      # Note we reverse the order of the arguments to `get_omega` to get C_inv.
      BufferedToeplitz.build(
          output_scale=get_omega(theta_hat, theta), buf_decay=theta_hat  # pylint: disable=arguments-out-of-order
      ),
  )


@jax.jit
@require_buf_decay_less_eq_one
def sensitivity_squared(
    blt: BufferedToeplitz, n: jax.typing.ArrayLike
) -> float:
  """Computes sensitivity**2 for a BLT strategy matrix C.

  See https://arxiv.org/pdf/2404.16706 Lemma 5.3

  Args:
    blt: The Buffered Linear Toeplitz operator representing C().
    n: The number of iterations; the limit of sensitivity as n -> jnp.inf is
      returned if n is jnp.inf.

  Returns:
    The max-column-norm-squared of C.
  """
  omega = blt.output_scale
  theta = blt.buf_decay
  num = jnp.array(n - 1)  # Still jnp.inf if n = jnp.inf

  # Vectorized via JAX broadcasting, omega and theta must be the same length.
  omega_pairs = omega * omega[:, jnp.newaxis]
  theta_pairs = theta * theta[:, jnp.newaxis]
  geo_pairs = geometric_sum(omega_pairs, theta_pairs, num=num)
  return 1.0 + geo_pairs.sum()


def _max_error_Gamma_j(
    omega: jax.Array, theta: jax.Array, n: jax.Array
) -> jax.Array:
  # Closed-form computation of
  # sum([geometric_sum(omega, theta, i) for i in range(1, n)])
  return (omega / (1.0 - theta)) * (1 - geometric_sum(1, theta, n) / n)


def _max_error_Gamma_j_series(
    omega: jax.Array, theta: jax.Array, n: jax.Array
) -> jax.Array:
  """Taylor series approximation to _max_error_Gamma_j."""
  # Auto-generated via sympy, see colab notebook
  # robust_max_error_for_blts.ipynb.ipynb
  # fmt: off
  x0 = theta - 1
  x1 = omega*(n - 2)*(n - 1)
  return -omega*(1/2 - 1/2*n) + (1/24)*x0**2*x1*(n - 3) + (1/6)*x0*x1
  # fmt: on


def robust_max_error_Gamma_j(
    omega: jax.Array, theta: jax.Array, n: jax.Array
) -> jax.Array:
  """Robustly computes _max_error_Gamma_j."""
  # See robust_max_error_for_blts.ipynb.ipynb
  # for computation of these regression constants.
  J_SLOPE = 0.43877484
  J_INTERCEPT = 2.91215085

  power = J_INTERCEPT + J_SLOPE * jnp.log(n)
  threshold = 1 - 10 ** (-power)
  predicate = theta < threshold

  # We need to avoid inf/nan in v0, v1, and v2 even if not selected, see:
  # https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
  safe_theta = jnp.where(predicate, theta, jnp.zeros_like(theta))
  v0 = _max_error_Gamma_j(omega, safe_theta, n)
  v1 = _max_error_Gamma_j_series(omega, theta, n)
  return jnp.where(predicate, v0, v1)


def _max_error_Gamma_jk(
    omega1: jax.Array,
    theta1: jax.Array,
    omega2: jax.Array,
    theta2: jax.Array,
    n: jax.Array,
) -> jax.Array:
  """Direct computation of Gamma_jk for max-error."""
  # Closed-form computation of
  # sum([geometric_sum(omega1, theta1, i) * geometric_sum(omega2, theta2, i)
  #      for i in range(1, n)])
  temp1 = omega1 * omega2 / ((1 - theta1) * (1 - theta2))
  temp2 = (
      n
      - geometric_sum(1, theta1, n)
      - geometric_sum(1, theta2, n)
      + geometric_sum(1, theta1 * theta2, n)
  ) / n
  return temp1 * temp2


def _max_error_Gamma_jk_series_j(
    omega1: jax.Array,
    theta1: jax.Array,
    omega2: jax.Array,
    theta2: jax.Array,
    n: jax.Array,
) -> jax.Array:
  """Compute _max_error_Gamma_jk with a series approximation for theta1."""
  # Auto-generated via sympy, see colab notebook
  # robust_max_error_for_blts.ipynb
  # fmt: off
  x0 = theta2 - 1
  x1 = theta2**(n + 1)
  x2 = -x1
  x3 = 6*x0
  x4 = theta2**n
  x5 = n - 1
  x6 = theta1 - 1
  x7 = theta2**(n + 2)
  return (-1/6*omega1*omega2*(
      n*x0**3*(3*n + x5*x6*(n - 2) - 3) + n*x3*(x2 + x4) - x3*(theta2 + x2)
      + 3*x6*(n*x0*(-x1*x5 + x4*x5) - 2*n*(x1 - x7)
              + 2*theta2**2 - 2*x7))/(n*x0**4))
  # fmt: on


def _max_error_Gamma_jk_series_jk(
    omega1: jax.Array,
    theta1: jax.Array,
    omega2: jax.Array,
    theta2: jax.Array,
    n: jax.Array,
) -> jax.Array:
  """Compute _max_error_Gamma_jk with a series approximation for theta1."""
  # Auto-generated via sympy, see colab notebook
  # robust_max_error_for_blts.ipynb
  # fmt: off
  x0 = n**2
  x1 = 3*n**3 + 9*n - 10*x0 - 2
  return ((1/24)*omega1*omega2*(-12*n + 8*x0 + x1*(theta1 - 1)
                                + x1*(theta2 - 1) + 4))
  # fmt: on


def robust_max_error_Gamma_jk(
    omega1: jax.Array,
    theta1: jax.Array,
    omega2: jax.Array,
    theta2: jax.Array,
    n: jax.Array,
) -> jax.Array:
  """Robustly computes _max_error_Gamma_jk.

  Robustly computes the Gamma_{j,k} term of the closed-form expression
  for max-error, by choosing either the direct computation or a Taylor
  series approximation.

  Used below to compute all pairs of Gamma_{j,k} via numpy broadcasting.


  Args:
    omega1:  The omega_j argument (a output_scale value), typically a 1D array.
    theta1:  The theta_j argument (a buf_decay value), typically a 1D array.
    omega2:  The omega_k argument (a output_scale value), typically a 2D column
      vector.
    theta2:  The theta_k argument (a buf_decay value), typically a 2D column
      vector.
    n: The value of n.

  Returns:
    The value of Gamma_jk.
  """
  # The _j series approximation needs theta1 > theta2:
  theta1, theta2 = jnp.maximum(theta1, theta2), jnp.minimum(theta1, theta2)

  JK_SLOPE = 0.35321577
  JK_INTERCEPT = 2.81518052
  power = JK_INTERCEPT + JK_SLOPE * jnp.log(n)
  threshold = 1 - 10 ** (-power)

  # We need to avoid inf/nan in v0, v1, and v2 even if not selected, see:
  # https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
  v0_predicate = theta1 < threshold
  v1_predicate = theta2 < threshold

  safe_theta1 = jnp.where(v0_predicate, theta1, jnp.zeros_like(theta1))
  safe_theta2 = jnp.where(v0_predicate, theta2, jnp.zeros_like(theta2))
  v0 = _max_error_Gamma_jk(omega1, safe_theta1, omega2, safe_theta2, n)

  safe_theta2 = jnp.where(v1_predicate, theta2, jnp.zeros_like(theta2))
  v1 = _max_error_Gamma_jk_series_j(omega1, theta1, omega2, safe_theta2, n)
  v2 = _max_error_Gamma_jk_series_jk(omega1, theta1, omega2, theta2, n)

  return jnp.where(
      v0_predicate,  # and theta2 <= theta1 by assumption
      # Both thetas are not close to 1, use the direct fn:
      v0,
      jnp.where(
          v1_predicate,
          # theta1 is near 1, theta2 is not
          v1,
          # Both thetas are near 1, use 2-variable series
          v2,
      ),
  )


@jax.jit
def iteration_error(
    inv_blt: BufferedToeplitz, i: jax.typing.ArrayLike
) -> jax.Array:
  """Computes the error on iteration ``i`` which is also the max error.

  That is, for a Buffered Linear Toeplitz matrix, the max error from iteration 0
  through ``i`` is achieved on iteration ``i``, so this equivalently computes
  the max error for ``i+1`` iterations.

  Here "error" is the squared error introduced in the ``n``-th iterate (partial
  sum) assuming unit-variance noise. This generally scales as O(n), and
  so optimization routines might normalize by an additional factor of n.

  This implements https://arxiv.org/pdf/2404.16706 Lemma 5.4.

  Args:
    inv_blt: The Buffered Linear Toeplitz operator where inv_blt.C() represents
      C^{-1} in the matrix factorization mechanism.
    i: The iteration for which to compute error, 0-indexed. To compute the max
      error for an ``n`` iteration mechanism, one should thus pass i = n-1 to
      this function.

  Returns:
    The squared-error (variance) on iteration ``iter``.
  """
  check_float64_dtype(inv_blt)
  # Note: The derivation in the paper is 1-indexed and uses `n`;
  # we follow that formula here, so we define n accordingly:
  n = i + 1

  omega = inv_blt.output_scale  # shape = (inv_blt._num_buffers,)
  theta = inv_blt.buf_decay  # shape = (inv_blt._num_buffers,)
  # (b, b) -> scalar, where b = inv_blt._num_buffers
  s1 = jnp.sum(
      robust_max_error_Gamma_j(omega, theta, n)
  )  # Vectorized, via JAX broadcasting
  s2 = jnp.sum(
      robust_max_error_Gamma_jk(
          omega, theta, omega[:, jnp.newaxis], theta[:, jnp.newaxis], n
      )
  )  # (b, b) -> scalar
  return n * (1 + 2 * s1 + s2)


@jax.jit
def max_error(inv_blt: BufferedToeplitz, n: jax.typing.ArrayLike) -> jax.Array:
  """Returns the max squared error for any iteration 0, ..., n-1."""
  # Note: For a BLT, the iteration error is increasing in `i`, so:
  return iteration_error(inv_blt, n - 1)


@jax.jit
def limit_max_error(
    inv_blt: BufferedToeplitz,
) -> jax.Array:
  """Computes limit_{n -> jnp.inf} (1/n)*max_error(blt, n).

  Args:
    inv_blt: The Buffered Linear Toeplitz operator where inv_blt.C() represents
      C^{-1} in the matrix factorization mechanism.

  Returns:
    The iteration error in the limit.
  """
  omega = inv_blt.output_scale  # shape = (inv_blt._num_buffers,)
  theta = inv_blt.buf_decay  # shape = (inv_blt._num_buffers,)

  # Re-shaping to utilize numpy broadcasting to behave
  # like a double for-loop, omega and theta should
  # have the same length.
  omega_pairs = omega * omega[:, jnp.newaxis]
  theta_pairs = (1 - theta) * (1 - theta[:, jnp.newaxis])
  cross_term_sum = jnp.sum(omega_pairs / theta_pairs)

  return 1 + 2 * jnp.sum(omega / (1 - theta)) + cross_term_sum


@jax.jit
@require_output_scale_gt_zero
def max_loss(blt: BufferedToeplitz, n: jax.Array) -> jax.Array:
  """Max squared error scaled by sensitivity**2."""
  check_float64_dtype(blt)
  sens_squared = sensitivity_squared(blt, n)
  inv_blt = blt.inverse(skip_checks=True)
  error = max_error(inv_blt, n)
  return error * sens_squared


@jax.jit
@require_output_scale_gt_zero
def limit_max_loss(
    blt: BufferedToeplitz,
) -> jax.Array:
  """Limit of (1/n) max squared error scaled by sensitivity**2."""
  check_float64_dtype(blt)
  sens_squared = sensitivity_squared(blt, n=jnp.inf)
  inv_blt = blt.inverse(skip_checks=True)
  error = limit_max_error(inv_blt)
  return error * sens_squared

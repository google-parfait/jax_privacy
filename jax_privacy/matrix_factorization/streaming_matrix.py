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

"""Definition of streaming matrix interface."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
from typing import Any, Generic, TypeAlias, TypeVar

import jax
from jax import numpy as jnp
import optax

State = TypeVar('State', bound=optax.ArrayTree)
Shape: TypeAlias = tuple[int, ...]
ShapePyTree = Any

# Disabling pylint invalid-name to allow mathematical notation including
# single-capital-letter variables for matrices.
# See README.md for notation conventions.
# pylint:disable=invalid-name


@dataclasses.dataclass(frozen=True)
class StreamingMatrix(Generic[State]):
  # pylint: disable-next=line-too-long
  r"""A linear mapping :math:`x \to Ax` for a lower-triangular (streaming) matrix :math:`A`.

  Via the attributes / member functions ``init_multiply`` and ``multiply_next``,
  this class allows you to efficiently compute a linear mapping
  :math:`x \to Ax` in streaming fashion (one element at a time). The precise
  meaning of the term "efficiently" is implementation-dependent, with examples
  including constant memory overhead, and / or without fully materializing
  :math:`A` or :math:`x`.

  Example Usage:
    >>> A = prefix_sum()
    >>> x = jnp.arange(1, 5).astype(float)
    >>> slices = []
    >>> state = A.init_multiply(x[0])
    >>> for i in range(len(x)):
    ...   result_slice, state = A.multiply_next(x[i], state)
    ...   slices.append(result_slice)
    >>> Ax = jnp.stack(slices)
    >>> print(Ax)
    [ 1.  3.  6. 10.]
    >>> print(jnp.cumsum(x))
    [ 1.  3.  6. 10.]

  See the constructor docstring for a full description of ``init_multiply`` and
  ``multiply_next``.

  Importantly, this design encodes the fact that ``Ax[i]`` may only depend on
  ``x[i]`` and state captured from computing ``Ax[0], ..., Ax[i-1]``. This is
  equivalent to :math:`A` having a lower-triangular matrix representation in the
  standard basis.

  In general, :math:`A` and :math:`x` may both be infinite; thus we sidestep the
  question of how many elements of :math:`A x` one wishes to compute by
  assuming the user provides a range.

  Attributes:
    init_multiply: A function that returns the initial state given the expected
      shape of inputs to each call to multiply_next.
    multiply_next: A function that returns (next_slice, updated_state) from
      (next_input, current_state).
  """

  init_multiply: Callable[[optax.ArrayTree], State]
  multiply_next: Callable[
      [optax.ArrayTree, State], tuple[optax.ArrayTree, State]
  ]

  @classmethod
  def from_array_implementation(
      cls,
      init_multiply_fn: Callable[[jax.Array | jax.ShapeDtypeStruct], State],
      multiply_next_fn: Callable[[jax.Array, State], tuple[jax.Array, State]],
  ) -> StreamingMatrix:
    """Construct a StreamingMatrix object from an implementation of init/next.

    This class method expects the ``init_multiply_fn`` and ``multiply_next_fn``
    to be defined w.r.t. a single ``jax.Array`` input. These implementations
    will be "lifted" to operate on pytrees of arrays.

    Args:
      init_multiply_fn: a function that returns the initial state given the
        expected shape of inputs to each call to next_fn.
      multiply_next_fn: a function that returns (next_slice, updated_state) from
        (next_input, current_state).

    Returns:
      A StreamingMatrix that operates over PyTrees of ``jax.Array`` objects.
    """

    def tree_unzip(tree, treedef):
      leaves = treedef.flatten_up_to(tree)
      return tuple(treedef.unflatten(x) for x in zip(*leaves))

    def lifted_init(abstract_value):
      return jax.tree.map(init_multiply_fn, abstract_value)

    def lifted_next(value, state):
      return tree_unzip(
          jax.tree.map(multiply_next_fn, value, state),
          jax.tree.structure(value),
      )

    return cls(lifted_init, lifted_next)

  def materialize(self, n: int) -> jax.Array:
    """A utility method to materialize this matrix as an n x n ndarray.

    Note ``n`` needs to be a parameter, because a general
    :class:`StreamingMatrix`
    can represent an infinite-dimensional matrix.

    NOTE: Primarily for debugging and testing implementations of init and next.

    Args:
      n: The size of the square matrix to materialize.

    Returns:
      An n x n materialization of this matrix.
    """
    return self @ jnp.eye(n)

  def row_norms_squared(self, n: int, scan_fn=jax.lax.scan) -> jax.Array:
    r"""Computes the row-wise :math:`L_2^2` norm of the matrix.

    Given a StreamingMatrix :math:`B = A C^{-1}`, this function computes the
    per-query expected squared error of the factorization :math:`A = BC`. The
    expected total squared error and the maximum expected squared error can be
    computed from this vector via :func:`jax.numpy.sum` and
    :func:`jax.numpy.max` respectively.

    This function consumes an optional scan_fn argument, which is primarily
    useful if you need to backpropagate through this function, in which case
    using a checkpointed scan can be helpful to avoid OOMing on GPUs.
    For example, the scan_fn defined below would store 8 intermediate states
    of the scan in memory, rather than the default which stores the entire
    scan history in memory during backpropagation.  The number of checkpoints
    determines the computation / memory tradeoff.

    ::

      scan_fn = functools.partial(
        equinox.internal.scan,
        kind='checkpointed',
        checkpoints=8,
      )

    Args:
      n: The number of rows to compute squared-norms of.
      scan_fn: A function with the same signature as jax.lax.scan.

    Returns:
      A vector of length n containing the row-wise :math:`L_2^2` norm of the
      matrix.
    """
    zero = jnp.zeros(n)

    def next_state_and_row_norm(state, i):
      # Note: state is first for use in scan_fn.
      ei = zero.at[i].set(1)
      row, state = self.multiply_next(ei, state)
      return state, row @ row

    return scan_fn(
        next_state_and_row_norm, self.init_multiply(zero), jnp.arange(n)
    )[1]


# TODO: b/329444015 - Consider making protected and updating callsites
# to call the member-function directly.
def scale_rows_and_columns(
    matrix: StreamingMatrix,
    row_scale: jax.Array | None = None,
    col_scale: jax.Array | None = None,
) -> StreamingMatrix:
  """Returns a new :class:`StreamingMatrix` with scaled rows and/or cols.

  Assumes row_scale and col_scale can be indexed into for as many outputs
  are generated from matrix. If ``jax.Array`` objects are used, note
  row_scale[i] for i > len(row_scale) will return row_scale[-1].

  Args:
    matrix: The matrix to wrap.
    row_scale: Multipliers to apply to the rows of ``matrix``, equivalent to
      ``jnp.diag(row_scale) @ matrix``.
    col_scale: Multipliers to apply to the columns of ``matrix``, equivalent to
      ``matrix @ jnp.diag(col_scale)``.

  Returns:
    The wrapped :class:`StreamingMatrix`.
  """
  result = matrix
  if row_scale is not None:
    result = multiply_streaming_matrices(diagonal(row_scale), result)
  if col_scale is not None:
    result = multiply_streaming_matrices(result, diagonal(col_scale))
  return result


def multiply_array(A: StreamingMatrix, x: jax.Array) -> jax.Array:
  r"""Computes the matrix-vector product :math:`Ax`."""

  # Reverse (value, state) -> (state, value) for scan.
  def f(state, value):
    return A.multiply_next(value, state)[::-1]

  return jax.lax.scan(f, A.init_multiply(x[0]), x)[1]


# TODO: b/329444015 - Consider making protected and updating callsites
# to call the member-function directly.
def multiply_streaming_matrices(
    A: StreamingMatrix,
    B: StreamingMatrix,
) -> StreamingMatrix:
  r"""Multiply a StreamingMatrix by another StreamingMatrix.

  Args:
    A: The left hand side matrix
    B: The right hand side matrix

  Returns:
    :math:`AB`, represented as another StreamingMatrix.
  """

  def init_multiply(abstract_value):
    B_state = B.init_multiply(abstract_value)
    intermediate, _ = jax.eval_shape(B.multiply_next, abstract_value, B_state)
    return A.init_multiply(intermediate), B_state

  def multiply_next(value, state):
    A_state, B_state = state
    inner, B_state = B.multiply_next(value, B_state)
    outer, A_state = A.multiply_next(inner, A_state)
    return outer, (A_state, B_state)

  return StreamingMatrix(init_multiply, multiply_next)


def identity() -> StreamingMatrix:
  """An implicit representation of the identity matrix."""
  return StreamingMatrix(lambda _: (), lambda value, _: (value, ()))


def prefix_sum() -> StreamingMatrix:
  """An implicit representation of the lower triangular matrix of ones."""

  def init_multiply(abstract_value):
    return jnp.zeros_like(abstract_value)

  def multiply_next(state, value):
    result = state + value
    return result, result

  return StreamingMatrix.from_array_implementation(init_multiply, multiply_next)


def diagonal(diag: jax.Array) -> StreamingMatrix:
  """An implicit representation of a diagonal matrix.

  The returned StreamingMatrix represents an infinitely large diagonal matrix.
  The diagonal elements are taken from the provided array ``diag`` up to row
  n = diag.size, and is equal to diag[-1] beyond that point.

  Args:
    diag: A 1D array of diagonal elements.

  Returns:
    A StreamingMatrix representing the corresponding diagonal matrix.
  """
  return StreamingMatrix.from_array_implementation(
      lambda _: jnp.array(0),
      lambda value, i: (value * diag.at[i].get(mode='clip'), i + 1),
  )


def momentum_sgd_matrix(
    momentum: float = 0, learning_rates: jax.Array | None = None
) -> StreamingMatrix:
  """An implicit representation of the momentum sgd matrix."""
  lr_sched = jnp.ones(1) if learning_rates is None else learning_rates
  if lr_sched.min() <= 0.0:
    raise ValueError(
        'Learning rates must be positive (zero learning rates may prevent '
        f'matrix factorization from succeeding.) Found {learning_rates}'
    )

  def init_multiply(abstract_value):
    dtype = jnp.promote_types(abstract_value.dtype, lr_sched.dtype)
    zero = jnp.zeros_like(abstract_value, dtype=dtype)
    return jnp.array(0), zero, zero

  def multiply_next(
      value: jax.Array, state: tuple[int, jax.Array, jax.Array]
  ) -> tuple[jax.Array, tuple[int, jax.Array, jax.Array]]:
    index, momentum_buf, result = state
    momentum_buf = momentum * momentum_buf + value
    # If index is out-of-bounds, return the last value in the array.
    result = result + lr_sched.at[index].get(mode='clip') * momentum_buf
    updated_state = (index + 1, momentum_buf, result)
    return result, updated_state

  return StreamingMatrix.from_array_implementation(init_multiply, multiply_next)


T = TypeVar('T', StreamingMatrix, jax.Array)


def _multiply_any(self: StreamingMatrix, other: T) -> T:
  """Multiply a StreamingMatrix by an array or a StreamingMatrix."""
  if isinstance(other, StreamingMatrix):
    return multiply_streaming_matrices(self, other)
  elif isinstance(other, jax.Array):
    return multiply_array(self, other)
  else:
    raise ValueError(f'Unsupported type for multiplication: {type(other)}')


def _multiply_scalar(self: StreamingMatrix, other: float) -> StreamingMatrix:
  """Multiply a StreamingMatrix by a scalar."""
  return multiply_streaming_matrices(self, diagonal(jnp.array([other])))


# Add some syntax sugar.
StreamingMatrix.__matmul__ = _multiply_any
StreamingMatrix.__mul__ = _multiply_scalar
StreamingMatrix.scale_rows_and_columns = scale_rows_and_columns

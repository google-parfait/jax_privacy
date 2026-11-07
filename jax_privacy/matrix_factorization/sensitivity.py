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

"""Library for computing sensitivity under multiple participations."""

import functools

import jax
import jax.numpy as jnp
from jax_privacy import _validate
import numpy as np
import numpy.typing  # pylint: disable=unused-import  # Makes np.typing available

from . import _checks as checks

# Disabling pylint invalid-name to allow mathematical notation including
# single-capital-letter variables for matrices.
# See README.md for notation conventions.
# pylint:disable=invalid-name


def single_participation_sensitivity(C: np.typing.ArrayLike) -> float:
  """Returns the sensitivity of a matrix with a single participation."""
  checks.check(C=C)
  return np.linalg.norm(C, axis=0).max()


def _ceil_div(x, y):
  """Integer division, rounding up (ceiling)."""
  return -(x // -y)


def minsep_true_max_participations(
    n: int, min_sep: int, max_participations: int | None = None
) -> int:
  """Returns the maximum number of participations for a min_sep pattern.

  This might be less than the ``max_participations`` limit (if it is given)
  because ``n`` is too small.

  Args:
    n: The number of rounds.
    min_sep: The minimum separation between participations, where ``min_sep=1``
      means adjacent indices can be selected.
    max_participations: The maximum number of participations.

  Returns:
    The largest number of participations that are actually possible based on the
    inputs.
  """
  # Ideally we would assert min_sep > 0 here, but we want to
  # use this function in jitted contexts.
  # TODO: b/329444015 - Add a skip_checks: bool = False option, add
  # checks here by default, and only disable in jitted contexts.
  max_part_ub = _ceil_div(n, min_sep)
  if max_participations is None:
    return max_part_ub
  else:
    return min(max_participations, max_part_ub)


def max_participation_for_linear_fn(
    x: jnp.ndarray, min_sep: int = 1, max_participations: int | None = None
) -> jnp.ndarray:
  # pylint: disable-next=line-too-long
  r"""Returns :math:`\max_u \langle x, u \rangle`, where u respects the given participation pattern.

  The vector ``u`` is represented by a set of indices that satisfy ``min_sep``
  and ``max_participations``. Note that the signs of the entries in ``x`` matter
  as we find :math:`\max_u \langle u, x \rangle`. Because the optimization is
  in one dimension, we can solve using a dynamic program with running time
  :math:`O(\operatorname{len}(x) \cdot (\text{max\_participations} + 1))`.

  Reference: See `(Amplified) Banded Matrix Factorization: A unified approach
  to private training <https://arxiv.org/abs/2306.08153>`_, Algorithm 3
  (VecSens).

  Arguments:
    x: A vector of values to optimize over.
    min_sep: Minimum separation between selected indices, e.g., if :math:`i` and
      :math:`j` are selected we must have :math:`|i - j| \ge
      \text{min}_\text{sep}`, so e.g. ``min_sep=1`` means adjacent indices can
      be selected.
    max_participations: Optional, the maximum number of participations. If
      ``None``, then ``max_participations`` is determined from ``len(x)`` and
      ``min_sep``.

  Returns:
    The optimal value.
  """
  # F[i, k] is the maximum attainable assuming
  # at most k participations for the subvector x[i:]
  # For each i, k, we have two choices:
  # 1. Add x[i] to solution, increase i by min_sep, decrease k by 1.
  # 2. Do not add x[i] to solution, find solution for subvector x[i+1:].
  # Thus, we have: F[i,k] = max { x[i] + F[i + min_sep, k-1], F[i + 1, k] }
  n = len(x)
  max_participations = minsep_true_max_participations(
      n, min_sep, max_participations
  )

  # Below is an equivalent vectorized implementation, where f = F[:, k].
  # Entries in f after f[n-1] will always be zero, so this is for convenience:
  f = jnp.zeros(n + min_sep)
  for _ in range(max_participations):
    # Update so f[i] reflects selecting x[i] plus optimally choosing
    # at most k-1 additional values to the right of i:
    f = f.at[:-min_sep].set(x + f[min_sep:])
    # Accumulate max right-to-left: instead of choosing i, we can
    # always elect to choose some value > i.
    f = jax.lax.cummax(f, reverse=True)
  return f[0]


def banded_lower_triangular_mask(n: int, num_bands: int) -> jnp.ndarray:
  """Returns n x n lower-triangular {0, 1} matrix with b bands of 1s."""
  _validate.positive(num_bands=num_bands)
  return (np.tri(n) - np.tri(n, k=-num_bands)).astype(np.int32)


def banded_symmetric_mask(n: int, num_bands: int) -> jnp.ndarray:
  """Returns n x n symmetric {0, 1} matrix with 2b - 1 bands of 1s."""
  _validate.positive(num_bands=num_bands)
  return (np.tri(n, k=num_bands - 1) - np.tri(n, k=-num_bands)).astype(np.int32)


def get_min_sep_sensitivity_upper_bound_for_X(
    X: jnp.ndarray, min_sep: int = 1, max_participations: int | None = None
) -> float:
  """Computes an upper bound on the min_sep sensitivity of X.

  Unlike get_sensitivity_banded_for_X, this method does not require X to be
  banded, and will provide a valid upper bound on sensitivity for any X.

  How this algorithm works:
    To compute min_sep sensitivity normally, we need to find the submatrix
    X_{pi, pi} with maximum L1 norm (sum of absolute values) over all possible
    pi satisfying the (min_sep, max_part) participation schema.

    This upper bound works in two stages:
      1. Find the subvector x_{pi} with maximum L1 norm for each row x of X.
      2. Call maximum L1 norms v_1, ..., v_n.
      3. Find the subvector v_{pi} with maximum L1 norm.

    This is a valid upper bound because it is computing the maximum over an
    enlarged space of possibilities.  Now, the subset of columns chosen do not
    have to be the same across rows, and the subset of rows chosen does not
    have to match the subset of columns chosen.

  If X is min_sep-banded, this upper bound is identical to
  get_sensitivity_banded_for_X, but this function requires O(n^2 k) time while
  get_sensitivity_banded_for_X requires O(n^2 + n k) time.

  Reference: See `Choquette-Choo et al. (2023)
  <https://arxiv.org/abs/2306.08153>`_.

  Args:
    X: The Gram matrix of the encoder matrix, C.T @ C.
    min_sep: Minimum separation between the participations of the same user. For
      example, ``min_sep = 1`` means the same user could participate on two
      consecutive rounds.
    max_participations: Optional, the maximum number of participations. If
      ``None``, then ``max_participations`` is determined from ``n.shape[0]``
      and ``min_sep``.

  Returns:
    The L2 sensitivity of :math:`C` satisfying :math:`C^T C = X`.
  """
  checks.check(X=X)
  row_max = jax.vmap(
      functools.partial(
          max_participation_for_linear_fn,
          min_sep=min_sep,
          max_participations=max_participations,
      )
  )(jnp.abs(X))
  result = max_participation_for_linear_fn(row_max, min_sep, max_participations)
  return float(jnp.sqrt(result))


def get_min_sep_sensitivity_upper_bound(
    C: jnp.ndarray, min_sep: int = 1, max_participations: int | None = None
) -> float:
  """Like get_min_sep_sensitivity_upper_bound_for_X, but takes the encoder C."""
  checks.check(C=C)
  return get_min_sep_sensitivity_upper_bound_for_X(
      C.T @ C, min_sep, max_participations
  )


def get_sensitivity_banded_for_X(
    X: jnp.ndarray,
    min_sep: int = 1,
    max_participations: int | None = None,
) -> float:
  r"""Computes the sensitivity of an X.

  This method requires (and checks) that the number of bands is less than
  or equal to ``min_sep + 1``.

  Note: The :math:`C` from which :math:`X` is derived being lower-triangular
  with at most ``min_sep`` non-zero bands including the main diagonal is a
  sufficient but not necessary condition for this calculation to hold. All that
  is necessary is that for two columns :math:`i` and :math:`j` with
  :math:`|i - j| > \text{min}_\text{sep}` are orthogonal,
  ``np.dot(h[:, i], h[:, j]) == 0``.

  Args:
    X: The Gram matrix of the encoder matrix, C.T @ C, which must be
      min_sep-banded (see README.md).
    min_sep: Minimum separation between the participations of the same user. For
      example, ``min_sep = 1`` means the same user could participate on two
      consecutive rounds.
    max_participations: Optional, the maximum number of participations. If
      ``None``, then ``max_participations`` is determined from ``n.shape[0]``
      and ``min_sep``.

  Returns:
    The L2 sensitivity of :math:`C` satisfying :math:`C^T C = X`.
  """
  checks.check(X=X)
  n = X.shape[0]
  _validate.in_range(1, n, min_sep=min_sep)

  # Check the condition under which this approach holds.
  expected_zeros_in_x = ~banded_symmetric_mask(n, min_sep).astype((jnp.bool_))

  if not jnp.all(X[expected_zeros_in_x] == 0):
    raise ValueError(
        'All columns of C corresponding to iterations i and j where '
        'user might participate (that is, |i - j| < min_sep + 1) '
        'must be orthogonal.'
    )

  x = jnp.diag(X)
  value = max_participation_for_linear_fn(x, min_sep, max_participations)
  return float(jnp.sqrt(value))


def get_sensitivity_banded(
    C: jnp.ndarray,
    min_sep: int = 1,
    max_participations: int | None = None,
) -> float:
  """Like get_sensitivity_banded_for_X(), but takes the encoder C."""
  checks.check(C=C)
  return get_sensitivity_banded_for_X(C.T @ C, min_sep, max_participations)


def fixed_epoch_sensitivity_for_X(X: jnp.ndarray, epochs: int) -> float:
  """Compute the sensitivity of X under (k,b)-fixed-epoch participation.

  Note that X can contain negative entries, which are handled essentially
  by taking an absolute value of the necessary entries.

  Reference: See `Choquette-Choo et al. (2022)
  <https://arxiv.org/abs/2211.06530>`_, and `Choquette-Choo et al. (2023)
  <https://arxiv.org/abs/2306.08153>`_ for the absolute-value trick.
  Essentially, this implementation applies Eq. (5) to the ``b = n / k`` possible
  participation patterns under (k, b)-fixed-epoch participation.

  Args:
    X: the Gram matrix of the encoder, which may contain negative entries.
    epochs: the number of epochs. Must be square in shape.

  Returns:
    The sensitivity of the matrix factorization mechanism defined by X.
  """
  # pytype fails to catch this type hint, leading to issues post-build.
  if not isinstance(X, jnp.ndarray):
    X = jnp.array(X)
  checks.check(X=X)
  _validate.positive(epochs=epochs)
  if X.shape[0] % epochs != 0:
    raise ValueError(f'epochs={epochs} must divide n={X.shape[0]}.')
  rounds_per_epoch = X.shape[0] // epochs
  submatrix_size = int(X.shape[0] // rounds_per_epoch)

  # The below is a peculiar way of indexing into an array. However, this is
  # needed because we require a static shape of X[x, y] to be known at all times
  # inside the function `submatrix_squared_sensitivity` below.
  all_indices = jnp.arange(0, X.shape[0], dtype=jnp.int64)
  all_indices = jnp.reshape(all_indices, (submatrix_size, -1))

  def submatrix_squared_sensitivity(
      indices: jnp.ndarray, X: jnp.ndarray
  ) -> jnp.ndarray:
    x, y = jnp.meshgrid(indices, indices)
    return jnp.abs(X[x, y]).sum()

  squared_sensitivity_of_X = functools.partial(
      submatrix_squared_sensitivity, X=X
  )
  squared_sensitivities = jax.vmap(squared_sensitivity_of_X, in_axes=1)
  return float(jnp.sqrt(jnp.max(squared_sensitivities(all_indices))))


def fixed_epoch_sensitivity(C: jnp.ndarray, epochs: int) -> float:
  """Like fixed_epoch_sensitivity_for_X(), but takes the encoder C."""
  checks.check(C=C)
  return fixed_epoch_sensitivity_for_X(C.T @ C, epochs)

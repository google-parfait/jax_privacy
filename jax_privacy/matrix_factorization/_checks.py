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

"""Utilities for confirming various symbols are used correctly."""

from jax_privacy import _validate
import numpy as np

# Disabling pylint invalid-name to allow mathematical notation including
# single-capital-letter variables for matrices.
# See README.md for notation conventions.
# pylint:disable=invalid-name


def _pad(s: str):
  return s + ' ' if s else ''


def check_lower_triangular(
    M: np.typing.ArrayLike, name: str = '', **allclose_kwargs
):
  if not np.allclose(M, np.tril(M), **allclose_kwargs):
    raise ValueError(
        f'Matrix {_pad(name)}should be lower-triangular, found\n{M}'
    )


def check_is_matrix(M: np.typing.ArrayLike, name: str = ''):
  _validate.ndim(2, **{name or 'matrix': M})


def check_square(M: np.typing.ArrayLike, name: str = ''):
  if (np.ndim(M) != 2) or (np.shape(M)[0] != np.shape(M)[1]):
    raise ValueError(f'Matrix {_pad(name)}should be square, found\n{M}')


def check_finite(M: np.typing.ArrayLike, name: str):
  if not np.all(np.isfinite(M)):
    raise ValueError(f'Matrix {_pad(name)}is not finite, found\n{M}')


def check_symmetric(M: np.typing.ArrayLike, name: str, **allclose_kwargs):
  M = np.asarray(M)
  if not np.allclose(M, M.T, **allclose_kwargs):
    raise ValueError(f'Matrix {_pad(name)}should be symmetric, found\n{M}')


def check(
    *,
    A: np.typing.ArrayLike | None = None,
    B: np.typing.ArrayLike | None = None,
    C: np.typing.ArrayLike | None = None,
    X: np.typing.ArrayLike | None = None,
    **allclose_kwargs,
):
  """Apply checks to matrices A = B @ C and X = C.T @ C.

  These checks are based on the current assumptions typical in this
  codebase, not on the most general possibilities allowed by the theory.
  For example, in full generality A need not be square, and B and C need
  not be lower triangular when they are square. However, these are the typical
  assumptions in this codebase, and hence we check these properties to err
  on the side of catching bugs; when needed these checks can always be removed
  in places that allow more general assumptions, or these checks can be updated.

  Any subset of the matrices can be provided.

  Args:
    A: The workload matrix.
    B: The B decoder matrix, such that A = B @ C.
    C: The encoder matrix.
    X: Symmetric matrix C.T @ C.
    **allclose_kwargs: kwargs to pass to np.allclose

  Raises:
    ValueError if matrices do not satisfy expected properties.
  """
  not_none = {}
  n = None  # Number of iterations

  if A is not None:
    n = np.shape(A)[0]
    check_finite(A, 'A')
    check_square(A, 'A')
    check_lower_triangular(A, 'A', **allclose_kwargs)
    not_none['A'] = A

  if B is not None:
    n, k = np.shape(B)
    check_finite(B, 'B')
    check_is_matrix(B, 'B')
    if n == k:
      # If B is square, it should be lower triangular.
      check_lower_triangular(B, 'B', **allclose_kwargs)
    not_none['B'] = B

  if C is not None:
    k, n = np.shape(C)
    check_finite(C, 'C')
    check_is_matrix(C, 'C')
    if k == n:
      # If C is square, it should be lower triangular.
      check_lower_triangular(C, 'C', **allclose_kwargs)
      # Note - square C should be invertible, but keeping checks lightweight.
    not_none['C'] = C

  if X is not None:
    n = np.shape(X)[0]
    check_finite(X, 'X')
    check_square(X, 'X')
    check_symmetric(X, 'X', **allclose_kwargs)
    # Note - X should also PD, but for now keeping checks lightweight.
    not_none['X'] = X

  # Verify properties to make sure this set of matrices is consistent.
  # We could also consider checking A = B @ C and X = C.T @ C.
  # It might be better to split a flag for "full_check" or similar,
  # to allow different speed  safety tradeoffs.

  if (B is not None) and (C is not None) and (np.shape(B)[1] != np.shape(C)[0]):
    raise ValueError(
        'B and C shapes do not match. Expected '
        'B.shape[1] == C.shape[0], but found '
        f'{np.shape(B)=} and {np.shape(C)=}'
    )

  expected_shapes = {
      'A': ('n', 'n'),
      'B': ('n', 'k'),
      'C': ('k', 'n'),
      'X': ('n', 'n'),
  }
  # n should always be not-None here if not_none is non-empty.
  correct_shapes = True
  for name, m in not_none.items():
    for i, letter in enumerate(expected_shapes[name]):
      if (letter == 'n') and (m.shape[i] != n):
        correct_shapes = False

  if not correct_shapes:
    raise ValueError(
        f'Expected matrix shapes to match {expected_shapes}, but found shapes: '
        + str({k: v.shape for k, v in not_none.items()})
    )

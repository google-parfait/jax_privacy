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

"""Centralized validation utilities for jax_privacy."""

import copy
import zlib

import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Generic validators
# ---------------------------------------------------------------------------


def non_negative(**kwargs):
  """Validates that all values are non-negative."""
  for name, value in kwargs.items():
    if value < 0:
      raise ValueError(f'Expected {name}={value} >= 0.')


def positive(**kwargs):
  """Validates that all values are positive and finite."""
  for name, value in kwargs.items():
    # A single negated check also rejects NaN, for which `value <= 0` is False.
    if not (np.isfinite(value) and value > 0):
      raise ValueError(f'Expected {name}={value} > 0.')


def in_range(lo, hi, closed='both', **kwargs):
  """Validates that all values are in the interval from lo to hi.

  Args:
    lo: The lower endpoint of the interval.
    hi: The upper endpoint of the interval.
    closed: Which endpoints are included: ``'both'`` for ``[lo, hi]``,
      ``'left'`` for ``[lo, hi)``, ``'right'`` for ``(lo, hi]``, or
      ``'neither'`` for ``(lo, hi)``.
    **kwargs: The named values to validate.
  """
  if closed not in ('both', 'left', 'right', 'neither'):
    raise ValueError(f'Unknown closed={closed!r}.')
  lo_closed = closed in ('both', 'left')
  hi_closed = closed in ('both', 'right')
  for name, value in kwargs.items():
    lo_ok = lo <= value if lo_closed else lo < value
    hi_ok = value <= hi if hi_closed else value < hi
    if not (lo_ok and hi_ok):
      interval = (
          f"{'[' if lo_closed else '('}{lo}, {hi}{']' if hi_closed else ')'}"
      )
      raise ValueError(f'Expected {name}={value} in {interval}.')


def equal(expected, **kwargs):
  """Validates that all values equal ``expected``."""
  for name, value in kwargs.items():
    if value != expected:
      raise ValueError(
          f'Provided {name}={value} does not match expected {expected}.'
      )


def tree_structure(reference, **kwargs):
  """Validates that all values have the same PyTree structure as reference."""
  expected = jax.tree.structure(reference)
  for name, value in kwargs.items():
    actual = jax.tree.structure(value)
    if actual != expected:
      raise ValueError(
          f'Expected {name} to have PyTree structure {expected}, got {actual}.'
      )


def is_int(context: str = '', **kwargs):
  """Validates that all values are integers."""
  for name, value in kwargs.items():
    if value != int(value):
      raise ValueError(f'Expected {name}={value} to be an integer. {context}')


def is_none(context: str = '', **kwargs):
  """Validates that all values are None."""
  for name, value in kwargs.items():
    if value is not None:
      raise ValueError(f'Expected {name}={value} to be None. {context}')


def batch(pytree, padding_example=None) -> int:
  """Validates a batch pytree and returns the batch size.

  Checks that the pytree has at least one leaf, all leaves are arrays with
  at least one dimension, and all leaves have the same size along axis 0.
  If ``padding_example`` is provided, also checks that ``pytree`` matches its
  PyTree structure, per-example shapes, and dtypes.

  Args:
    pytree: A pytree of arrays representing a batch of data.
    padding_example: Optional single-example PyTree specification to validate
      against.

  Returns:
    The batch size (size of the first dimension).

  Raises:
    ValueError: If the pytree is empty, contains non-array leaves, leaves
      with zero dimensions, leaves with inconsistent first-axis sizes, or
      does not match ``padding_example``.
  """
  leaves = jax.tree.leaves(pytree)
  if not leaves:
    raise ValueError('Batch pytree is empty or contains no leaves.')
  for i, leaf in enumerate(leaves):
    if not hasattr(leaf, 'shape'):
      raise ValueError(
          f'Expected all batch leaves to be arrays, but leaf {i} has type '
          f'{type(leaf).__name__}.'
      )
    if leaf.ndim == 0:
      raise ValueError(
          'Expected all batch leaves to have at least one dimension, but '
          f'leaf {i} is a scalar.'
      )
  sizes = {leaf.shape[0] for leaf in leaves}
  if len(sizes) != 1:
    raise ValueError(
        'All batch leaves must have the same size along axis 0, '
        f'got sizes: {sorted(sizes)}.'
    )
  if padding_example is not None:
    tree_structure(padding_example, dataset=pytree)
    data_meta = [(x.shape[1:], x.dtype) for x in leaves]
    pad_meta = [(x.shape, x.dtype) for x in jax.tree.leaves(padding_example)]
    equal(pad_meta, dataset_shape_dtype=data_meta)
  return sizes.pop()


def multihost(first_example, rng):
  """Validates that the dataset and RNG match across all JAX processes."""
  if jax.process_count() > 1:
    to_bytes = lambda x: getattr(x, 'tobytes', lambda: str(x).encode())()
    crcs = [
        zlib.crc32(to_bytes(x)) % 2**31 for x in jax.tree.leaves(first_example)
    ]
    seed = copy.deepcopy(np.random.default_rng(rng)).integers(2**31)
    multihost_utils.assert_equal(
        (seed, *crcs), 'dataset and rng must match across all JAX processes'
    )


def strategy(value, max_size):
  """Validates BandMF strategy coefficients.

  Checks that the strategy is a 1D array with size in ``[1, max_size]``.

  Args:
    value: The strategy coefficients as an array-like.
    max_size: The maximum allowed size (typically the number of iterations).

  Raises:
    ValueError: If the strategy is not 1D or its size is out of range.
  """
  arr = np.asarray(value)
  if arr.ndim != 1:
    raise ValueError(f'strategy must be a 1D array, got {arr.ndim}D.')
  if arr.size == 0 or arr.size > max_size:
    raise ValueError(
        f'strategy size must be in [1, {max_size}], got {arr.size}.'
    )


def multi_owner(example_ids, user_ids):
  """Validates parallel edge-list arrays for a MultiOwnerGraph."""
  example_ids = np.asarray(example_ids)
  user_ids = np.asarray(user_ids)
  if example_ids.ndim != 1 or user_ids.ndim != 1:
    raise ValueError('example_ids and user_ids must be 1D arrays.')
  if len(example_ids) != len(user_ids):
    raise ValueError('example_ids and user_ids must have the same length')
  if len(example_ids) == 0:
    raise ValueError('empty graphs are not allowed.')
  pairs = np.stack([example_ids, user_ids], axis=1)
  if len(np.unique(pairs, axis=0)) < len(pairs):
    raise ValueError('Duplicate (example, user) id pairs are not allowed.')


def x64_enabled():
  """Validates that 64-bit mode is enabled."""
  if not jax.config.jax_enable_x64:
    raise ValueError(
        'grid_scale requires 64-bit mode. Call '
        "jax.config.update('jax_enable_x64', True) before using "
        'grid_scale, otherwise jnp.int64 is silently truncated to '
        'int32.'
    )


def discrete_clipping(
    grid_scale,
    rescale_to_unit_norm=False,
    normalize_by=1.0,
    l2_clip_norm=None,
):
  """Validates discrete Gaussian clipping parameters."""
  if grid_scale is None:
    return
  if rescale_to_unit_norm:
    raise ValueError('rescale_to_unit_norm cannot be set with grid_scale.')
  if normalize_by != 1.0:
    raise ValueError(
        'normalize_by is not compatible with grid_scale. Normalization '
        'should be applied after noise addition.'
    )
  if l2_clip_norm is not None and not jnp.isscalar(l2_clip_norm):
    raise ValueError(
        'Per-layer PyTree clipping is not compatible with grid scale. '
        'l2_clip_norm must be a Real scalar.'
    )
  x64_enabled()

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

import importlib
import warnings

from absl.testing import absltest
from absl.testing import parameterized
import hypothesis
from hypothesis import strategies as st
import jax
import jax.numpy as jnp
from jax_privacy.matrix_factorization import banded
from jax_privacy.matrix_factorization import streaming_matrix
import numpy as np


# Disabling pylint invalid-name to allow mathematical notation including
# single-capital-letter variables for matrices.
# See README.md for notation conventions.
# pylint:disable=invalid-name

# Some code in this project requires working in f64 space for numerical
# stability, for ease we set the jax config globally.
# TODO: b/329504711 - investigate ways to move this move "locally" into the
# operations below and avoid setting globally for all code that imports this
jax.config.update('jax_enable_x64', True)


def assert_allclose(a, b, atol=1e-6, rtol=1e-6):
  # A wrapper with the same defaults as tf.TestCase.assertAllClose
  np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def is_lower_triangular_banded(C: jnp.ndarray, bands: int) -> bool:
  """Checks if a given matrix is banded and lower triangular."""
  if bands > len(C):
    raise ValueError(f'{bands=}, but C is {C.shape}')
  if bands <= 0:
    raise ValueError(f'bands be >=1, found {bands=}')
  if (len(C.shape) != 2) or (C.shape[0] != C.shape[1]):
    raise ValueError(f'C must be a square matrix, found {C.shape=}')
  n = C.shape[0]
  for i in range(n):
    for j in range(n):
      if abs(i - j) >= bands and C[i, j] != 0:
        return False
      if i < j and C[i, j] != 0:
        return False
  return True


class BandedTest(parameterized.TestCase):

  def test_is_lower_triangular_banded(self):
    with self.assertRaisesRegex(ValueError, 'must be a square'):
      is_lower_triangular_banded(jnp.array([1]), bands=1)
    self.assertTrue(is_lower_triangular_banded(jnp.array([[1]]), bands=1))
    self.assertTrue(is_lower_triangular_banded(jnp.eye(4), bands=1))
    self.assertTrue(is_lower_triangular_banded(jnp.tri(5), bands=5))
    self.assertFalse(is_lower_triangular_banded(jnp.tri(5), bands=4))

  def test_default(self):
    n = 8
    eye = jnp.eye(n, dtype=jnp.float64)

    C = banded.ColumnNormalizedBanded.default(n, 1).materialize()
    np.testing.assert_array_equal(C, eye)
    self.assertEqual(C.dtype, jnp.float64)
    self.assertTrue(is_lower_triangular_banded(C, n))

  def test_default_avoids_divide_by_zero_warning(self):
    """Fichtenberger ratios skip k=0 so (2k-1)/(2k) is never -1/0."""
    with warnings.catch_warnings():
      warnings.simplefilter('error', RuntimeWarning)
      C = banded.ColumnNormalizedBanded.default(8, 4)
    # Unnormalized Algorithm-1 prefixes: 1, 1/2, 3/8, 5/16.
    expected = np.array([1.0, 0.5, 0.375, 0.3125])
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(np.asarray(C.params[0]), expected, atol=1e-6)

  def test_minsep_sensitivity_squared(self):
    C = banded.ColumnNormalizedBanded.default(32, bands=4)
    with self.assertRaises(ValueError):
      banded.minsep_sensitivity_squared(C, 3, 9)
    self.assertEqual(banded.minsep_sensitivity_squared(C, 4, None), 8.0)
    self.assertEqual(banded.minsep_sensitivity_squared(C, 4, 6), 6.0)

  @hypothesis.settings(deadline=None, derandomize=True, max_examples=10)
  @hypothesis.given(n=st.integers(1, 16), bands=st.integers(1, 16))
  def test_banded_is_column_normalized_and_banded(self, n: int, bands: int):
    hypothesis.assume(bands <= n)
    # this also tests from_banded_toeplitz implicitly.
    C = banded.ColumnNormalizedBanded.default(n, bands).materialize()
    assert_allclose(jnp.linalg.norm(C, axis=0), jnp.ones(n))
    self.assertTrue(is_lower_triangular_banded(C, bands))

  def test_parameter_mapping(self):
    params = jnp.array([[1, 4], [2, 5], [3, 6]])
    C = banded.ColumnNormalizedBanded(params=params)
    actual = C.materialize()
    # fmt: off
    # pylint: disable=bad-whitespace
    expected = jnp.array([
        [1 / jnp.sqrt(17), 0,                0],
        [4 / jnp.sqrt(17), 2 / jnp.sqrt(29), 0],
        [0,                5 / jnp.sqrt(29), 1],
    ])
    # pylint: enable=bad-whitespace
    # fmt: on
    assert_allclose(actual, expected)

  def test_toeplitz_constructor(self):
    coef = jnp.array([[1, 2, 3]])
    norm = jnp.linalg.norm(coef)
    C = banded.ColumnNormalizedBanded.from_banded_toeplitz(5, coef)
    exp = jnp.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 0], [1, 0, 0]])
    assert_allclose(C.params, exp / norm)

  @hypothesis.settings(deadline=None, derandomize=True, max_examples=10)
  @hypothesis.given(n=st.integers(1, 16), bands=st.integers(1, 16))
  def test_banded_inverse_matches_materialize(self, n: int, bands: int):
    hypothesis.assume(bands <= n)
    encoder = banded.ColumnNormalizedBanded.default(n, bands)
    materialized_inverse = jnp.linalg.inv(encoder.materialize())
    C1 = encoder.inverse_as_streaming_matrix()
    assert_allclose(materialized_inverse, C1.materialize(n))

    # Test for a tensor RHS.
    rhs = jnp.arange(n * 2 * 3).reshape((n, 2, 3))
    expected = jnp.tensordot(materialized_inverse, rhs, axes=1)
    assert_allclose(expected, C1 @ rhs)

  def maybe_skip(self, scan_fn):
    if scan_fn == 'equinox' and importlib.util.find_spec('equinox') is None:
      self.skipTest('equinox not installed.')
    if scan_fn == 'dinosaur' and importlib.util.find_spec('dinosaur') is None:
      self.skipTest('dinosaur not installed.')

  @parameterized.named_parameters(
      ('equinox', 'equinox'),
      ('dinosaur', 'dinosaur'),
      ('default', jax.lax.scan),
  )
  def test_implicit_error(self, scan_fn=jax.lax.scan):
    self.maybe_skip(scan_fn)
    C0 = banded.ColumnNormalizedBanded.default(9, 1)
    A = streaming_matrix.prefix_sum()
    self.assertEqual(banded.per_query_error(C0, A, scan_fn=scan_fn).mean(), 5.0)

  @parameterized.named_parameters(
      # ('equinox', 'equinox'),  # TODO: b/377404877 - Broken at head.
      ('dinosaur', 'dinosaur'),
      ('default', jax.lax.scan),
  )
  def test_optimize_improves_init(self, scan_fn: jax.lax.scan):
    self.maybe_skip(scan_fn)
    C0 = banded.ColumnNormalizedBanded.default(8, 4)
    A = streaming_matrix.prefix_sum()
    C = banded.optimize(
        8, bands=4, C=C0, A=A, max_optimizer_steps=10, scan_fn=scan_fn
    )
    self.assertLess(
        jnp.mean(banded.per_query_error(C, A)),
        jnp.mean(banded.per_query_error(C0, A)),
    )


if __name__ == '__main__':
  absltest.main()

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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax_privacy.matrix_factorization import sensitivity as sensitivity_lib
import numpy as np

# Disabling pylint invalid-name to allow mathematical notation including
# single-capital-letter variables for matrices.
# See README.md for notation conventions.
# pylint:disable=invalid-name


class SensitivityTest(parameterized.TestCase):
  """Tests for sensitivity."""

  def test_single_participation_sensitivity(self):
    C = jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 3.0, -6.0],
    ])
    np.testing.assert_allclose(
        sensitivity_lib.single_participation_sensitivity(C), 6.0
    )
    np.testing.assert_allclose(
        sensitivity_lib.single_participation_sensitivity(C[:3, :3]), np.sqrt(5)
    )
    np.testing.assert_allclose(
        sensitivity_lib.single_participation_sensitivity(np.array([[3]])), 3.0
    )

  @parameterized.product(
      min_sep=[1, 2, 5, 10], max_participations=[1, 2, 5, None]
  )
  def test_minsep_true_max_participations_n_is_one(
      self, min_sep, max_participations
  ):
    self.assertEqual(
        sensitivity_lib.minsep_true_max_participations(
            n=1, min_sep=min_sep, max_participations=max_participations
        ),
        1,
    )

  def test_minsep_true_max_participations(self):
    minsep_true_max_participations = (
        sensitivity_lib.minsep_true_max_participations
    )
    self.assertEqual(minsep_true_max_participations(n=1, min_sep=1), 1)
    self.assertEqual(
        minsep_true_max_participations(n=1, min_sep=2, max_participations=3), 1
    )

    self.assertEqual(minsep_true_max_participations(n=5, min_sep=1), 5)
    self.assertEqual(minsep_true_max_participations(n=5, min_sep=2), 3)
    self.assertEqual(minsep_true_max_participations(n=5, min_sep=3), 2)
    self.assertEqual(minsep_true_max_participations(n=5, min_sep=4), 2)
    self.assertEqual(minsep_true_max_participations(n=5, min_sep=5), 1)
    self.assertEqual(minsep_true_max_participations(n=5, min_sep=6), 1)

    self.assertEqual(
        minsep_true_max_participations(n=5, min_sep=1, max_participations=2), 2
    )
    self.assertEqual(
        minsep_true_max_participations(n=5, min_sep=2, max_participations=2), 2
    )
    self.assertEqual(
        minsep_true_max_participations(n=5, min_sep=2, max_participations=4), 3
    )

    self.assertEqual(minsep_true_max_participations(n=4, min_sep=1), 4)
    self.assertEqual(minsep_true_max_participations(n=4, min_sep=2), 2)
    self.assertEqual(minsep_true_max_participations(n=4, min_sep=3), 2)
    self.assertEqual(minsep_true_max_participations(n=4, min_sep=4), 1)


class MaxLinearFnTest(absltest.TestCase):
  """Tests for max_participation_for_linear_fn."""

  def test_zero_separation_selects_all(self):
    x = jnp.array([1.0, 0.5, 0.5, 0.5, 0.1])
    val = sensitivity_lib.max_participation_for_linear_fn(x, min_sep=1)
    self.assertEqual(val, 2.6)

  def test_no_separation(self):
    # No separation, select ajacent, select first and last
    x = jnp.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    val = sensitivity_lib.max_participation_for_linear_fn(x, min_sep=1)
    self.assertEqual(val, 4.0)

  def test_multiple_solutions(self):
    x = jnp.array([1.0, 1.0, 1.0, 1.0])
    val = sensitivity_lib.max_participation_for_linear_fn(
        x, min_sep=2, max_participations=2
    )
    self.assertEqual(val, 2.0)

  def test_no_participations(self):
    x = jnp.array([1.0, 1.0, 1.0, 1.0])
    val = sensitivity_lib.max_participation_for_linear_fn(
        x, min_sep=1, max_participations=0
    )
    self.assertEqual(val, 0.0)

  def test_length_one(self):
    # Length 1, separation doesn't matter
    for min_sep in [1, 2, 3]:
      x = jnp.array([2.0])
      val = sensitivity_lib.max_participation_for_linear_fn(
          x, min_sep=min_sep, max_participations=4
      )
      self.assertEqual(val, 2.0)

  def test_finds_single_participaton_optimal(self):
    for opt_index in range(5):
      x = jnp.zeros(shape=5).at[opt_index].set(1)
      val = sensitivity_lib.max_participation_for_linear_fn(
          x, min_sep=2, max_participations=1
      )
      self.assertEqual(val, 1.0)

  def test_optimal_has_fewer_than_max_participations(self):
    x = jnp.array([0.0, 1.0, 0.0])
    val = sensitivity_lib.max_participation_for_linear_fn(
        x, min_sep=2, max_participations=2
    )
    self.assertEqual(val, 1.0)

  def test_negative_values_in_x(self):
    for one_idx in range(4):
      x = -jnp.ones(shape=4).at[one_idx].set(-1)
      for max_participations in [1, 2, None]:
        # A single participation at one_idx is always optimal
        val = sensitivity_lib.max_participation_for_linear_fn(
            x, min_sep=1, max_participations=max_participations
        )
        self.assertEqual(val, 1.0)

  def test_all_negative_x_no_participations(self):
    # This is an edge case, 0 participations will in fact be optimal,
    # with 0 value.
    x = jnp.array([-1, -1, -1])
    for max_participations in [1, 2, None]:
      for min_sep in [1, 2, 3]:
        # A single participation at one_idx is always optimal
        val = sensitivity_lib.max_participation_for_linear_fn(
            x, min_sep=min_sep, max_participations=max_participations
        )
        self.assertEqual(val, 0.0)


class GetSensitivityBandedTest(parameterized.TestCase):

  def test_lower_triangular_mask(self):
    for n in [1, 2, 5]:
      # With b=1, we get the identity matrix
      np.testing.assert_allclose(
          sensitivity_lib.banded_lower_triangular_mask(n, num_bands=1),
          jnp.eye(n),
      )

      # With b=n, we get all 1s below the diagonal.
      np.testing.assert_allclose(
          sensitivity_lib.banded_lower_triangular_mask(n, num_bands=n),
          jnp.tri(n),
      )

    # General case
    np.testing.assert_allclose(
        jnp.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]]),
        sensitivity_lib.banded_lower_triangular_mask(3, num_bands=2),
    )

  @parameterized.named_parameters((f'{n=}', n) for n in [1, 2, 4, 7])
  def test_symmetric_mask_edge_cases(self, n):
    # With b=1, we get the identity matrix
    np.testing.assert_allclose(
        sensitivity_lib.banded_symmetric_mask(n, num_bands=1), np.eye(n)
    )

    # With b=n, we get all 1s.
    np.testing.assert_allclose(
        sensitivity_lib.banded_symmetric_mask(n, num_bands=n),
        jnp.ones(shape=(n, n)),
    )

  def test_symmetric_mask(self):
    # The general case:
    np.testing.assert_allclose(
        np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]]),
        sensitivity_lib.banded_symmetric_mask(3, num_bands=2),
    )

  def test_diag_strategy_matrix(self):
    C = jnp.diag(jnp.array([0.0, 3.0, 0.0, 1.0, 2.0]))

    # 2 participations
    sensitivity = sensitivity_lib.get_sensitivity_banded(C, min_sep=2)
    self.assertEqual(sensitivity, jnp.sqrt(3.0**2 + 2.0**2))

    # 1 participation
    sensitivity = sensitivity_lib.get_sensitivity_banded(
        C, min_sep=2, max_participations=1
    )
    self.assertEqual(sensitivity, 3.0)

  def test_not_orthogonal(self):
    C = jnp.tri(3)
    with self.assertRaisesRegex(ValueError, 'must be orthogonal'):
      sensitivity_lib.get_sensitivity_banded(C, min_sep=2)

  @parameterized.product(n=[1, 2, 5, 10], max_participations=[1, 3, None])
  def test_min_sep_n(self, n, max_participations):
    # With min_sep=n, we should get single-participation, reguardless
    # of max_participations.
    C = jnp.tri(n)
    np.testing.assert_allclose(
        sensitivity_lib.get_sensitivity_banded(
            C, min_sep=n, max_participations=max_participations
        ),
        sensitivity_lib.single_participation_sensitivity(C),
    )

  def test_banded(self):
    # fmt: off, pylint: disable=bad-whitespace
    C = jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, -6.0],
    ])
    # fmt: on, pylint: enable=bad-whitespace

    with self.assertRaisesRegex(ValueError, 'must be orthogonal'):
      sensitivity_lib.get_sensitivity_banded(C, min_sep=1)

    # 1 participation
    sensitivity = sensitivity_lib.get_sensitivity_banded(
        C, min_sep=2, max_participations=1
    )
    self.assertAlmostEqual(sensitivity, 6.0, places=5)

    # Multiple participations
    for min_sep in [2, 3]:
      sensitivity = sensitivity_lib.get_sensitivity_banded(C, min_sep=min_sep)
      self.assertAlmostEqual(sensitivity, np.sqrt(5.0 + 36.0), places=5)


class SensitivityUpperBoundTest(absltest.TestCase):

  def test_matches_banded_sensitivity(self):
    # fmt: off, pylint: disable=bad-whitespace
    C = jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, -6.0],
    ])
    # fmt: on, pylint: enable=bad-whitespace
    # 1 participation
    sensitivity = sensitivity_lib.get_min_sep_sensitivity_upper_bound(
        C, min_sep=2, max_participations=1
    )
    self.assertAlmostEqual(sensitivity, 6.0, places=5)

    # Multiple participations
    for b in [2, 3]:
      sensitivity = sensitivity_lib.get_min_sep_sensitivity_upper_bound(C, b)
      self.assertAlmostEqual(sensitivity, np.sqrt(5.0 + 36.0), places=5)

  def test_dense_X(self):
    # fmt: off, pylint: disable=bad-whitespace
    C = jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 1.0, 0.0, 0.0],
        [0.5, 1.0, 1.0, 0.0],
        [0.4, 0.0, 1.0, -6.0],
    ])
    # fmt: on, pylint: enable=bad-whitespace
    X = C.T @ C
    upper = sensitivity_lib.get_min_sep_sensitivity_upper_bound_for_X(X, 1)
    expected = jnp.sqrt(jnp.abs(X).sum())
    self.assertAlmostEqual(upper, expected, places=5)


class FixedEpochSensitivityTest(parameterized.TestCase):

  def test_one_iter(self):
    ans = sensitivity_lib.fixed_epoch_sensitivity(jnp.eye(1) * 5, epochs=1)
    self.assertAlmostEqual(ans, 5.0)

  @parameterized.named_parameters((f'{i} epochs', i) for i in [1, 2, 4, 8, 16])
  def test_identity(self, epochs):
    ans = sensitivity_lib.fixed_epoch_sensitivity(jnp.eye(16), epochs=epochs)
    self.assertAlmostEqual(ans, jnp.sqrt(epochs))

  @parameterized.named_parameters((f'{i} epochs', i) for i in [-3, 0])
  def test_raises_zero(self, epochs):
    with self.assertRaisesRegex(ValueError, 'Expected epochs='):
      sensitivity_lib.fixed_epoch_sensitivity(jnp.eye(16), epochs=epochs)

  @parameterized.named_parameters((f'{i} epochs', i) for i in [3, 7, 14])
  def test_raises_divisible(self, epochs):
    with self.assertRaisesRegex(ValueError, 'must divide n'):
      sensitivity_lib.fixed_epoch_sensitivity(jnp.eye(16), epochs=epochs)

  def test_single_participation(self):
    n = 6
    C = jax.random.normal(jax.random.PRNGKey(321), (n, n)) * jnp.tri(n)
    ans = sensitivity_lib.fixed_epoch_sensitivity(C, epochs=1)
    self.assertAlmostEqual(ans, jnp.linalg.norm(C, axis=0).max(), places=5)

  def test_full_participation(self):
    n = 16
    C = jnp.abs(jax.random.normal(jax.random.PRNGKey(456), (n, n))) * jnp.tri(n)
    ans = sensitivity_lib.fixed_epoch_sensitivity(C, epochs=n)
    self.assertAlmostEqual(ans, jnp.linalg.norm(C.sum(axis=1)), places=4)


if __name__ == '__main__':
  absltest.main()

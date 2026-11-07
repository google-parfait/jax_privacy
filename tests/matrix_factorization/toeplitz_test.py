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

# Note: When building from source, these tests should use jax with compiler
# optimizations, e.g. `-c opt`. Otherwise runtimes may be excessive and timeouts
# may occur. If `jax` is installed via a pip package, it should already be
# optimized.

from absl.testing import absltest
from absl.testing import parameterized
import dp_accounting
import hypothesis
from hypothesis import strategies as st
import jax
from jax import numpy as jnp
from jax_privacy.matrix_factorization import dense
from jax_privacy.matrix_factorization import sensitivity
from jax_privacy.matrix_factorization import toeplitz
import numpy as np

# Note: Some of these tests fail quite badly without this line:
jax.config.update('jax_enable_x64', True)

hypothesis.settings.register_profile(
    'default', deadline=None, max_examples=10, derandomize=True, database=None
)
hypothesis.settings.load_profile('default')


# Disabling pylint invalid-name to allow mathematical notation including
# single-capital-letter variables for matrices.
# See README.md for notation conventions.
# pylint:disable=invalid-name


# Tuples of (coef, n) to test.
NAMED_C_MATRIX_PARAMS = [
    ('n=1', [7.3], 3),
    ('n=len(coef)', [1, 0.5, 0.2], 3),
    ('n>len(coef)', [1, 0.7, 2.0], 4),
    ('neg_below_main_diag', [1, 0.0, -0.2], 5),
    ('n=25', np.linspace(1, 0.001, num=25), 25),
    ('n=None', np.linspace(0.5, 2.0, num=11), None),
]

# Tuples of coef to test.
NAMED_C_INV_MATRIX_PARAMS = [
    ('n=1', [7.3]),
    ('n=3', [1, -0.5, -0.2]),
    ('n=5', [5.0, -0.7, -2.0, -1.0, -0.3]),
    ('pos_below_main_diag', [1, 0.0, 0.7]),
    ('n=16', np.concatenate([np.ones(1), np.linspace(-0.5, -0.001, num=15)])),
]

# List of Toeplitz coefficients for use in tests
COEFS = [
    [0.1],
    [1.0],
    [0.1, 0.5, 1.0],  # Increasing
    np.linspace(1.0, 0.1, num=16),
    [1.0, 0.0, 0.0, -0.5, 0.0, 0.3],
]


class ToeplitzTest(parameterized.TestCase):

  def test_reconcile(self):
    # We test the `_reconcile` function here, so in general we only need
    # to test the case where n is not None in tests below.

    def tuple_equal(result, expected):
      jax.tree_util.tree_map(np.testing.assert_allclose, result, expected)

    # n is set
    tuple_equal(toeplitz._reconcile([1], 1), (jnp.ones(1), 1))
    tuple_equal(toeplitz._reconcile([1, 2], 1), (jnp.ones(1), 1))
    tuple_equal(toeplitz._reconcile([1, 2], 2), (jnp.array([1, 2]), 2))
    tuple_equal(toeplitz._reconcile([1, 2], 3), (jnp.array([1, 2]), 3))

    # n is None
    tuple_equal(toeplitz._reconcile([1]), (jnp.ones(1), 1))
    tuple_equal(toeplitz._reconcile([1, 2]), (jnp.array([1, 2]), 2))
    tuple_equal(toeplitz._reconcile([1, 2, 3]), (jnp.array([1, 2, 3]), 3))

  def test_materialize_lower_triangular(self):
    np.testing.assert_allclose(
        jnp.array([[1]]), toeplitz.materialize_lower_triangular(jnp.array([1]))
    )
    np.testing.assert_allclose(
        # fmt: off
        jnp.array([[2, 0, 0],
                   [1, 2, 0],
                   [3, 1, 2]]),
        # fmt: on
        toeplitz.materialize_lower_triangular(jnp.array([2, 1, 3])),
    )

  def test_materialize_lower_triangular_n(self):
    # n < len(coef)
    np.testing.assert_allclose(
        jnp.array([[1]]),
        toeplitz.materialize_lower_triangular(coef=[1, 2, 3], n=1),
    )

    # n > len(coef)
    np.testing.assert_allclose(
        # fmt: off
        jnp.array([[2, 0, 0],
                   [1, 2, 0],
                   [0, 1, 2]]),
        # fmt: on
        toeplitz.materialize_lower_triangular(jnp.array([2, 1]), n=3),
    )

  @parameterized.named_parameters((f'{n=}', n) for n in [1, 2, 10, 100])
  def test_fhu_coefs(self, n):
    coef = toeplitz.optimal_max_error_strategy_coefs(n)
    C = toeplitz.materialize_lower_triangular(coef)
    np.testing.assert_allclose(
        np.tri(n),
        C @ C,
        atol=1e-6,
    )

  @parameterized.named_parameters((f'{n=}', n) for n in [1, 2, 10, 100])
  def test_fhu_inv_coefs(self, n):
    coef = toeplitz.optimal_max_error_strategy_coefs(n)
    C = toeplitz.materialize_lower_triangular(coef)
    inv_coef = toeplitz.optimal_max_error_noising_coefs(n)
    C_inv = toeplitz.materialize_lower_triangular(inv_coef)
    np.testing.assert_allclose(
        np.eye(n),
        C @ C_inv,
        atol=1e-6,
    )

  def test_banded_inverse_square_root_noising_coefs(self):
    expected = jnp.array([1, -1 / 2, -1 / 8, -1 / 16, -5 / 128])
    np.testing.assert_allclose(
        toeplitz.banded_inverse_square_root_noising_coefs(5),
        expected,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        toeplitz.banded_inverse_square_root_noising_coefs(
            5, workload_coef=jnp.ones(5)
        ),
        expected,
        atol=1e-6,
    )

  @parameterized.named_parameters((f'{k=}', k) for k in [1, 2, 4, 8, 16])
  def test_banded_inverse_sensitivity_squared_matches_toeplitz(
      self, max_participations
  ):
    num_bands = 8
    n = 128
    min_sep = n // max_participations

    noising_coef = toeplitz.banded_inverse_square_root_noising_coefs(num_bands)
    strategy_coef = toeplitz.inverse_coef(noising_coef, n=n)

    projected_sensitivity_squared = (
        toeplitz.compute_banded_inverse_sensitivity_squared(
            n=n,
            noising_coef=noising_coef,
            min_sep=min_sep,
            max_participations=max_participations,
            use_matrix_upper_bound=False,
        )
    )
    matrix_sensitivity_squared = (
        toeplitz.compute_banded_inverse_sensitivity_squared(
            n=n,
            noising_coef=noising_coef,
            min_sep=min_sep,
            max_participations=max_participations,
            use_matrix_upper_bound=True,
        )
    )
    toeplitz_sensitivity_squared = toeplitz.minsep_sensitivity_squared(
        strategy_coef,
        min_sep=min_sep,
        max_participations=max_participations,
        n=n,
    )

    np.testing.assert_allclose(
        projected_sensitivity_squared,
        toeplitz_sensitivity_squared,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        matrix_sensitivity_squared,
        toeplitz_sensitivity_squared,
        atol=1e-6,
    )

  @parameterized.named_parameters(
      ('full', 16, 16),
      ('basic', 15, 6),
      ('single_band', 16, 1),
      ('single_iter', 1, 1),
  )
  def test_solve_toeplitz(self, n, bands):
    # We set the first coefficient to 1.0 to make sure the matrix is
    # reasonably well conditioned.
    coef = jax.random.normal(jax.random.key(1234), (bands,)).at[0].set(1.0)
    b = jax.random.normal(jax.random.key(5678), (n,))
    x = toeplitz.solve_banded(coef, b)

    C = toeplitz.materialize_lower_triangular(
        jnp.zeros(n).at[: len(coef)].set(coef)
    )
    np.testing.assert_allclose(x, np.linalg.solve(C, b), atol=1e-6)

  @hypothesis.given(
      lhs_coef=st.sampled_from(COEFS),
      rhs_coef=st.sampled_from(COEFS),
      n=st.integers(1, 32),
  )
  def test_multiply(self, lhs_coef, rhs_coef, n):
    L = toeplitz.materialize_lower_triangular(lhs_coef, n)
    R = toeplitz.materialize_lower_triangular(rhs_coef, n)
    expected_coef = (L @ R)[:, 0]
    actual_coef = toeplitz.multiply(lhs_coef, rhs_coef, n)
    actual_coef = toeplitz.pad_coefs_to_n(actual_coef, n)
    np.testing.assert_allclose(actual_coef, expected_coef, atol=1e-6)

  def test_multiply_no_n(self):
    lhs_coef = jnp.array([1, 2, 3, 4])
    rhs_coef = jnp.array([5, 6, 0, 0])
    L = toeplitz.materialize_lower_triangular(lhs_coef)
    R = toeplitz.materialize_lower_triangular(rhs_coef)
    expected_coef = (L @ R)[:, 0]
    actual_coef = toeplitz.multiply(lhs_coef, rhs_coef, n=None)
    np.testing.assert_allclose(actual_coef, expected_coef, atol=1e-6)

  def test_multiply_checks(self):
    lhs_coef = jnp.array([1, 2, 3, 4])
    rhs_coef = jnp.array([5, 6])
    with self.assertRaisesRegex(
        ValueError, 'lhs_coef and rhs_coef must have the same length'
    ):
      toeplitz.multiply(lhs_coef, rhs_coef, n=None)

  @hypothesis.given(n=st.integers(1, 32))
  def test_inverse_is_column_normalized(self, n: int):
    coef = jnp.array([1, 0.8, 0.6, 0.4])
    coef = coef / jnp.linalg.norm(coef[:n])
    C_inv = toeplitz.inverse_as_streaming_matrix(coef, column_normalize_for_n=n)
    C = jnp.linalg.inv(C_inv.materialize(n))
    np.testing.assert_allclose(
        jnp.linalg.norm(C, axis=0),
        jnp.ones(n),
        atol=1e-6,
    )

  @hypothesis.given(n=st.integers(1, 32))
  def test_inverse_is_correct(self, n: int):
    coef = jnp.array([1, 0.8, 0.6, 0.4])
    C_inv = toeplitz.inverse_as_streaming_matrix(coef, None)
    np.testing.assert_allclose(
        jnp.linalg.inv(C_inv.materialize(n)),
        toeplitz.materialize_lower_triangular(coef, n),
        atol=1e-6,
    )

  @parameterized.named_parameters(*[(f'{n=}', n) for n in [1, 2, 5, 10]])
  def test_inverse_toeplitz_coef(self, n):
    coef = jnp.linspace(1.0, 0.0, num=n)
    C = toeplitz.materialize_lower_triangular(coef)
    inv_coef = toeplitz.inverse_coef(coef)
    Cinv = toeplitz.materialize_lower_triangular(inv_coef)
    np.testing.assert_allclose(C @ Cinv, jnp.eye(n), atol=1e-6)

  @parameterized.named_parameters(*[(f'{b=}', b) for b in [1, 2, 5, 10]])
  def test_banded_inverse_toeplitz_coef(self, b):
    n = 2 * b
    coef = jnp.linspace(1.0, 0.0, num=b)
    C = toeplitz.materialize_lower_triangular(coef, n)
    inv_coef = toeplitz.inverse_coef(coef, n)
    Cinv = toeplitz.materialize_lower_triangular(inv_coef, n)
    np.testing.assert_allclose(C @ Cinv, jnp.eye(n), atol=1e-6)

  @parameterized.product(n=[1, 2, 11], coef_idx=[0, 1, 2])
  def test_single_participation_sensitivity(self, n, coef_idx):
    coefs = [[1.0], [1.0, 0.5], jnp.linspace(1.0, 0.1, num=11)]
    coef = coefs[coef_idx]
    np.testing.assert_allclose(
        toeplitz.minsep_sensitivity_squared(
            coef, n=n, min_sep=1, max_participations=1
        ),
        toeplitz.sensitivity_squared(coef, n),
    )

  @parameterized.product(n=[1, 2, 3, 5, 11], k=[1, 4, 11])
  def test_sensitivity_squared_identity(self, n, k):
    actual_max_participations = min(n, k)
    np.testing.assert_allclose(
        toeplitz.minsep_sensitivity_squared(
            strategy_coef=[1.0], n=n, min_sep=1, max_participations=k
        ),
        actual_max_participations,
    )

  @parameterized.product(n=[1, 2, 3, 5, 11], k=[1, 4, 11])
  def test_sensitivity_squared_tri(self, n, k):
    C = np.tri(n)
    coef = np.ones(n)
    np.testing.assert_allclose(
        toeplitz.minsep_sensitivity_squared(
            coef, n=n, min_sep=1, max_participations=k
        ),
        # sensitivity**2 is the sum of the L2-norm-squared of the
        # first k columns of C:
        np.sum(np.sum(C[:, :k], axis=1) ** 2),
    )

  @parameterized.product(
      n_and_epochs=[
          (1, 1),
          (2, 1),
          (2, 2),
          (10, 1),
          (10, 5),
          (10, 10),
      ],
      bands=[2, 13, 10, 11],
  )
  def test_sensitivity_squared_fixed_epoch(self, n_and_epochs, bands):
    n, epochs = n_and_epochs
    # For a Toeplitz matrix with positive decreasing entries, the worst
    # case is a fixed-epoch participation pattern. Note we must have
    # n % epochs == 0 here.
    coef = jnp.linspace(1.0, 0.001, num=bands)
    C = toeplitz.materialize_lower_triangular(coef, n=n)
    expected = sensitivity.fixed_epoch_sensitivity(C, epochs) ** 2
    min_sep = n // epochs

    # We should get the same result if we set a higher more max_participations,
    # or infer it from n and min_sep:
    for max_particpations in [epochs, epochs + 10, None]:
      np.testing.assert_allclose(
          toeplitz.minsep_sensitivity_squared(
              coef,
              min_sep=min_sep,
              max_participations=max_particpations,
              n=n,
          ),
          expected,
          atol=1e-6,
      )

  @parameterized.product(
      n=[1, 5, 50],
      max_participations=[1, 2, 5, 50],
  )
  def test_sensitivity_squared_min_sep_one(self, n, max_participations):
    coef = jnp.sqrt(jnp.linspace(1.0, 0.001, num=n))
    C = toeplitz.materialize_lower_triangular(coef)
    np.testing.assert_allclose(
        toeplitz.minsep_sensitivity_squared(
            coef, min_sep=1, n=n, max_participations=max_participations
        ),
        np.linalg.norm(np.sum(C[:, :max_participations], axis=1)) ** 2,
        rtol=1e-10,
    )

  def test_sensitivity_squared_bad_min_sep(self):
    with self.assertRaisesRegex(ValueError, 'Expected min_sep='):
      toeplitz.minsep_sensitivity_squared(strategy_coef=[1.0], min_sep=0, n=1)

  @parameterized.product(
      coef=[[-1.0], [1.0, 2.0, 1.0], [1.0, -0.5, 0.3], [-1, 0.5]],
      min_sep=[1, 2],
      max_part=[1, 2, None],
  )
  def test_sensitivity_squared_upper_bound(self, coef, min_sep, max_part):
    """When coefs violate assumptions, the result is an upper bound."""
    coef = jnp.array(coef)
    n = 6
    C = toeplitz.materialize_lower_triangular(coef, n=n)
    true_sens_sq = (
        sensitivity.get_min_sep_sensitivity_upper_bound(
            C,
            min_sep=min_sep,
            max_participations=max_part,
        )
        ** 2
    )
    toeplitz_sens_sq = toeplitz.minsep_sensitivity_squared(
        coef,
        min_sep=min_sep,
        max_participations=max_part,
        n=n,
    )
    self.assertGreaterEqual(float(toeplitz_sens_sq), float(true_sens_sq) - 1e-6)

  def test_sensitivity_squared_projection_is_majorant(self):
    """Regression test: the projection must be a non-increasing majorant.

    With coef = [1.0, 0.5, 0.8], the correct projection via
    cummax(abs(coef)[::-1])[::-1] gives [1.0, 0.8, 0.8] (a majorant), ensuring
    the result is an upper bound on true sensitivity. An incorrect projection
    via cummin(abs(coef)) would give [1.0, 0.5, 0.5] (a minorant), which
    underestimates the true sensitivity.
    """
    coef = jnp.array([1.0, 0.5, 0.8])
    n = 6
    C = toeplitz.materialize_lower_triangular(coef, n=n)
    true_sens_sq = (
        sensitivity.get_min_sep_sensitivity_upper_bound(
            C,
            min_sep=1,
            max_participations=2,
        )
        ** 2
    )
    toeplitz_sens_sq = toeplitz.minsep_sensitivity_squared(
        coef,
        min_sep=1,
        max_participations=2,
        n=n,
    )
    # With the correct majorant projection, this is an upper bound.
    self.assertGreaterEqual(float(toeplitz_sens_sq), float(true_sens_sq) - 1e-6)
    # The correct projection strictly increases sensitivity relative to cummin.
    # This ensures the test fails if cummax(abs[::-1])[::-1] is replaced with
    # cummin(abs).
    wrong_projection = jax.lax.cummin(jnp.abs(coef))
    correct_projection = jax.lax.cummax(jnp.abs(coef)[::-1])[::-1]
    self.assertFalse(jnp.allclose(wrong_projection, correct_projection))


def _sensitivity_squared(coef: jnp.ndarray, n: int | None) -> jnp.ndarray:
  """Single-participation sensitivity."""
  coef, _ = toeplitz._reconcile(coef, n)
  return toeplitz._l2_norm_squared(coef)


# TODO: b/329444015 - Replace these helpers with calls do dense.py
# when error computations are available there.


def _per_query_error(
    coef: jnp.ndarray, n: int | None, workload_coef: jax.Array | None = None
) -> jax.Array:
  coef, n = toeplitz._reconcile(coef, n)
  if workload_coef is None:
    workload_coef = jnp.ones(n)
  workload_coef = toeplitz.pad_coefs_to_n(workload_coef, n)
  C = toeplitz.materialize_lower_triangular(coef, n=n)
  A = toeplitz.materialize_lower_triangular(workload_coef, n=n)
  return dense.per_query_error(strategy_matrix=C, workload_matrix=A)


class ToeplitzErrorTest(parameterized.TestCase):

  @hypothesis.given(
      name_coef_n_tuple=st.sampled_from(NAMED_C_MATRIX_PARAMS),
      workload=st.sampled_from(
          ['default', 'prefix_sum', 'eye', 'banded', 'extra_entries']
      ),
      config=st.sampled_from(['jit', 'eager']),
  )
  def test_per_query_error(self, name_coef_n_tuple, workload, config):
    _, coef, n = name_coef_n_tuple
    _, true_n = toeplitz._reconcile(coef, n)
    if workload == 'default':
      workload_coef = None
    elif workload == 'prefix_sum':
      workload_coef = jnp.ones(true_n)
    elif workload == 'eye':
      workload_coef = jnp.ones(1)
    elif workload == 'banded':
      workload_coef = jnp.array([1, 0.9, 0.5, 0.1])
    elif workload == 'extra_entries':
      workload_coef = jnp.linspace(0.99, 0.1, num=true_n + 5)
    else:
      raise ValueError(f'Unknown workload: {workload}')

    per_query_error = toeplitz.per_query_error
    if config == 'jit':
      per_query_error = jax.jit(per_query_error, static_argnames=['n'])

    inv_coef = toeplitz.inverse_coef(coef, n)
    expected = _per_query_error(coef, n, workload_coef=workload_coef)
    kwargs = {
        'workload_coef': workload_coef,
        'n': n,
    }
    np.testing.assert_allclose(
        per_query_error(strategy_coef=coef, **kwargs),
        expected,
        err_msg='Failure computing error for strategy_coef',
    )
    np.testing.assert_allclose(
        per_query_error(noising_coef=inv_coef, **kwargs),
        expected,
        err_msg='Failure computing error for noising_coef',
    )

  @hypothesis.given(
      name_coef_n_tuple=st.sampled_from(NAMED_C_MATRIX_PARAMS),
      workload=st.sampled_from(
          ['default', 'prefix_sum', 'eye', 'banded', 'extra_entries']
      ),
  )
  def test_max_error_is_last_iterate(self, name_coef_n_tuple, workload):
    _, coef, n = name_coef_n_tuple
    _, true_n = toeplitz._reconcile(coef, n)
    if workload == 'default':
      workload_coef = None
    elif workload == 'prefix_sum':
      workload_coef = jnp.ones(true_n)
    elif workload == 'eye':
      workload_coef = jnp.ones(1)  # Rest implicitly 0
    elif workload == 'banded':
      workload_coef = jnp.array([1, 0.9, 0.5, 0.1])
    elif workload == 'extra_entries':
      workload_coef = jnp.linspace(0.99, 0.1, num=true_n + 5)
    else:
      raise ValueError(f'Unknown workload: {workload}')

    err = toeplitz.per_query_error(
        strategy_coef=jnp.array(coef), n=n, workload_coef=workload_coef
    )
    np.testing.assert_allclose(jnp.max(err), err[-1])

  @hypothesis.given(
      n=st.integers(1, 8),
  )
  def test_mean_loss(self, n):
    coef = jnp.array([1.0, 0.4, 0.3, 0.1])
    loss = toeplitz.loss(strategy_coef=coef, n=n)

    expected = _sensitivity_squared(coef, n) * jnp.mean(
        _per_query_error(coef, n)
    )
    np.testing.assert_allclose(loss, expected)

    # Confirm loss is scale invariant
    np.testing.assert_allclose(
        loss, toeplitz.loss(strategy_coef=2.3 * coef, n=n)
    )


class ToeplitzOptimizationTest(parameterized.TestCase):

  @parameterized.named_parameters((f'{b}_bands', b) for b in [1, 5, 10, 15])
  def test_toeplitz_more_bands_is_better(self, b):
    coef1 = toeplitz.optimize_banded_toeplitz(16, b, max_optimizer_steps=25)
    coef2 = toeplitz.optimize_banded_toeplitz(16, b + 1, max_optimizer_steps=25)
    self.assertLessEqual(
        toeplitz.loss(strategy_coef=coef2, n=16),
        toeplitz.loss(strategy_coef=coef1, n=16),
    )

  @parameterized.product(
      max_participations=[1, 2, 4, 8, 16],
      momentum=[None, 0.0, 0.9],
  )
  def test_optimize_banded_inverse_toeplitz_improves_over_bisr(
      self, max_participations, momentum
  ):
    n = 128
    min_sep = n // max_participations
    num_bands = 8
    if momentum is None:
      workload_coef = None
    else:
      decay = jnp.asarray(momentum ** np.arange(n))
      workload_coef = toeplitz.multiply(jnp.ones(n), decay, n=n)

    optimized_coef = toeplitz.optimize_banded_inverse_toeplitz(
        n=n,
        min_sep=min_sep,
        num_bands=num_bands,
        max_participations=max_participations,
        workload_coef=workload_coef,
        max_optimizer_steps=30,
        reduction_fn=jnp.mean,
    )
    bisr_coef = toeplitz.banded_inverse_square_root_noising_coefs(num_bands)

    self.assertTrue(jnp.all(jnp.isfinite(optimized_coef)))
    np.testing.assert_allclose(optimized_coef[0], 1.0, atol=1e-6)

    optimized_error = jnp.mean(
        toeplitz.per_query_error(
            noising_coef=optimized_coef,
            n=n,
            workload_coef=workload_coef,
        )
    )
    optimized_sensitivity_squared = (
        toeplitz.compute_banded_inverse_sensitivity_squared(
            n=n,
            noising_coef=optimized_coef,
            min_sep=min_sep,
            max_participations=max_participations,
        )
    )
    optimized_loss = optimized_error * optimized_sensitivity_squared

    bisr_error = jnp.mean(
        toeplitz.per_query_error(
            noising_coef=bisr_coef,
            n=n,
            workload_coef=workload_coef,
        )
    )
    bisr_sensitivity_squared = (
        toeplitz.compute_banded_inverse_sensitivity_squared(
            n=n,
            noising_coef=bisr_coef,
            min_sep=min_sep,
            max_participations=max_participations,
        )
    )
    bisr_loss = bisr_error * bisr_sensitivity_squared

    self.assertLessEqual(optimized_loss, bisr_loss)


class ToeplitzAmplificationTest(parameterized.TestCase):

  def test_optimize_banded_toeplitz_for_amplifications(self):
    dataset_size = 1024
    batch_size = 16
    coef, stddev = toeplitz.optimize_coefs_for_amplifications(
        n=100,
        dataset_size=dataset_size,
        expected_batch_size=batch_size,
        epsilon=2.0,
        delta=1e-6,
    )
    bands = len(coef)
    self.assertGreater(stddev, 0)
    self.assertLessEqual(bands, dataset_size // batch_size)

  def test_large_epsilon(self):
    # Note: This step requires multiple steps of optimization
    # and noise calibration, so it can take a while.

    # With a large epsilon, correlated noise is more effective than
    # amplification, so we expect the maximum number of bands is optimal.
    dataset_size = 1000
    batch_size = 100
    max_bands = dataset_size // batch_size
    coef, stddev = toeplitz.optimize_coefs_for_amplifications(
        n=100,
        dataset_size=dataset_size,
        expected_batch_size=batch_size,
        epsilon=20.0,
        delta=1e-6,
    )
    self.assertGreater(stddev, 0)
    bands = len(coef)
    self.assertEqual(bands, max_bands)

  def test_small_epsilon(self):
    # Note: This step requires multiple steps of optimization
    # and noise calibration, so it can take a while.

    # With a small epsilon, amplification is more effective than
    # correlated noise, so we expect a single band is optimal.
    dataset_size = 1000
    batch_size = 100
    coef, stddev = toeplitz.optimize_coefs_for_amplifications(
        n=100,
        dataset_size=dataset_size,
        expected_batch_size=batch_size,
        epsilon=0.1,
        delta=1e-6,
    )
    self.assertGreater(stddev, 0)
    bands = len(coef)
    self.assertEqual(bands, 1)

  def test_helper(self):
    n = 100
    dataset_size = 100
    batch_size = 10
    epsilon = 2.0
    delta = 1e-6
    helper = toeplitz._AmplifiedBandMFHelper(
        n=n,
        dataset_size=dataset_size,
        batch_size=batch_size,
        epsilon=epsilon,
        delta=delta,
    )
    max_bands = 10
    self.assertEqual(helper.max_bands, max_bands)
    self.assertEqual(helper.sensitivity(coef=jnp.array([1.0])), 1.0)
    self.assertEqual(
        helper.sensitivity(coef=jnp.array([1.0, 1.0])), jnp.sqrt(2)
    )

  def test_helper_stddev_at_max_bands(self):
    n = 100
    dataset_size = 100
    batch_size = 10
    epsilon = 2.0
    delta = 1e-6
    helper = toeplitz._AmplifiedBandMFHelper(
        n=n,
        dataset_size=dataset_size,
        batch_size=batch_size,
        epsilon=epsilon,
        delta=delta,
    )
    max_bands = 10
    assert max_bands == dataset_size // batch_size
    # If bands == max_bands, then there is no amplification, and
    # we should get the same standard deviation as the non-amplified case
    # assuming bands-min-sep participation.
    k = n // max_bands
    assert k * max_bands == n
    coef = toeplitz.optimal_max_error_strategy_coefs(max_bands)

    b_min_sep_sens = jnp.sqrt(
        toeplitz.minsep_sensitivity_squared(
            coef, min_sep=max_bands, max_participations=k, n=n
        )
    )

    total_nm = dp_accounting.calibrate_dp_mechanism(
        make_fresh_accountant=dp_accounting.rdp.RdpAccountant,
        make_event_from_param=lambda nm: dp_accounting.GaussianDpEvent(
            noise_multiplier=nm
        ),
        target_epsilon=epsilon,
        target_delta=delta,
        bracket_interval=dp_accounting.ExplicitBracketInterval(0.01, 10.0),
    )
    expected_stddev = total_nm * b_min_sep_sens

    helper_stddev = helper.required_stddev(coef=coef)
    np.testing.assert_allclose(
        helper_stddev, expected_stddev, rtol=1e-4, atol=1e-5
    )


if __name__ == '__main__':
  absltest.main()

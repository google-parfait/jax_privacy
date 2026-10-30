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


import functools
import os
import sys

from absl.testing import absltest
from absl.testing import parameterized
import hypothesis
from hypothesis import strategies as st
import jax
import jax.numpy as jnp
from jax_privacy.matrix_factorization import buffered_toeplitz
from jax_privacy.matrix_factorization import sensitivity
from jax_privacy.matrix_factorization import toeplitz
import numpy as np


hypothesis.settings.register_profile(
    'dpftrl_default',
    database=None,
    deadline=None,  # 200ms disable
    derandomize=True,
    max_examples=10,
    verbosity=hypothesis.Verbosity.normal,
)

profile = os.getenv('HYPOTHESIS_PROFILE', default='dpftrl_default')
hypothesis.settings.load_profile(profile)
print(
    f'Using hypothesis profile "{profile}":\n{hypothesis.settings.default}',
    file=sys.stderr,
    flush=True,
)


def scale_max_examples(scale: float) -> int:
  """Scales the current max_examples."""
  return max(1, int(round(scale * hypothesis.settings.default.max_examples)))


# Many BLT operations benefit from x64 precision, so we enable it for tests.
jax.config.update('jax_enable_x64', True)

# Disabling pylint invalid-name to allow mathematical notation including
# single-capital-letter variables for matrices.
# See README.md for notation conventions.
# pylint:disable=invalid-name


def assert_allclose(*args, atol=1e-7, **kwargs):
  # Wrapper around np.testing.assert_allclose with default atol.
  np.testing.assert_allclose(*args, atol=atol, **kwargs)


def make_blt(
    buf_decay: list[float], output_scale: list[float]
) -> buffered_toeplitz.BufferedToeplitz:
  return buffered_toeplitz.BufferedToeplitz.build(
      buf_decay=buf_decay, output_scale=output_scale
  )


# We use tuples of floats, to avoid constructing jnp.ndarrays in
# the default arguments which fails, see b/140864213.
# We construct a wide range of examples here, and use
# hypothesis.assume based on the assumptions of the different routines.
BLT_TUPLES = (
    [
        # buf_decay (theta), output_scale (omega)
        ([0.99], [0.1]),
        ([0.9, 0.5, 0.001], [0.1, 0.2, 0.5]),
        ([0.001], [0.5]),
        ([0.001], [-0.1]),  # Negative output_scale, usually a C^{-1}
        ([0.99, -0.1], [0.01, -0.1]),
        ([0.99, -0.1], [-0.2, 0.1]),
        # Some codepaths now support buf_decay = 1:
        ([1.0], [0.001]),
        ([1.0, 1 - 1e-9, 0.99, 0.1], [1e-6, 1e-5, 1e-3, 1e-2]),
        # An early "good" for n=100k C parameterization
        # with some unusal parameters (a negative buffer decay,
        # and two negative output scales):
        (
            [0.99994829, 0.99591045, 0.76319372, -0.3667193, 0.69954866],
            [0.01202588, 0.08567775, 0.66450507, -0.00384594, -0.25671158],
        ),
        # Optimization for n=100k before numerical issues were fixed (not
        # sure if this mattered).  But some theta's very near 1:
        (
            [0.99999999, 0.77422499, 0.54944999, 0.32467499, 0.09989999],
            [0.00999262, 0.00999992, 0.00999996, 0.00999997, 0.00999998],
        ),
    ]
    + [  # Small theta gaps
        ([0.9, 0.9 - 10 ** (-p)], [0.1, 0.1]) for p in range(1, 12)
    ]
    + [  # Thetas near one:
        ([1 - 10 ** (-p), 0.8], [0.1, 0.1]) for p in range(1, 12)
    ]
)


def direct_max_error_for_inv(inv_blt, n):
  return toeplitz.per_query_error(noising_coef=inv_blt.toeplitz_coefs(n)).max()


def brute_force_mse_for_inv(inv_blt, n):
  return toeplitz.per_query_error(noising_coef=inv_blt.toeplitz_coefs(n)).mean()


def brute_force_max_loss(blt, n, min_sep=1, max_participations=None):
  coef = blt.toeplitz_coefs(n)
  sens_squared = toeplitz.minsep_sensitivity_squared(
      coef,
      min_sep=min_sep,
      n=n,
      max_participations=max_participations,
  )
  error = jnp.max(toeplitz.per_query_error(strategy_coef=coef))
  return sens_squared * error


# TODO: b/303715739 - Implement better auto-generation of BLT
# test cases, using hypothesis routines or otherwise. For example,
# this is a reasonable recipe for generating C:
#   buf_decay = np.random.uniform(0, 1, size=nbuf)
#   # Scaling output_scale but 1/nbuf avoids C with large condition numbers
#   output_scale = np.random.uniform(0, 1, size=nbuf) / nbuf


class BufferedToeplitzTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('x64_enabled', True),
      ('x64_disabled', False),
  )
  def test_float32_dtype(self, x64_enabled):
    with jax.enable_x64(x64_enabled):
      blt = buffered_toeplitz.BufferedToeplitz.build(
          buf_decay=[0.9], output_scale=[0.1], dtype=jnp.float32
      )
      self.assertEqual(blt.buf_decay.dtype, jnp.float32)
      self.assertEqual(blt.output_scale.dtype, jnp.float32)
      self.assertEqual(blt.dtype, jnp.float32)

  def test_float64_without_x64_raises(self):
    with self.assertRaisesRegex(ValueError, 'dtype=jnp.float64'):
      with jax.enable_x64(False):
        buffered_toeplitz.BufferedToeplitz.build(
            buf_decay=[0.9], output_scale=[0.1]
        )

  def test_validate_params_are_vectors(self):
    with self.assertRaisesRegex(
        ValueError, 'buf_decay and output_scale must be.*1D'
    ):
      buffered_toeplitz.BufferedToeplitz.build(
          buf_decay=[[0.9]], output_scale=[0.1]
      )

  def test_validate_param_lengths_match(self):
    with self.assertRaisesRegex(
        ValueError, 'buf_decay and output_scale must have the same length'
    ):
      buffered_toeplitz.BufferedToeplitz.build(
          buf_decay=[0.9, 0.8], output_scale=[0.1]
      )

  def test_validate_param_dtypes_match(self):
    with self.assertRaisesRegex(
        ValueError, 'buf_decay and output_scale must have the same dtype'
    ):
      # We can't use .build(...) because that sets the dtype.
      buffered_toeplitz.BufferedToeplitz(
          buf_decay=jnp.array([0.9], dtype=jnp.float64),
          output_scale=jnp.array([0.1], dtype=jnp.float32),
      ).validate()

  @parameterized.product(n=[1, 2, 10, 11], k=[1, 2, 3])
  def test_coefs(self, n, k):
    buf_decay = jnp.array([1.0, 0.9, 0])
    output_scale = jnp.array([0.01, 0.1, 1])
    blt = buffered_toeplitz.BufferedToeplitz.build(
        buf_decay=buf_decay[:k], output_scale=output_scale[:k]
    )
    coefs1 = blt.toeplitz_coefs(n)
    coefs2 = blt.materialize(n)[:, 0]
    assert_allclose(coefs1, coefs2, atol=1e-9)

  @parameterized.named_parameters((f'{n=}', n) for n in [1, 2, 11])
  def test_zero_buffer_identity_blt(self, n):
    # The 0-buffer BLT is the identity mattrix:
    blt = buffered_toeplitz.BufferedToeplitz.build(
        buf_decay=[], output_scale=[]
    )
    I = jnp.eye(n)
    assert_allclose(blt.materialize(n), I)
    assert_allclose(blt.as_streaming_matrix().materialize(n), I)
    assert_allclose(blt.inverse_as_streaming_matrix().materialize(n), I)
    assert_allclose(
        (blt.as_streaming_matrix() @ blt.inverse_as_streaming_matrix()) @ I, I
    )
    assert_allclose(
        (blt.inverse_as_streaming_matrix() @ blt.as_streaming_matrix()) @ I, I
    )

  @parameterized.named_parameters((f'{n=}', n) for n in [1, 2, 11])
  def test_inverse_and_lt_multiply(self, n):
    blt = buffered_toeplitz.BufferedToeplitz.build(
        buf_decay=jnp.array([0.1, 0.2, 0.3]),
        output_scale=jnp.array([0.1, 0.2, 0.3]),
    )

    C = blt.as_streaming_matrix()
    C_inv = blt.inverse_as_streaming_matrix()

    A = jnp.tri(n)
    assert_allclose(A, C_inv @ (C @ A))
    assert_allclose(A, (C_inv @ C) @ A)
    assert_allclose(A, (C @ C_inv) @ A)
    assert_allclose(A, C @ (C_inv @ A))

  def test_build_and_str(self):
    blt = buffered_toeplitz.BufferedToeplitz.build(
        buf_decay=[0.5, 0.9], output_scale=[0.6, 0.1]
    )
    # Also test the blt is in canonical order via .canonicalize()
    assert_allclose(blt.buf_decay, [0.9, 0.5])
    assert_allclose(blt.output_scale, [0.1, 0.6])
    self.assertEqual(blt.buf_decay.dtype, jnp.float64)
    self.assertEqual(blt.output_scale.dtype, jnp.float64)

    self.assertIn('Initial Toeplitz coefs=[1', str(blt))
    self.assertIn('buf_decay=[0.9', str(blt))

  @hypothesis.given(n=st.integers(1, 20))
  def test_materialize(self, n: int):
    blt = buffered_toeplitz.BufferedToeplitz.build(
        buf_decay=[0.0], output_scale=[-1.0]
    )
    expected_coefs = jnp.zeros(n).at[:2].set([1, -1][:n])
    assert_allclose(blt.toeplitz_coefs(n), expected_coefs)
    assert_allclose(
        blt.materialize(n),
        toeplitz.materialize_lower_triangular(expected_coefs),
    )

  @parameterized.named_parameters((f'{d=}', d) for d in range(1, 11))
  def test_from_rational_approx_to_sqrt_x(self, num_buffers: int):
    n = 10  # We compare to the first `n` optimal Toeplitz coefficients.
    # Relative tolerance expected as indexed by num_buffers - 1. More buffers
    # leads to a better approximation of the optimal coefficients.
    rtols = [1.10, 0.41, 0.25, 0.25, 0.15, 0.15, 0.09, 0.09, 0.06, 0.06]

    blt = buffered_toeplitz.BufferedToeplitz.from_rational_approx_to_sqrt_x(
        num_buffers=num_buffers, max_buf_decay=1.0
    )
    opt_toeplitz_strategy_coefs = toeplitz.optimal_max_error_strategy_coefs(n=n)
    blt_coefs = blt.toeplitz_coefs(n)
    np.testing.assert_allclose(
        blt_coefs, opt_toeplitz_strategy_coefs, rtol=rtols[num_buffers - 1]
    )

  @hypothesis.given(
      buffers=st.integers(1, 10),
      max_buf_decay=st.floats(0.1, 1.0),
      max_pillutla_score=st.floats(0.1, 1.0),
  )
  def test_from_rational_approx_to_sqrt_x_projection(
      self, buffers: int, max_buf_decay: float, max_pillutla_score: float
  ):
    blt = buffered_toeplitz.BufferedToeplitz.from_rational_approx_to_sqrt_x(
        num_buffers=buffers,
        max_buf_decay=max_buf_decay,
        max_pillutla_score=max_pillutla_score,
    )
    self.assertLessEqual(blt.buf_decay[0], max_buf_decay + 1e-10)
    self.assertLessEqual(blt.pillutla_score(), max_pillutla_score + 1e-10)

  @parameterized.parameters(-1, 0)
  def test_from_rational_approx_to_sqrt_x_projection_raises(self, nbuf: int):
    with self.assertRaisesRegex(ValueError, 'num_buffers must be >= 1'):
      buffered_toeplitz.BufferedToeplitz.from_rational_approx_to_sqrt_x(
          num_buffers=nbuf
      )

  @hypothesis.given(
      n=st.integers(1, 20),
      buffers=st.integers(1, 5),
  )
  def test_inverse_materialized_and_streaming(self, n: int, buffers: int):

    blt = buffered_toeplitz.BufferedToeplitz.build(
        buf_decay=jnp.linspace(1.0, 0.0, num=buffers),
        # Dividing by n improves conditionoing.
        output_scale=jnp.linspace(1e-4, 1.0, num=buffers) / n,
    )
    C = blt.materialize(n)
    I = jnp.eye(n)
    assert_allclose(
        C,
        toeplitz.materialize_lower_triangular(blt.toeplitz_coefs(n)),
        atol=1e-5,
    )

    Cinv = blt.inverse_as_streaming_matrix().materialize(n)
    # Larger n and buffers can produce poorly-conditioned matrices.
    assert_allclose(C @ Cinv, I, atol=1e-5)

    # Implicit multiplication
    assert_allclose(
        (blt.as_streaming_matrix() @ blt.inverse_as_streaming_matrix()) @ I, I
    )


def _max_loss_for_identity_strategy(
    n: int, min_sep: int, max_participations: int | None = None
):
  # Sensitivity squared is just the true max participations
  sens_squared = sensitivity.minsep_true_max_participations(
      n=n, min_sep=min_sep, max_participations=max_participations
  )
  # Max error is just n (from the last iteration)
  error = n
  return sens_squared * error


def float64_blt():
  with jax.enable_x64():
    return buffered_toeplitz.BufferedToeplitz.build(
        buf_decay=[0.9], output_scale=[0.1], dtype=jnp.float64
    )


def float32_blt():
  return buffered_toeplitz.BufferedToeplitz.build(
      buf_decay=[0.0], output_scale=[-1.0], dtype=jnp.float32
  )


class BufferedToeplitzFloat32Test(parameterized.TestCase):

  @parameterized.named_parameters(
      ('float32_blt', float32_blt),
      ('float64_blt', float64_blt),
  )
  def test_materialize_float32(self, blt_fn):
    blt = blt_fn()
    with jax.enable_x64(False):
      # Materialize of a float32 or float64 BLT without x64 enabled should
      # not fail, and produce a float32 result.
      C = blt.materialize(1)
      assert_allclose(C, jnp.array([[1]], dtype=jnp.float32))
      self.assertEqual(C.dtype, jnp.float32)

  def test_inverse_float64_without_x64_raises(self):
    with self.assertRaisesRegex(ValueError, 'dtype=jnp.float64'):
      with jax.enable_x64(False):
        float64_blt().inverse()

  def test_inverse_float32_without_x64(self):
    with jax.enable_x64(False):
      inv_blt = float32_blt().inverse()
      self.assertEqual(inv_blt.dtype, jnp.float32)

  def test_iteration_error_raises(self):
    with self.assertRaisesRegex(ValueError, 'dtype=jnp.float64'):
      with jax.enable_x64(False):
        # Note that float64_blt() is really a strategy matrix,
        # not a noising_matrix, but it doesn't matter for this test.
        buffered_toeplitz.iteration_error(float64_blt(), i=3)

  def test_max_loss_raises(self):
    with self.assertRaisesRegex(ValueError, 'dtype=jnp.float64'):
      with jax.enable_x64(False):
        buffered_toeplitz.max_loss(float64_blt(), n=3)

  def test_limit_max_loss_raises(self):
    with self.assertRaisesRegex(ValueError, 'dtype=jnp.float64'):
      with jax.enable_x64(False):
        buffered_toeplitz.limit_max_loss(float64_blt())


class OptimizationTest(parameterized.TestCase):
  """Tests based on optimizing LossFn.build_min_sep."""

  @hypothesis.given(
      num_buffers=st.integers(1, 4),
      parameterization=st.sampled_from([
          buffered_toeplitz.Parameterization.buf_decay_pair(),
          buffered_toeplitz.Parameterization.strategy_blt(),
      ]),
  )
  def test_parameterization(self, num_buffers, parameterization):
    blt = buffered_toeplitz.get_init_blt(num_buffers=num_buffers)
    params = parameterization.params_from_blt(blt)
    blt2, _ = parameterization.blt_and_inverse_from_params(params)

    assert_allclose(blt.buf_decay, blt2.buf_decay)
    assert_allclose(blt.output_scale, blt2.output_scale)

  @parameterized.named_parameters(
      (
          'max_error_min_sep',
          'max',
          functools.partial(
              buffered_toeplitz.LossFn.build_min_sep,
              min_sep=2,
              max_participations=2,
          ),
      ),
      (
          'mean_error_min_sep',
          'mean',
          functools.partial(
              buffered_toeplitz.LossFn.build_min_sep,
              min_sep=2,
              max_participations=2,
          ),
      ),
  )
  def test_loss_fn_and_init(self, error, constructor):
    n = 10
    blt = buffered_toeplitz.get_init_blt(num_buffers=3)
    inv_blt = blt.inverse()

    # Default penalty strengths
    loss_fn = constructor(n=n, error=error)
    loss = loss_fn.loss(blt)
    penalized_loss = loss_fn.penalized_loss(
        blt, inv_blt, normalize_by_approx_optimal_loss=False
    )
    self.assertGreater(loss, 0)
    self.assertGreater(penalized_loss, loss)
    self.assertLess(
        loss_fn.penalized_loss(
            blt, inv_blt, normalize_by_approx_optimal_loss=True
        ),
        penalized_loss,
    )

    # A penalty multiplier < 1 should decrease the penalized loss.
    loss_fn = constructor(
        n=n, error=error, penalty_multipliers={'output_scale>0': 0.01}
    )
    penalized_loss2 = loss_fn.penalized_loss(
        blt, inv_blt, normalize_by_approx_optimal_loss=False
    )
    self.assertLess(penalized_loss2, penalized_loss)

    # penalty_strength=0.0 should make penalized_loss == loss
    loss_fn = constructor(n=n, error=error, penalty_strength=0.0)
    loss = loss_fn.loss(blt)
    penalized_loss = loss_fn.penalized_loss(
        blt, inv_blt, normalize_by_approx_optimal_loss=False
    )
    np.testing.assert_allclose(penalized_loss, loss)

  def test_loss_fn_and_init_closed_form_single_participation(self):
    n = 10
    loss_fn = buffered_toeplitz.LossFn.build_closed_form_single_participation(
        n=n
    )
    blt = buffered_toeplitz.get_init_blt(num_buffers=3)
    self.assertGreater(loss_fn.loss(blt), 0)
    self.assertGreater(loss_fn.penalized_loss(blt, blt.inverse()), 0)

  def test_loss_fn_bad_strength(self):
    loss_fn = buffered_toeplitz.LossFn.build_closed_form_single_participation(
        n=10,
        penalty_multipliers={'bad_penalty': 0.0},
    )
    blt = buffered_toeplitz.get_init_blt(num_buffers=3)
    with self.assertRaisesRegex(
        ValueError, 'Unrecognized penalty multipliers.*bad'
    ):
      loss_fn.penalized_loss(blt, blt.inverse())

  @parameterized.product(
      n=[1, 2, 11, 100],
      min_sep=[1, 2, 10, 11],
      max_participations=[1, 2, None],
  )
  def test_min_sep_loss(self, n, min_sep, max_participations):
    loss = buffered_toeplitz.LossFn.build_min_sep(
        n=n, min_sep=min_sep, max_participations=max_participations, error='max'
    )
    blt = buffered_toeplitz.get_init_blt(num_buffers=2)
    assert_allclose(
        loss.loss(blt),
        brute_force_max_loss(
            blt, n=n, min_sep=min_sep, max_participations=max_participations
        ),
    )

  def test_optimize_increasing_nbuf(self):
    def opt_fn(nbuf):
      if nbuf == 0:
        return jnp.zeros(nbuf), 10.0
      elif nbuf == 1:
        return jnp.zeros(nbuf), 9.0
      elif nbuf == 2:
        return jnp.zeros(nbuf), 8.0
      else:
        raise RuntimeError('Optimization failed!!')

    params = buffered_toeplitz._optimize_increasing_nbuf(
        opt_fn, min_buffers=0, max_buffers=5
    )
    np.testing.assert_array_equal(params, jnp.zeros(2))

  def test_optimize_increasing_nbuf_not_implemented_error(self):
    def opt_fn(nbuf):
      if nbuf == 0:
        return jnp.zeros(nbuf), 10.0
      elif nbuf == 1:
        return jnp.zeros(nbuf), 9.0
      else:
        raise NotImplementedError('Not implemented.')

    with self.assertRaisesRegex(NotImplementedError, 'Not implemented.'):
      buffered_toeplitz._optimize_increasing_nbuf(
          opt_fn, min_buffers=0, max_buffers=5
      )

  @parameterized.named_parameters(
      ('n=1,b=1,k=1', 1, 1, 1),
      ('n=10,b=1,k=3', 10, 1, 3),
      ('n=32,b=8,k=None', 32, 8, None),
      ('n=32,b=2,k=16', 32, 2, 16),
      ('n=33,b=2,k=17', 33, 2, None),
  )
  def test_optimization(self, n, min_sep, k):

    min_buffers = 0
    max_buffers = 3
    blt = buffered_toeplitz.optimize(
        n=n,
        min_sep=min_sep,
        max_participations=k,
        min_buffers=min_buffers,
        max_buffers=max_buffers,
        max_optimizer_steps=10,
    )
    self.assertIsInstance(blt, buffered_toeplitz.BufferedToeplitz)
    num_buffers = blt.buf_decay.shape[0]
    self.assertGreaterEqual(num_buffers, min_buffers)
    self.assertLessEqual(num_buffers, max_buffers)

  @parameterized.named_parameters(
      ('n=10,b=1,k=3', 10, 1, 3),
      ('n=32,b=8,k=None', 32, 8, None),
      ('n=32,b=2,k=20', 32, 2, 20),  # k greater than max possible
      ('n=33,b=2,k=17', 33, 2, None),
  )
  def test_optimize_zero_buffers_correct_loss(self, n, min_sep, k):
    blt = buffered_toeplitz.optimize(
        n=n,
        min_sep=min_sep,
        max_participations=k,
        min_buffers=0,
        max_buffers=0,
    )
    loss = brute_force_max_loss(blt, n=n, min_sep=min_sep, max_participations=k)
    num_buffers = blt.buf_decay.shape[0]
    self.assertEqual(num_buffers, 0)
    expected_loss = _max_loss_for_identity_strategy(
        n=n, min_sep=min_sep, max_participations=k
    )
    assert_allclose(loss, expected_loss)

  @parameterized.named_parameters((f'{n=}', n) for n in [1, 2, 8, 11])
  def test_full_participation_produces_identity(self, n):
    blt = buffered_toeplitz.optimize(
        n=n,
        min_sep=1,
        max_participations=n,
        min_buffers=0,
        max_buffers=3,
        max_optimizer_steps=100,
    )
    num_buffers = blt.buf_decay.shape[0]
    self.assertEqual(num_buffers, 0)

  @parameterized.named_parameters(
      ('n=k=3_nbuf=2', 3, 2),
      ('n=k=16_nbuf=4', 16, 4),
      ('n=k=32_nbuf=3', 32, 3),
      ('n=k=33_nbuf=3', 33, 3),
  )
  def test_optimization_hard_cases(self, nk, num_buffers):
    # Note the optimal solution for n=k with minsep=1 should
    # be the identity matrix, so the optimization can be ill-behaved here.
    # If we allowed 0 buffers, we would get that solution exactly.
    blt = buffered_toeplitz.optimize(
        n=nk,
        min_sep=1,
        max_participations=nk,
        min_buffers=num_buffers,
        max_buffers=num_buffers,
    )

    self.assertIsInstance(blt, buffered_toeplitz.BufferedToeplitz)
    self.assertEqual(blt.buf_decay.shape, (num_buffers,))
    self.assertEqual(blt.output_scale.shape, (num_buffers,))

    # Because we know the identity matrix is optimal, we should be fairly
    # close to this performance.
    expected_loss = _max_loss_for_identity_strategy(
        n=nk, min_sep=1, max_participations=nk
    )
    loss_fn = buffered_toeplitz.LossFn.build_min_sep(
        n=nk, min_sep=1, max_participations=nk
    )
    assert_allclose(loss_fn.loss(blt), expected_loss, atol=1e-3, rtol=1e-2)

  @hypothesis.settings(max_examples=scale_max_examples(0.25))
  @hypothesis.given(
      num_buffers=st.integers(1, 3),
      max_participations=st.integers(2, 12),
      min_sep=st.integers(5, 100),
  )
  def test_optimize_better_than_baseline(
      self, num_buffers, max_participations, min_sep
  ):
    n = 500

    # Hard-coded BLTs optimized for n=1000, max_participations = 2, min_sep = 5.
    # These should be worse than optimizing for the actual max_participations
    # and min_sep.
    baseline_blt = {
        1: buffered_toeplitz.BufferedToeplitz.build(
            buf_decay=[0.9877027559462715], output_scale=[0.10595704738115597]
        ),
        2: buffered_toeplitz.BufferedToeplitz.build(
            buf_decay=[0.993433070819512, 0.6342644427247973],
            output_scale=[0.07538320617916519, 0.5144060910151949],
        ),
        3: buffered_toeplitz.BufferedToeplitz.build(
            buf_decay=[
                0.9934331427558322,
                0.6342832547407998,
                0.6342732636848848,
            ],
            output_scale=[
                0.07538346317367392,
                0.4808525558278832,
                0.0335536865929295,
            ],
        ),
    }[num_buffers]

    loss_fn = functools.partial(
        brute_force_max_loss,
        n=n,
        min_sep=min_sep,
        max_participations=max_participations,
    )

    blt = buffered_toeplitz.optimize(
        n=n,
        min_sep=min_sep,
        max_participations=max_participations,
        min_buffers=num_buffers,
        max_buffers=num_buffers,
    )
    self.assertLess(loss_fn(blt), 1.001 * loss_fn(baseline_blt))

  def test_optimize_single_participation_golden(self):
    num_buffers = 2
    blt = buffered_toeplitz.optimize(
        error='max',
        min_buffers=num_buffers,
        max_buffers=num_buffers,
        n=100000,
    )
    assert_allclose(blt.output_scale, [0.0122853238, 0.1878393665], rtol=1e-5)
    assert_allclose(blt.buf_decay, [0.9999528892, 0.9857068667], rtol=1e-5)

  @hypothesis.given(
      n_nbuf_rtol_tuple=st.sampled_from([
          # fmt: off
          # pylint:disable=bad-whitespace
          # We spot check a range of n and nbuf (both above and below the
          # number of buffers that should be optimal).
          # (n, nbuf, rtol)
          (         10, 1, 1.04),
          (        100, 2, 1.03),
          (        100, 3, 1.03),
          (      1_000, 2, 1.04),
          (      1_000, 3, 1.03),
          (     10_000, 1, 1.45),  # Too few buffers.
          (     10_000, 3, 1.03),
          (     10_000, 4, 1.03),
          (    100_000, 2, 1.16),  # Too few buffers.
          (    100_000, 4, 1.03),
          (  1_000_000, 5, 1.05),
          ( 10_000_000, 2, 1.40),  # Too few buffers.
          ( 10_000_000, 4, 1.05),
          ( 10_000_000, 6, 1.06),
          ( 10_000_000, 8, 1.10),
          (100_000_000, 2, 1.60),  # Too few buffers.
          (100_000_000, 7, 1.05),
          # fmt: on
          # pylint:enable=bad-whitespace
      ])
  )
  def test_optimize_single_participation_near_optimal(self, n_nbuf_rtol_tuple):
    n, num_buffers, rtol = n_nbuf_rtol_tuple
    blt = buffered_toeplitz.optimize(
        error='max',
        min_buffers=num_buffers,
        max_buffers=num_buffers,
        n=n,
    )
    root_loss = np.sqrt(buffered_toeplitz.max_loss(blt, n))
    # We compare our loss to that of the tight-up-to-constant-factors
    # upper bound on the optimal loss of R. Mathias.
    # “The Hadamard operator norm of a circulant and applications” Cor 3.5.
    # See also https://arxiv.org/pdf/2404.16706 Eq. (2.8)
    root_loss_bound = 1 + np.log(n) / np.pi
    ratio = root_loss / root_loss_bound
    # We don't usually print() a lot during tests, but this summary is
    # useful for updating this test to keep the tolerances tight.
    if ratio < 0.95 * rtol:
      status = '***'  # rtol could be tightened
    elif ratio >= rtol:
      status = '!!!'  # Failing, might need to raise rtol
    else:
      status = ''  # OK
    print(
        f'{n=:10d}, {num_buffers=:2d}, current {rtol=:5.2f}, actual'
        f' {ratio=:6.3f} {status}'
    )
    self.assertLessEqual(
        root_loss,
        rtol * root_loss_bound,
        msg=(
            f'Loss {root_loss} is not close enough to the bound'
            f' {root_loss_bound} with {rtol=} at {n=}, {num_buffers=}. '
            f'Would pass at rtol={ratio:.3f}'
        ),
    )

  @hypothesis.given(
      n=st.integers(1, 1000),
      num_buffers=st.integers(1, 3),
  )
  @hypothesis.example(n=58, num_buffers=3)
  @hypothesis.settings(max_examples=scale_max_examples(0.25))
  def test_optimize_closed_form_matches_materialized(
      self,
      n,
      num_buffers,
  ):
    # This is a bit of an integration test. We verify that for
    # single-participation cases, the closed-form loss
    # (`LossFn.build_closed_form_single_participation`) and the loss computed
    # from the materialized coefficients (`LossFn.build_min_sep`) matrix are
    # equal, and optimizing these losses produces a roughly equivalent BLT
    # (within 2%).
    error = 'max'

    closed_form_loss = (
        buffered_toeplitz.LossFn.build_closed_form_single_participation(n=n)
    )
    closed_form_blt, _ = buffered_toeplitz.optimize_loss(
        loss_fn=closed_form_loss,
        num_buffers=num_buffers,
    )
    coef_loss = buffered_toeplitz.LossFn.build_min_sep(
        n=n, min_sep=1, max_participations=1, error=error
    )
    coef_loss_blt, _ = buffered_toeplitz.optimize_loss(
        loss_fn=coef_loss,
        num_buffers=num_buffers,
    )
    loss1 = closed_form_loss.loss(closed_form_blt)
    loss2 = coef_loss.loss(closed_form_blt)
    assert_allclose(
        loss1,
        loss2,
        err_msg=(
            'closed_form_loss does not match coef_loss on the'
            ' closed-form-optimized BLT'
        ),
    )
    loss3 = closed_form_loss.loss(coef_loss_blt)
    loss4 = coef_loss.loss(coef_loss_blt)
    assert_allclose(
        loss3,
        loss4,
        err_msg=(
            'closed_form_loss does not match coef_loss on the'
            ' coef-loss-optimized BLT'
        ),
    )

    assert_allclose(
        loss1,
        loss3,
        rtol=0.02,
        err_msg=(
            'Closed-form optimization (x) does not match coefficient-based'
            ' optimization (y)'
        ),
    )


class ClosedFormsTest(parameterized.TestCase):
  """Tests closed-form implementations and optimization."""

  def test_geometric_sum_basic(self):
    geometric_sum = buffered_toeplitz.geometric_sum
    assert_allclose(geometric_sum(1.3, 0.9, num=0), 0)
    assert_allclose(geometric_sum(1.3, 0.9, num=1), 1.3)
    assert_allclose(geometric_sum(1.3, 0.9, num=2), 1.3 * (1 + 0.9))
    assert_allclose(geometric_sum(1.3, 0.9, num=3), 1.3 * (1 + 0.9 + 0.9**2))
    assert_allclose(geometric_sum(1.3, 0.9, num=jnp.inf), 1.3 / 0.1)
    self.assertTrue(jnp.isinf(geometric_sum(1.3, 1.0, num=jnp.inf)))
    for n in [1, 2, 5, 100000]:
      assert_allclose(geometric_sum(1.3, 1.0, num=n), 1.3 * n)

  @hypothesis.given(n=st.integers(1, 1000), pow_ten=st.integers(1, 10))
  def test_geometric_sum_near_one(self, n, pow_ten):
    r = 1.0 - 10 ** (-pow_ten)
    a = 1.0

    def brute_force_geometric_sum(a, r, num) -> jnp.ndarray:
      return a * jnp.sum(r ** jnp.arange(num))

    assert_allclose(
        brute_force_geometric_sum(a, r, n),
        buffered_toeplitz.geometric_sum(a, r, n),
    )

  @hypothesis.given(
      a=st.floats(0.1, 10.0),
      magnitude=st.floats(1.001, 5.0),
      negative=st.booleans(),
      n=st.integers(1, 40),
  )
  # theta_j * theta_k for a buf_decay of -1.5, which arises in max_error:
  @hypothesis.example(a=1.0, magnitude=2.25, negative=False, n=20)
  @hypothesis.example(a=1.0, magnitude=1.5, negative=True, n=20)
  def test_geometric_sum_ratio_magnitude_above_one(
      self, a, magnitude, negative, n
  ):
    # The Taylor expansion used by geometric_sum is only valid in a
    # neighborhood of r = 1, so |r| > 1 must fall back on the (exact) direct
    # calculation. A one-sided threshold previously sent every r > 1 to the
    # series approximation, silently returning a wildly incorrect value.
    r = -magnitude if negative else magnitude
    brute_force_grad = jax.value_and_grad(
        lambda a, r, n: a * jnp.sum(r ** jnp.arange(0, n)), argnums=(0, 1)
    )
    grad_fn = jax.value_and_grad(
        buffered_toeplitz.geometric_sum, argnums=(0, 1)
    )
    v1, g1 = brute_force_grad(a, r, n)
    v2, g2 = grad_fn(a, r, n)
    assert_allclose(v1, v2, rtol=1e-9, err_msg='Values are not equal.')
    assert_allclose(g1, g2, rtol=1e-7, err_msg='Gradients are not equal.')

  @hypothesis.given(
      a=st.floats(0.0, 1e6), r=st.floats(0.0, 1.0), n=st.integers(1, 100)
  )
  @hypothesis.example(a=1.0, r=1.0, n=1)
  @hypothesis.example(a=1.0, r=1.0, n=5)
  @hypothesis.example(a=1.0, r=1 - 1e-4, n=5)  # Uses direct computation
  @hypothesis.example(a=1.0, r=1 - 1e-5, n=5)  # Uses series approx
  @hypothesis.example(a=1.0, r=1 - 1e-5, n=100)  # Uses direct computation
  @hypothesis.example(a=1.0, r=1 - 1e-8, n=100)  # Uses series approx
  def test_geometric_sum_gradients(self, a, r, n):
    brute_force_grad = jax.value_and_grad(
        # Slightly awkard expression to work around some issues
        # with autodiff with jnp.sum and zero exponent.
        lambda a, r, n: a * jnp.sum(r ** jnp.arange(0, n)),
        argnums=(0, 1),
    )

    grad_fn = jax.value_and_grad(
        buffered_toeplitz.geometric_sum, argnums=(0, 1)
    )
    v1, g1 = brute_force_grad(a, r, n)
    v2, g2 = grad_fn(a, r, n)
    assert_allclose(v1, v2, atol=1e-8, err_msg='Values are not equal.')
    assert_allclose(g1, g2, atol=1e-8, err_msg='Gradients are not equal.')

  @hypothesis.given(blt_tuple=st.sampled_from(BLT_TUPLES))
  def test_get_inverse_parameters_and_blt_pair(self, blt_tuple):
    blt = make_blt(*blt_tuple)
    hypothesis.assume(buffered_toeplitz.min_buf_decay_gap(blt.buf_decay) > 1e-8)

    inv_blt = blt.inverse(blt)

    # Test the Toeplitz coefficients:
    inv_coefs = blt.inverse_as_streaming_matrix().materialize(n=25)[:, 0]
    inv_coefs2 = inv_blt.toeplitz_coefs(n=25)
    assert_allclose(inv_coefs, inv_coefs2)

    # Test the inverse of the inverse recovers the blt:
    blt2 = inv_blt.inverse(inv_blt)
    assert_allclose(blt.buf_decay, blt2.buf_decay, err_msg=f'{blt=}\n{blt2=}')
    assert_allclose(
        blt.output_scale, blt2.output_scale, err_msg=f'{blt=}\n{blt2=}'
    )

    # Test we actually have an inverse, just to be safe.
    C = blt.materialize(n=7)
    Cinv = inv_blt.as_streaming_matrix().materialize(n=7)
    assert_allclose(Cinv @ C, jnp.eye(7))

    # Use this pair of (theta, theta_hat) to test blt_pair_from_theta_pair.
    # Exclude test cases where thetas are not sufficiently distinct.

    alt_blt, alt_inv_blt = buffered_toeplitz.blt_pair_from_theta_pair(
        blt.buf_decay, inv_blt.buf_decay
    )
    assert_allclose(
        alt_blt.buf_decay,
        blt.buf_decay,
    )
    assert_allclose(alt_blt.output_scale, blt.output_scale)
    assert_allclose(alt_inv_blt.buf_decay, inv_blt.buf_decay)
    assert_allclose(alt_inv_blt.output_scale, inv_blt.output_scale)

  @hypothesis.given(
      blt_tuple=st.sampled_from(BLT_TUPLES), n=st.sampled_from([1, 2, 10])
  )
  def test_sensitivity(self, blt_tuple, n):
    blt = make_blt(*blt_tuple)
    sens_sq = toeplitz.sensitivity_squared(blt.toeplitz_coefs(n))
    sens_sq2 = buffered_toeplitz.sensitivity_squared(blt, n)
    assert_allclose(sens_sq, sens_sq2, err_msg='Sensitivities are not equal')
    sens_sq_max = buffered_toeplitz.sensitivity_squared(
        blt, n=buffered_toeplitz.jnp.inf
    )
    self.assertLess(sens_sq2, sens_sq_max + 1e-10)

  def test_sensitivty_too_big_buf_decay(self):
    blt = make_blt([1.5], [0.1])
    self.assertTrue(jnp.isinf(buffered_toeplitz.sensitivity_squared(blt, n=10)))
    self.assertTrue(
        jnp.isinf(buffered_toeplitz.sensitivity_squared(blt, n=jnp.inf))
    )

  @hypothesis.given(blt_tuple=st.sampled_from(BLT_TUPLES))
  def test_limit_sensitivity(self, blt_tuple):
    big_n = 1e20
    blt = make_blt(*blt_tuple)
    # A buf_decay of 1 will give us large but finite sensitivity,
    # but limit will be infinite:
    hypothesis.assume(jnp.all(blt.buf_decay < 1))
    sens_big_n = buffered_toeplitz.sensitivity_squared(blt, big_n)
    sens_limit = buffered_toeplitz.sensitivity_squared(blt, n=np.inf)
    assert_allclose(sens_big_n, sens_limit)

  @hypothesis.given(
      blt_tuple=st.sampled_from(BLT_TUPLES), n=st.integers(500, 1000)
  )
  def test_max_error(self, blt_tuple, n):
    # Note: For small n, the approximations are actually a bit worse,
    # but we don't really care about this regime, hence we prefer
    # to keep n >= 500 with a tight rtol rather than relaxing it so
    # the test passes with smaller n.
    inv_blt = make_blt(*blt_tuple)
    max_err = direct_max_error_for_inv(inv_blt, n)
    closed_form_error = buffered_toeplitz.max_error(inv_blt, n)
    assert_allclose(max_err, closed_form_error, rtol=1e-6)

  def test_max_error_hard_case(self):
    inv_blt = make_blt(
        # Use a range of theta magnitudes, and small omegas to produce
        # "reasonable" Toeplitz coefficients.
        # Because the computation of max_error usese all pairs of
        # thetas from this list, we should hit all the different Taylor
        # series approximations in addition to the direct computation.
        [1 - 10 ** (-p) for p in range(11)],
        [-0.01 * 10 ** (-np.sqrt(p)) for p in range(11)],
    )
    for n in [1000, 10000, 100000]:
      max_err = direct_max_error_for_inv(inv_blt, n)
      closed_form_error = buffered_toeplitz.max_error(inv_blt, n)
      assert_allclose(
          max_err,
          closed_form_error,
          # This is a bit lax, but with 10*10 = 100 Gamma_jk computations,
          # there is a lot of opportunity for errors to add up, so
          # this is probably reasonable.
          rtol=1e-5,
          err_msg=f'Max error is not equal at {n=}',
      )

  @hypothesis.given(blt_tuple=st.sampled_from(BLT_TUPLES))
  def test_limit_iteration_error(self, blt_tuple):
    inv_blt = make_blt(*blt_tuple)
    # A buf_decay == 1 will not work with infinite n, we expect error to
    # blow up.
    hypothesis.assume(jnp.all(inv_blt.buf_decay < 1))
    really_big_n = 1e20
    closed_form_error = (
        buffered_toeplitz.max_error(inv_blt, really_big_n) / really_big_n
    )
    limit_error = buffered_toeplitz.limit_max_error(inv_blt)
    assert_allclose(limit_error, closed_form_error)

  @hypothesis.given(blt_tuple=st.sampled_from(BLT_TUPLES), n=st.integers(1, 50))
  @hypothesis.example(([1.0, 1 - 1e-9, 0.99, 0.1], [1e-6, 1e-5, 1e-3, 1e-2]), 1)
  def test_max_iter_loss(self, blt_tuple, n):
    blt = make_blt(*blt_tuple)
    hypothesis.assume(jnp.all(blt.output_scale > 0))
    loss = buffered_toeplitz.max_loss(blt, n)

    # Direct loss calculation:

    loss2 = toeplitz.loss(
        strategy_coef=blt.toeplitz_coefs(n), reduction_fn=jnp.max
    )

    assert_allclose(
        loss, loss2, err_msg='max_loss does not match direct calculation'
    )

    # The limiting loss scaled by n should be a lower bound, see writeup.
    limit_loss = n * buffered_toeplitz.limit_max_loss(blt)
    if jnp.isfinite(limit_loss):
      self.assertLessEqual(limit_loss, loss + 1e-10)

  @parameterized.parameters(8, 32, 128)
  def test_max_loss_when_noising_blt_oscillates(self, n):
    # A strategy BLT with buf_decay in (0, 1) and output_scale > 0 passes every
    # guard on max_loss, yet its inverse (the noising matrix C^{-1}) can have a
    # buf_decay below -1 (here -1.5), so theta_j * theta_k > 1. Such BLTs lie
    # outside the class optimize() produces, but max_loss is public and must
    # not silently return a wrong value for them: it previously returned a
    # negative "squared error" for this BLT.
    blt = make_blt([0.5], [2.0])
    self.assertLess(blt.inverse().buf_decay[0], -1.0)

    loss = buffered_toeplitz.max_loss(blt, n)
    expected = toeplitz.loss(
        strategy_coef=blt.toeplitz_coefs(n), reduction_fn=jnp.max
    )
    self.assertGreater(loss, 0.0)
    assert_allclose(
        loss,
        expected,
        rtol=1e-9,
        err_msg='max_loss does not match direct calculation',
    )

  @hypothesis.given(blt_tuple=st.sampled_from(BLT_TUPLES), n=st.integers(1, 50))
  def test_max_iter_loss_gradients(self, blt_tuple, n):
    blt = make_blt(*blt_tuple)
    hypothesis.assume(jnp.all(blt.output_scale > 0))

    grad1 = jax.grad(buffered_toeplitz.max_loss)(blt, n)
    brute_force_loss = lambda blt, n: toeplitz.loss(
        strategy_coef=blt.toeplitz_coefs(n), reduction_fn=jnp.max
    )
    grad2 = jax.grad(brute_force_loss)(blt, n)

    assert_allclose(
        grad1.buf_decay,
        grad2.buf_decay,
        err_msg='buf_decay gradient does does not match direct calculation',
        atol=1e-4,
        rtol=1e-4,
    )
    assert_allclose(
        grad1.output_scale,
        grad2.output_scale,
        err_msg='output_scale gradient does does not match direct calculation',
        atol=1e-4,
        rtol=1e-4,
    )


if __name__ == '__main__':
  absltest.main()

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
import chex
import jax
from jax import numpy as jnp
from jax_privacy import noise_addition
from jax_privacy.matrix_factorization import buffered_toeplitz
from jax_privacy.matrix_factorization import streaming_matrix
from jax_privacy.matrix_factorization import toeplitz
import numpy as np
import optax
import scipy.stats

_ITERATIONS = 3
_STDDEV = 1.0
_SEED = 1234

_PARAM_SHAPES = {
    'scalar': jax.ShapeDtypeStruct(shape=(), dtype=jnp.float64),
    'vector': jax.ShapeDtypeStruct(shape=(10,), dtype=jnp.float64),
    'tuple': (jax.ShapeDtypeStruct((6, 6), jnp.float64),) * 2,
    'heterogeneous_dtypes': (
        jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.float32),
        jax.ShapeDtypeStruct(shape=(3,), dtype=jnp.float64),
    ),
}


def gaussian_privatizer_fn(prng_key: jax.Array | None = None):
  if prng_key is None:
    prng_key = jax.random.key(_SEED)
  return noise_addition.gaussian_privatizer(prng_key=prng_key, stddev=_STDDEV)


def dense_matrix_factorization_privatizer_fn(
    noising_matrix: jax.Array | None = None,
    prng_key: jax.Array | None = None,
    stddev: float = _STDDEV,
):
  if prng_key is None:
    prng_key = jax.random.key(_SEED)
  if noising_matrix is None:
    noising_matrix = jnp.eye(_ITERATIONS) + 0.1
  return noise_addition.matrix_factorization_privatizer(
      noising_matrix,
      prng_key=prng_key,
      stddev=stddev,
  )


def streaming_matrix_factorization_privatizer_fn(
    noising_matrix: streaming_matrix.StreamingMatrix | None = None,
    prng_key: jax.Array | None = None,
):
  if noising_matrix is None:
    noising_matrix = streaming_matrix.momentum_sgd_matrix(momentum=0.95)
  if prng_key is None:
    prng_key = jax.random.key(_SEED)
  return noise_addition.matrix_factorization_privatizer(
      noising_matrix,
      prng_key=prng_key,
      stddev=_STDDEV,
  )


def tree_stack(trees):
  return jax.tree.map(lambda *args: jnp.stack(args, axis=0), *trees)


_PRIVATIZER_FNS = (
    gaussian_privatizer_fn,
    dense_matrix_factorization_privatizer_fn,
    streaming_matrix_factorization_privatizer_fn,
)


class PrivatizerTest(chex.TestCase, parameterized.TestCase):

  def test_mf_privatizer_raises_non_matrix_arg(self):
    with self.assertRaisesRegex(ValueError, 'Expected 2D'):
      noise_addition.matrix_factorization_privatizer(
          jnp.ones(10),
          prng_key=jax.random.key(10),
          stddev=0.0,
      )

  @parameterized.parameters(
      gaussian_privatizer_fn,
      lambda: dense_matrix_factorization_privatizer_fn(jnp.eye(_ITERATIONS)),
  )
  def test_normality_and_magnitude(self, privatizer_fn):
    privatizer = privatizer_fn()

    # we use a large vector here to make it unlikely the test passes by chance.
    grads = jnp.zeros(512)
    state = privatizer.init(grads)

    values = []

    for _ in range(_ITERATIONS):
      noise, state = privatizer.update(grads, state)
      values.extend(jax.tree.leaves(noise))

    samples = [float(x) for x in jnp.concatenate(values)]

    # Should not reject the null hypothesis for correct Gaussian scale.
    result = scipy.stats.kstest(samples, scipy.stats.norm(scale=_STDDEV).cdf)
    self.assertGreater(result.pvalue, 0.025)

    # Should reject the null hypothesis for incorrect Gaussian scale.
    low_scale = _STDDEV * 2 / 3
    result = scipy.stats.kstest(samples, scipy.stats.norm(scale=low_scale).cdf)
    self.assertLess(result.pvalue, 0.05)

    high_scale = _STDDEV * 3 / 2
    result = scipy.stats.kstest(samples, scipy.stats.norm(scale=high_scale).cdf)
    self.assertLess(result.pvalue, 0.05)

  @parameterized.product(
      privatizer_fn=_PRIVATIZER_FNS,
      jit=(True, False),
  )
  def test_output_dtype_matches_input(self, privatizer_fn, jit):

    example_params = (
        jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.float32),
        jax.ShapeDtypeStruct(shape=(3,), dtype=jnp.bfloat16),
        jax.ShapeDtypeStruct(shape=(4,), dtype=jnp.float64),
        jax.ShapeDtypeStruct(shape=(5,), dtype=jnp.float16),
    )

    privatizer = privatizer_fn()

    maybe_jit = jax.jit if jit else (lambda x: x)

    state = state0 = maybe_jit(privatizer.init)(example_params)
    grads = optax.tree.zeros_like(example_params)
    for _ in range(_ITERATIONS):
      result, state = maybe_jit(privatizer.update)(
          sum_of_clipped_grads=grads,
          noise_state=state,
      )
      chex.assert_trees_all_equal_shapes_and_dtypes(result, example_params)
      chex.assert_trees_all_equal_shapes_and_dtypes(state, state0)

  @parameterized.named_parameters(_PARAM_SHAPES.items())
  def test_mf_privatizer_matches_explicit_matmul(self, example_grad):
    noising_matrix = jax.random.normal(
        jax.random.key(0), shape=[_ITERATIONS, _ITERATIONS]
    )
    eye = jnp.eye(_ITERATIONS)
    mf_privatizer = dense_matrix_factorization_privatizer_fn(noising_matrix)
    sgd_privatizer = dense_matrix_factorization_privatizer_fn(eye)
    grads = optax.tree.zeros_like(example_grad)
    mf_state = mf_privatizer.init(grads)
    sgd_state = sgd_privatizer.init(grads)
    correlated_noise_vectors = []
    uncorrelated_noise_vectors = []
    for _ in range(_ITERATIONS):
      mf_noise, mf_state = mf_privatizer.update(grads, mf_state)
      sgd_noise, sgd_state = sgd_privatizer.update(grads, sgd_state)
      correlated_noise_vectors.append(mf_noise)
      uncorrelated_noise_vectors.append(sgd_noise)

    mf_noise = tree_stack(correlated_noise_vectors)
    sgd_noise = tree_stack(uncorrelated_noise_vectors)

    def matrix_multiply(x):
      Cinv = noising_matrix  # pylint: disable=invalid-name
      return (Cinv @ x.reshape(x.shape[0], -1)).reshape(x.shape).astype(x.dtype)

    expected_noise = jax.tree.map(matrix_multiply, sgd_noise)
    chex.assert_trees_all_close(mf_noise, expected_noise, atol=1e-7)

  @parameterized.named_parameters(_PARAM_SHAPES.items())
  def test_matrix_multiply_zero_matrix(self, example_grad):
    noising_matrix = jnp.zeros([_ITERATIONS, _ITERATIONS])
    mf_privatizer = dense_matrix_factorization_privatizer_fn(noising_matrix)
    grads = jax.tree.map(jnp.ones_like, example_grad)
    state = mf_privatizer.init(grads)

    for _ in range(_ITERATIONS):
      result, state = mf_privatizer.update(grads, state)
      # No noise is added with noising_matrix = 0, so we expect grads unchanged.
      chex.assert_trees_all_close(result, grads, atol=1e-5)

  @parameterized.parameters(*_PRIVATIZER_FNS)
  def test_functional_purity(self, privatizer_fn):
    privatizer = privatizer_fn()
    grads = jnp.zeros(10)
    state1 = state2 = privatizer.init(grads)
    for _ in range(_ITERATIONS):
      result1, state1 = privatizer.update(grads, state1)
      result2, state2 = privatizer.update(grads, state2)
      chex.assert_trees_all_equal(result1, result2)
      chex.assert_trees_all_equal(state1, state2)

  @parameterized.parameters(*_PRIVATIZER_FNS)
  def test_noise_unique(self, privatizer_fn):
    # Tests uniqueness across iterations, indices, pytree leaves, and prng_key.
    privatizer1 = privatizer_fn(prng_key=jax.random.key(1234))
    privatizer2 = privatizer_fn(prng_key=jax.random.key(5678))
    grads = {'a': jnp.zeros(4), 'b': jnp.zeros(4)}

    values = []
    state1 = privatizer1.init(grads)
    state2 = privatizer2.init(grads)

    for _ in range(_ITERATIONS):
      noise1, state1 = privatizer1.update(grads, state1)
      noise2, state2 = privatizer2.update(grads, state2)

      values.extend(jax.tree.leaves(noise1))
      values.extend(jax.tree.leaves(noise2))

    values = [float(x) for x in jnp.concatenate(values)]
    self.assertLen(set(values), len(values))

  @parameterized.named_parameters(
      ('dense_identity', np.eye(_ITERATIONS)),
      ('dense_scaled_shifted', np.eye(_ITERATIONS) + 0.1),
      ('dense_random', np.random.rand(_ITERATIONS, _ITERATIONS)),
      ('streaming_prefix', streaming_matrix.prefix_sum()),
      ('streaming_identity', streaming_matrix.identity()),
  )
  def test_scale_invariance(self, noising_matrix):
    # Test equivalence between modifying stddev or noising_matrix.
    key = jax.random.key(_SEED)

    privatizer0 = noise_addition.matrix_factorization_privatizer(
        noising_matrix=noising_matrix, prng_key=key, stddev=_STDDEV
    )
    privatizer1 = noise_addition.matrix_factorization_privatizer(
        noising_matrix=noising_matrix * _STDDEV, prng_key=key, stddev=1.0
    )

    params = jnp.zeros(10)
    state0 = privatizer0.init(params)
    state1 = privatizer1.init(params)

    for _ in range(_ITERATIONS):
      noise0, state0 = privatizer0.update(params, state0)
      noise1, state1 = privatizer1.update(params, state1)
      chex.assert_trees_all_close(noise0, noise1, atol=1e-5)

  @parameterized.parameters(['banded', 'blt'])
  def test_identity_banded_blt_exact_match(self, name):
    identity = streaming_matrix.identity()
    if name == 'banded':
      noising_matrix = toeplitz.inverse_as_streaming_matrix(jnp.array([1]))
    else:
      noising_matrix = buffered_toeplitz.BufferedToeplitz.build(
          [], [], dtype=jnp.float32
      ).inverse_as_streaming_matrix()

    def get_noise_sample(noising_matrix):
      privatizer = noise_addition.matrix_factorization_privatizer(
          noising_matrix=noising_matrix, prng_key=0, stddev=1.0
      )
      params = jnp.zeros(10)
      state = privatizer.init(params)
      noise, _ = privatizer.update(params, state)
      return noise

    # We expect the noise to be the same, not just statistically similar.
    chex.assert_trees_all_close(
        get_noise_sample(identity),
        get_noise_sample(noising_matrix),
    )

  @parameterized.named_parameters(
      ('dense_identity', np.eye(_ITERATIONS)),
      ('streaming_identity', streaming_matrix.identity()),
  )
  def test_optax_chain(self, noising_matrix):
    key = jax.random.key(_SEED)

    privatizer = noise_addition.matrix_factorization_privatizer(
        noising_matrix=noising_matrix, prng_key=key, stddev=_STDDEV
    )
    optimizer = optax.adamw(learning_rate=0.1)
    noisy_optimizer = optax.chain(privatizer, optimizer)

    params = jnp.zeros(10)
    state0 = noisy_optimizer.init(params)
    update, state1 = noisy_optimizer.update(params, state0, params)
    chex.assert_trees_all_equal_shapes_and_dtypes(state0, state1)
    chex.assert_trees_all_equal_shapes_and_dtypes(update, params)

  @parameterized.parameters(*_PRIVATIZER_FNS)
  def test_init_for_step(self, privatizer_fn):
    privatizer = privatizer_fn()
    params = {'a': jnp.zeros(4), 'b': jnp.zeros((2, 3))}
    expected_state = privatizer.init(params)
    for step in range(_ITERATIONS):
      actual_state = noise_addition.init_for_step(privatizer, params, step=step)
      actual_noise, _ = privatizer.update(params, actual_state)
      expected_noise, expected_state = privatizer.update(params, expected_state)
      chex.assert_trees_all_close(actual_noise, expected_noise)

  @parameterized.parameters(
      gaussian_privatizer_fn,
      dense_matrix_factorization_privatizer_fn,
  )
  def test_init_for_step_eliminates_dead_noise_generation(self, privatizer_fn):
    # A 4 TB ShapeDtypeStruct would OOM if XLA allocated zeros or noise buffers.
    huge_params = jax.ShapeDtypeStruct(shape=(10**12,), dtype=jnp.float32)
    privatizer = privatizer_fn()
    state = noise_addition.init_for_step(privatizer, huge_params, step=2)
    expected_state = noise_addition.init_for_step(
        privatizer, jnp.zeros(1, dtype=jnp.float32), step=2
    )
    chex.assert_trees_all_equal(state, expected_state)

  def test_init_for_step_reuses_compilation_across_steps(self):
    privatizer = streaming_matrix_factorization_privatizer_fn()
    params = {'a': jnp.zeros(4), 'b': jnp.zeros((2, 3))}
    noise_addition._init_for_step.clear_cache()

    for step in (0, 1, 2, jnp.int32(3)):
      _ = noise_addition.init_for_step(privatizer, params, step=step)

    self.assertEqual(noise_addition._init_for_step._cache_size(), 1)

  def test_init_for_step_bounds_peak_memory(self):
    noising_matrix = toeplitz.inverse_as_streaming_matrix(
        jnp.ones(17) / jnp.sqrt(17.0), column_normalize_for_n=32
    )
    privatizer = noise_addition.matrix_factorization_privatizer(
        noising_matrix=noising_matrix, prng_key=jax.random.key(0), stddev=1.0
    )
    params = {f'w{i}': jnp.zeros((64, 64), dtype=jnp.float32) for i in range(8)}
    expected_state = privatizer.init(params)
    state_bytes = sum(x.nbytes for x in jax.tree.leaves(expected_state))

    compiled = noise_addition._init_for_step.lower(
        privatizer, params, jnp.int32(3)
    ).compile()
    mem = compiled.memory_analysis()
    self.assertIsNotNone(mem)
    peak_bytes = mem.output_size_in_bytes + mem.temp_size_in_bytes
    self.assertLess(peak_bytes, int(1.25 * state_bytes))

    for _ in range(3):
      _, expected_state = privatizer.update(params, expected_state)
    actual_state = noise_addition.init_for_step(privatizer, params, step=3)
    chex.assert_trees_all_close(
        privatizer.update(params, actual_state)[0],
        privatizer.update(params, expected_state)[0],
    )


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()

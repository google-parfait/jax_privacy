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

"""Composed streaming operators initialize state from intermediate dtypes."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax_privacy.matrix_factorization import streaming_matrix as sm
import numpy as np


class StreamingCompositionDtypeTest(parameterized.TestCase):

  @parameterized.product(
      dtype=[jnp.int32, jnp.float16, jnp.float32],
      shape=[(4,), (4, 2), (4, 2, 3)],
  )
  def test_composed_prefix_and_diagonal_match_separate_application(
      self, dtype, shape
  ):
    x = jnp.arange(1, np.prod(shape) + 1, dtype=dtype).reshape(shape)
    diagonal = sm.diagonal(jnp.array([0.5, 1.5, 2.0, 0.25], jnp.float32))
    matrix = sm.prefix_sum() @ diagonal
    expected = sm.prefix_sum() @ (diagonal @ x)
    for multiply in [
        lambda data: matrix @ data,
        jax.jit(lambda data: matrix @ data),
    ]:
      actual = multiply(x)
      self.assertEqual(actual.dtype, expected.dtype)
      np.testing.assert_allclose(actual, expected, rtol=2e-6)

  @parameterized.product(
      dtype=[jnp.int32, jnp.float16, jnp.float32],
      x64_enabled=[False, True],
      compiled=[False, True],
  )
  def test_scalar_multiplication_uses_the_promoted_input_dtype(
      self, dtype, x64_enabled, compiled
  ):
    with jax.enable_x64(x64_enabled):
      x = jnp.arange(1, 5, dtype=dtype)

      def multiply(values):
        return (sm.prefix_sum() * 0.5) @ values

      actual = jax.jit(multiply)(x) if compiled else multiply(x)
      np.testing.assert_allclose(
          actual, np.cumsum(np.asarray(x, dtype=np.float64)) * 0.5
      )
      # The scalar diagonal uses the configured default floating-point dtype.
      expected_dtype = jnp.float64 if x64_enabled else jnp.float32
      self.assertEqual(actual.dtype, expected_dtype)

  def test_initialization_does_not_consume_the_first_diagonal_entry(self):
    matrix = sm.prefix_sum() @ sm.diagonal(jnp.array([0.5, 2.0, 4.0]))
    state = matrix.init_multiply(jax.ShapeDtypeStruct((), jnp.int32))
    outputs = []
    for x in [1, 2, 3]:
      y, state = matrix.multiply_next(jnp.array(x, jnp.int32), state)
      outputs.append(y)
    np.testing.assert_allclose(outputs, [0.5, 4.5, 16.5])

  def test_manual_pytree_stream_preserves_leaf_structure(self):
    values = {
        'a': jnp.arange(1, 7, dtype=jnp.int32).reshape(3, 2),
        'b': jnp.arange(1, 4, dtype=jnp.float16),
    }
    diagonal = jnp.array([0.5, 2.0, 4.0])
    matrix = sm.prefix_sum() @ sm.diagonal(diagonal)
    abstract = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), values
    )
    state = matrix.init_multiply(abstract)
    outputs = []
    for i in range(3):
      y, state = matrix.multiply_next(
          jax.tree.map(lambda x, index=i: x[index], values), state
      )
      outputs.append(y)
    for key, value in values.items():
      actual = jnp.stack([output[key] for output in outputs])
      factor = diagonal.reshape((3,) + (1,) * (value.ndim - 1))
      expected = jnp.cumsum(value.astype(jnp.float32) * factor, axis=0)
      np.testing.assert_allclose(actual, expected)

  def test_gradients_through_composed_diagonal_match_dense_linear_map(self):
    x = jnp.array([1.0, 2.0, 3.0], jnp.float16)

    def objective(diag):
      y = (sm.prefix_sum() @ sm.diagonal(diag)) @ x
      return jnp.square(y).sum()

    def reference(diag):
      y = jnp.tril(jnp.ones((3, 3))) @ (diag * x.astype(jnp.float32))
      return jnp.square(y).sum()

    diagonal = jnp.array([0.5, 2.0, 4.0])
    value, grad = jax.jit(jax.value_and_grad(objective))(diagonal)
    expected = jax.value_and_grad(reference)(diagonal)
    np.testing.assert_allclose(value, expected[0])
    np.testing.assert_allclose(grad, expected[1])

  def test_identity_and_same_dtype_compositions_are_unchanged(self):
    x = jnp.arange(1, 5, dtype=jnp.float32)
    for matrix in [
        sm.identity() @ sm.prefix_sum(),
        sm.prefix_sum() @ sm.identity(),
    ]:
      np.testing.assert_allclose(matrix @ x, jnp.cumsum(x))
    matrix = sm.prefix_sum() @ sm.diagonal(jnp.array([0.5]))
    np.testing.assert_allclose(
        matrix.materialize(3), 0.5 * np.tril(np.ones((3, 3)))
    )


if __name__ == '__main__':
  absltest.main()

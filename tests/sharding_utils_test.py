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
import chex
import jax
import jax.numpy as jnp
from jax_privacy import sharding_utils
import numpy as np
import optax


class ShardingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    chex.set_n_cpu_devices(8)
    axis_types = (jax.sharding.AxisType.Explicit,) * 3
    self.mesh = jax.make_mesh((2, 2, 2), ('x', 'y', 'z'), axis_types=axis_types)
    jax.sharding.set_mesh(self.mesh)

  def test_flatten_pspec(self):
    self.assertEqual(
        sharding_utils._flatten_pspec(
            jax.sharding.PartitionSpec(None, ('x', 'y'), None, 'z')
        ),
        jax.sharding.PartitionSpec(('x', 'y', 'z')),
    )
    self.assertEqual(
        sharding_utils._flatten_pspec(
            jax.sharding.PartitionSpec('data', None, ('replica', 'mdl'))
        ),
        jax.sharding.PartitionSpec(('data', 'replica', 'mdl')),
    )

  def test_reshape_add(self):

    x = jax.sharding.reshard(
        jnp.zeros((2, 4, 8)), jax.sharding.PartitionSpec(None, 'y', 'x')
    )
    y = jax.sharding.reshard(
        jnp.arange(2 * 4 * 8), jax.sharding.PartitionSpec(('x', 'y', 'z'))
    )
    z = sharding_utils.local_reshape_add(x, y)

    self.assertEqual(z.shape, x.shape)
    self.assertEqual(z.sharding, x.sharding)

    actual = np.sort(np.array(z).flatten())
    chex.assert_trees_all_equal(actual, y)

  def test_flatten_with_zero_redundancy(self):
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(None, 'y')
    )
    shaped_value = jax.ShapeDtypeStruct((2, 6), jnp.float32, sharding=sharding)
    flat_value = sharding_utils.flatten_with_zero_redundancy(shaped_value)
    self.assertEqual(
        flat_value.sharding.spec, jax.sharding.PartitionSpec(('x', 'y', 'z'))
    )
    self.assertEqual(flat_value.sharding.mesh.axis_names, ('x', 'y', 'z'))
    self.assertEqual(flat_value.shape, (16,))

  def test_ceiling_to_multiple(self):
    self.assertEqual(sharding_utils._ceiling_to_multiple(2, 4), 4)
    self.assertEqual(sharding_utils._ceiling_to_multiple(3, 4), 4)
    self.assertEqual(sharding_utils._ceiling_to_multiple(4, 4), 4)
    self.assertEqual(sharding_utils._ceiling_to_multiple(5, 4), 8)

  def test_flatten_zeros_like_preserves_metadata(self):
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(None, 'y')
    )
    x = jax.device_put(jnp.ones((2, 6), dtype=jnp.float32), sharding)
    flattened = sharding_utils.flatten_with_zero_redundancy(x)
    zeros = jnp.zeros_like(flattened)
    self.assertEqual(zeros.shape, flattened.shape)
    self.assertEqual(zeros.dtype, flattened.dtype)
    self.assertEqual(zeros.sharding.spec, flattened.sharding.spec)
    self.assertEqual(
        zeros.sharding.mesh.axis_names, flattened.sharding.mesh.axis_names
    )

  def test_compute_early_stopping_order(self):
    is_padding = jnp.array([0, 0, 0, 1, 1, 1])

    # Padding examples should appear at the end after reshaping batch axis.
    expected_true_microbatches = {
        1: np.array([[0], [0], [0], [1], [1], [1]]),
        2: np.array([[0, 0], [0, 1], [1, 1]]),
        3: np.array([[0, 0, 0], [1, 1, 1]]),
        6: np.array([[0, 0, 0, 1, 1, 1]]),
    }

    for microbatch_size, expected in expected_true_microbatches.items():
      perm = sharding_utils.compute_early_stopping_order(6, microbatch_size)
      actual = optax.microbatching.reshape_batch_axis(
          is_padding[perm], microbatch_size
      )
      chex.assert_trees_all_equal(actual, expected)


if __name__ == '__main__':
  absltest.main()

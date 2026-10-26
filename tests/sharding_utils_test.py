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


class RandomTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    chex.set_n_cpu_devices(8)

    axis_types = (jax.sharding.AxisType.Explicit,) * 2
    self.mesh = jax.make_mesh((4, 2), ('x', 'y'), axis_types=axis_types)
    jax.set_mesh(self.mesh)

  def test_uniqueness(self):
    rng = np.random.default_rng(42)
    shape = (104,)
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(('x', 'y'))
    )

    sample1 = sharding_utils._parallel_sample(rng, shape, sharding)
    sample2 = sharding_utils._parallel_sample(rng, shape, sharding)
    combined = jnp.concatenate([sample1, sample2])
    # Assert uniqueness within and across samples
    self.assertLen(np.unique(combined), combined.size)

  @parameterized.parameters(
      {'shape': (20,), 'pspec': jax.sharding.PartitionSpec('x')},
      {'shape': (16, 8), 'pspec': jax.sharding.PartitionSpec('x', 'y')},
      {'shape': (8, 8, 4), 'pspec': jax.sharding.PartitionSpec('x', None, 'y')},
  )
  def test_arbitrary_shapes_and_outputs(self, shape, pspec):
    rng = np.random.default_rng(42)
    sharding = jax.sharding.NamedSharding(self.mesh, pspec)

    output = sharding_utils._parallel_sample(
        rng, shape, sharding, dtype=jnp.float32
    )

    self.assertEqual(output.shape, shape)
    self.assertEqual(output.dtype, jnp.float32)
    self.assertEqual(output.sharding, sharding)

  def test_statistics(self):
    rng = np.random.default_rng(123)
    shape = (100000,)
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(('x', 'y'))
    )

    output = sharding_utils._parallel_sample(
        rng, shape, sharding, sampler=np.random.Generator.standard_normal
    )

    mean = jnp.mean(output)
    std = jnp.std(output)

    np.testing.assert_allclose(mean, 0.0, atol=0.02)
    np.testing.assert_allclose(std, 1.0, atol=0.02)

  @parameterized.product(num_devices=(1, 2, 8), shape=((0,), (16,)))
  def test_sampling_single_device_and_empty_arrays(self, num_devices, shape):
    mesh = jax.make_mesh(
        (num_devices,),
        ('data',),
        devices=jax.devices()[:num_devices],
        axis_types=(jax.sharding.AxisType.Explicit,),
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.P('data'))
    struct = {
        'value': jax.ShapeDtypeStruct(shape, jnp.float32, sharding=sharding)
    }

    with jax.set_mesh(mesh):
      output = sharding_utils.parallel_sample_pytree(
          np.random.default_rng(42), struct
      )['value']

    self.assertEqual(output.shape, shape)
    self.assertEqual(output.dtype, jnp.float32)
    self.assertTrue(output.sharding.is_equivalent_to(sharding, len(shape)))
    self.assertTrue(np.isfinite(output).all())

  def test_pytree_structures(self):
    rng = np.random.default_rng(42)
    sharding_a = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec('x', 'y')
    )
    sharding_b = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec('x')
    )

    struct = {
        'a': jax.ShapeDtypeStruct((32, 32), jnp.float32, sharding=sharding_a),
        'b': jax.ShapeDtypeStruct((16,), jnp.float32, sharding=sharding_b),
    }

    output = sharding_utils.parallel_sample_pytree(rng, struct)

    self.assertIsInstance(output, dict)
    self.assertEqual(output['a'].shape, (32, 32))
    self.assertEqual(output['a'].sharding, sharding_a)
    self.assertEqual(output['b'].shape, (16,))
    self.assertEqual(output['b'].sharding, sharding_b)


if __name__ == '__main__':
  absltest.main()

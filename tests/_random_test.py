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
from jax_privacy import _random
import numpy as np


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

    sample1 = _random._parallel_sample(rng, shape, sharding)
    sample2 = _random._parallel_sample(rng, shape, sharding)
    combined = np.concatenate([sample1, sample2])
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

    output = _random._parallel_sample(rng, shape, sharding, dtype=jnp.float32)

    self.assertEqual(output.shape, shape)
    self.assertEqual(output.dtype, jnp.float32)
    self.assertEqual(output.sharding, sharding)

  def test_statistics(self):
    rng = np.random.default_rng(123)
    shape = (100000,)
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(('x', 'y'))
    )

    output = _random._parallel_sample(
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
      output = _random.parallel_sample_pytree(
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

    output = _random.parallel_sample_pytree(rng, struct)

    self.assertIsInstance(output, dict)
    self.assertEqual(output['a'].shape, (32, 32))
    self.assertEqual(output['a'].sharding, sharding_a)
    self.assertEqual(output['b'].shape, (16,))
    self.assertEqual(output['b'].sharding, sharding_b)


if __name__ == '__main__':
  absltest.main()

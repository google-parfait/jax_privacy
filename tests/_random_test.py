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

from unittest import mock

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

  @parameterized.parameters(1, 8, 31, 64, 128)
  def test_secure_seed_bit_width(self, bits):
    seed = _random.secure_seed(bits)
    self.assertIsInstance(seed, int)
    self.assertGreaterEqual(seed, 0)
    self.assertLess(seed, 2**bits)

  @parameterized.parameters(0, -5)
  def test_secure_seed_invalid_bits(self, bits):
    with self.assertRaises(ValueError):
      _random.secure_seed(bits)

  def test_secure_seed_multihost_broadcast(self):
    expected = (2**100 + 7).to_bytes(16, 'big')
    arr = np.frombuffer(expected, dtype=np.uint8)
    with mock.patch.object(
        _random.multihost_utils, 'broadcast_one_to_all', return_value=arr
    ) as mock_bcast:
      self.assertEqual(_random.secure_seed(128), 2**100 + 7)
      mock_bcast.assert_called_once()

  def test_default_rng_preserves_generator_identity(self):
    rng = np.random.default_rng(42)
    self.assertIs(_random.default_rng(rng), rng)

  def test_default_rng_none_broadcasts_seed(self):
    with mock.patch.object(_random, 'secure_seed', return_value=12345):
      rng = _random.default_rng(None)
    expected = np.random.default_rng(12345)
    self.assertEqual(rng.integers(2**31), expected.integers(2**31))

  def test_default_prng_key_resolves_all_types(self):
    expected_7 = jax.random.key(int(np.random.default_rng(7).integers(2**31)))
    self.assertTrue(jnp.array_equal(_random.default_prng_key(7), expected_7))

    rng = np.random.default_rng(99)
    rng_key = _random.default_prng_key(rng)
    expected = jax.random.key(int(np.random.default_rng(99).integers(2**31)))
    self.assertTrue(jnp.array_equal(rng_key, expected))

    with mock.patch.object(_random, 'secure_seed', return_value=777):
      none_key = _random.default_prng_key(None)
    expected_none = jax.random.key(
        int(np.random.default_rng(777).integers(2**31))
    )
    self.assertTrue(jnp.array_equal(none_key, expected_none))

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

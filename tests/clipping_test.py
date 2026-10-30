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

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import dp_accounting
import jax
import jax.numpy as jnp
from jax_privacy import clipping
import numpy as np
import optax


def cartesian_product(**kwargs):
  return [dict(zip(kwargs, v)) for v in itertools.product(*kwargs.values())]


PYTREE_STRUCTS = [
    jax.ShapeDtypeStruct(shape=(5, 5), dtype=jnp.float32),
    {
        'a': jax.ShapeDtypeStruct(shape=(10,), dtype=jnp.float16),
        'b': jax.ShapeDtypeStruct(shape=(5, 5), dtype=jnp.float32),
        'c': jax.ShapeDtypeStruct(shape=(), dtype=jnp.bfloat16),
    },
]


class ClipPyTreeTest(parameterized.TestCase):

  @parameterized.parameters(
      *cartesian_product(
          pytree=PYTREE_STRUCTS,
          clip_norm=[0.0, 1.0, 2.0, jnp.inf],
          rescale_to_unit_norm=[True, False],
      )
  )
  def test_clip_pytree_output_dtype_matches_input(self, pytree, **kwargs):
    pytree = optax.tree.random_like(jax.random.key(0), pytree)
    clipped, _ = clipping.clip_pytree(pytree, **kwargs)

    jitted = jax.jit(clipping.clip_pytree, static_argnums=(2,))
    clipped2, _ = jitted(pytree, **kwargs)
    chex.assert_trees_all_equal_shapes_and_dtypes(pytree, clipped, clipped2)

  @parameterized.parameters(
      *cartesian_product(
          pytree=PYTREE_STRUCTS,
          clip_norm=[0.0, 1.0, 2.0, jnp.inf],
          rescale_to_unit_norm=[True, False],
      )
  )
  def test_clip_pytree_has_bounded_norm(self, pytree, **kwargs):
    pytree = optax.tree.random_like(jax.random.key(0), pytree)
    clipped, _ = clipping.clip_pytree(pytree, **kwargs)

    # Float16/bfloat16 rounding can exceed the bound by ~1 ULP.
    atol = max(float(jnp.finfo(x.dtype).eps) for x in jax.tree.leaves(clipped))
    if kwargs['rescale_to_unit_norm']:
      self.assertLessEqual(optax.tree.norm(clipped), 1.0 + atol)
    else:
      self.assertLessEqual(optax.tree.norm(clipped), kwargs['clip_norm'] + atol)

  @parameterized.parameters(*cartesian_product(pytree=PYTREE_STRUCTS))
  def test_clip_pytree_with_large_clip_norm(self, pytree):
    pytree = optax.tree.random_like(jax.random.key(0), pytree)
    clip_norm = optax.tree.norm(pytree) * 1.5
    clipped_tree, _ = clipping.clip_pytree(pytree, clip_norm)
    chex.assert_trees_all_close(clipped_tree, pytree, atol=1e-6)

  def test_clip_pytree_clip_norm_zero_rescale(self):
    """Tests clip_pytree when clip_norm is 0."""
    pytree = {'a': jnp.array([3.0, 0.0]), 'b': jnp.array([0.0, 4.0])}
    clip_norm = 0.0

    clipped, norm = clipping.clip_pytree(
        pytree, clip_norm, rescale_to_unit_norm=True
    )
    self.assertAlmostEqual(norm, 5.0)
    self.assertAlmostEqual(optax.tree.norm(clipped), 1.0)
    # Normalization is the limiting behavior as clip_norm -> 0.
    expected = {'a': jnp.array([0.6, 0.0]), 'b': jnp.array([0.0, 0.8])}
    chex.assert_trees_all_close(clipped, expected)

  @parameterized.parameters(
      *cartesian_product(pytree=PYTREE_STRUCTS, clip_norm=[0.0, 1.0, jnp.inf])
  )
  def test_clip_pytree_zero_norm_pytree(self, pytree, clip_norm):
    """Tests clip_pytree when the input pytree has zero norm."""
    zero_tree = optax.tree.zeros_like(pytree)

    clipped, norm = clipping.clip_pytree(
        zero_tree, clip_norm, rescale_to_unit_norm=False
    )
    self.assertAlmostEqual(norm, 0.0)
    chex.assert_trees_all_close(clipped, zero_tree)
    self.assertAlmostEqual(optax.tree.norm(clipped), 0.0)

    clipped_rescaled, norm_rescaled = clipping.clip_pytree(
        zero_tree, clip_norm, rescale_to_unit_norm=True
    )
    self.assertAlmostEqual(norm_rescaled, 0.0)
    chex.assert_trees_all_close(clipped_rescaled, zero_tree)
    self.assertAlmostEqual(optax.tree.norm(clipped_rescaled), 0.0)

  @parameterized.parameters([jnp.nan, jnp.inf, -jnp.inf])
  def test_clip_pytree_non_finite_input_returns_non_finite_norm(self, bad):
    """Output may be non-finite, but only when the returned norm is too."""
    pytree = jnp.array([3.0, 0.0, bad, 2.0])
    _, norm = clipping.clip_pytree(pytree, 1.0)
    self.assertFalse(jnp.isfinite(norm))

  @parameterized.parameters(
      *cartesian_product(
          clip_norm=[1e-5, 1.2e-5, 1e-6],
          dtype=[jnp.float16, jnp.bfloat16],
      )
  )
  def test_clip_pytree_float16_rescale_no_overflow(self, clip_norm, dtype):
    """Scale factor may exceed dtype max; output must still be bounded."""
    pytree = jnp.array([1e-5], dtype=dtype)
    clipped, norm = clipping.clip_pytree(
        pytree,
        clip_norm,
        rescale_to_unit_norm=True,
    )
    self.assertEqual(clipped.dtype, dtype)
    self.assertTrue(jnp.isfinite(norm))
    chex.assert_tree_all_finite(clipped)
    self.assertLessEqual(optax.tree.norm(clipped), 1.0)

  @parameterized.parameters([jnp.float16, jnp.bfloat16])
  def test_clip_pytree_low_precision_squared_underflow_clipped(self, dtype):
    """Clips small gradients when squared norm underflows in leaf dtype."""
    # In float16, (1e-4)**2 = 1e-8 underflows to 0.0.
    # Accumulating squared norms in float32 ensures the norm is correctly
    # computed as ~1e-4 rather than underflowing to 0.0 (which would leave the
    # gradient unclipped and violate sensitivity bounds).
    pytree = jnp.array([1e-4], dtype=dtype)
    clip_norm = 1e-5
    clipped, norm = clipping.clip_pytree(pytree, clip_norm=clip_norm)
    self.assertEqual(clipped.dtype, dtype)
    self.assertTrue(jnp.isfinite(norm))
    self.assertGreater(float(norm), clip_norm)
    # The gradient must be scaled down by clip_norm / norm ≈ 0.1 to ~1e-5.
    chex.assert_trees_all_close(
        clipped, jnp.array([1e-5], dtype=dtype), rtol=1e-2
    )
    atol = jnp.finfo(dtype).eps
    self.assertLessEqual(
        float(optax.tree.norm(clipped.astype(jnp.float32))),
        clip_norm * (1.0 + atol),
    )

  def test_clip_pytree_float16_sum_of_squares_overflow_finite_norm(self):
    """Sum of squares exceeds float16 max (65504); norm must remain finite."""
    # 100 elements of 100.0 in float16: each element is <= 65504, but sum of
    # squares is 100 * 10000 = 1,000,000 > 65504.
    pytree = jnp.full((100,), 100.0, dtype=jnp.float16)
    clip_norm = 2000.0
    clipped, norm = clipping.clip_pytree(pytree, clip_norm=clip_norm)
    self.assertEqual(clipped.dtype, jnp.float16)
    self.assertTrue(jnp.isfinite(norm))
    self.assertAlmostEqual(float(norm), 1000.0, places=1)
    chex.assert_tree_all_finite(clipped)
    chex.assert_trees_all_close(clipped, pytree)

  def test_clip_pytree_per_layer_full_clip_norm_tree(self):
    """Tests per-layer clipping with complete clip_norm tree."""
    pytree = {'a': np.array([3.0, 4.0]), 'b': np.array([0.0, 10.0])}
    clip_norm = {'a': np.array(1.0), 'b': np.array(15.0)}
    expected_output = {
        # 'a' base norm is 5.0. Limit is 1.0.
        # Scale = 1.0 / 5.0 = 0.2
        # Scaled output = [3.0, 4.0] * 0.2 = [0.6, 0.8]
        'a': np.array([0.6, 0.8]),
        # 'b' base norm is 10.0. Limit is 15.0 (Unclipped).
        # Scale = 1.0
        # Scaled output = [0.0, 10.0] * 1.0 = [0.0, 10.0]
        'b': np.array([0.0, 10.0]),
    }
    expected_norms = {
        'a': np.array(5.0, dtype=np.float32),
        'b': np.array(10.0, dtype=np.float32),
    }
    clipped_output, norms = clipping.clip_pytree(pytree, clip_norm)
    chex.assert_trees_all_close(clipped_output, expected_output)
    chex.assert_trees_all_close(norms, expected_norms)

  def test_clip_pytree_per_layer_zero_clip_norm_tree(self):
    """Tests per-layer clipping with zero clip_norm tree."""
    pytree = {
        'layer1': {'w': np.array([0.0, 0.0]), 'b': np.array([0.0])},
    }
    clip_norm = {'layer1': np.array(2.0)}
    expected_output = {
        'layer1': {
            # 'layer1' base norm = 0.0; sqrt(0^2 + 0^2 + 0^2).
            'w': np.array([0.0, 0.0]),
            'b': np.array([0.0]),
        },
    }
    expected_norms = {
        'layer1': np.array(0.0, dtype=np.float32),
    }
    clipped_output, norms = clipping.clip_pytree(pytree, clip_norm)
    chex.assert_trees_all_close(clipped_output, expected_output)
    chex.assert_trees_all_close(norms, expected_norms)

  def test_clip_pytree_per_layer_tuple_nodes_full_norm_tree(self):
    """Tests per-layer clipping with pytree containing tuple leaves."""
    pytree = {'layer1': (np.array([3.0, 4.0]), np.array([0.0, 10.0]))}
    clip_norm = {'layer1': (np.array(1.0), np.array(15.0))}
    expected_output = {
        'layer1': (
            # Element 0: base norm is 5.0. Limit is 1.0.
            # Scale = 1.0 / 5.0 = 0.2
            # Scaled output = [3.0, 4.0] * 0.2 = [0.6, 0.8]
            np.array([0.6, 0.8]),
            # Element 1: base norm is 10.0. Limit is 15.0 (Unclipped).
            # Scale = 1.0
            # Scaled output = [0.0, 10.0] * 1.0 = [0.0, 10.0]
            np.array([0.0, 10.0]),
        )
    }
    expected_norms = {
        'layer1': (
            np.array(5.0, dtype=np.float32),
            np.array(10.0, dtype=np.float32),
        )
    }
    clipped_output, norms = clipping.clip_pytree(pytree, clip_norm)
    chex.assert_trees_all_close(clipped_output, expected_output)
    chex.assert_trees_all_close(norms, expected_norms)

  def test_clip_pytree_per_layer_prefix_clip_norm_tree(self):
    """Tests per-layer clipping with clip_norm as prefix of pytree."""
    pytree = {
        'layer1': {'w': np.array([0.0, 3.0]), 'b': np.array([4.0])},
        'layer2': {'w': np.array([0.0, 8.0]), 'b': np.array([6.0])},
    }
    clip_norm = {'layer1': np.array(2.0), 'layer2': np.array(5.0)}
    expected_output = {
        'layer1': {
            # 'layer1' base norm = 5.0; sqrt(0^2 + 3^2 + 4^2).
            # Limit is 2.0. Scale = 2.0 / 5.0 = 0.4
            'w': np.array([0.0, 1.2]),
            'b': np.array([1.6]),
        },
        'layer2': {
            # 'layer2' base norm = 10.0; sqrt(0^2 + 8^2 + 6^2).
            # Limit is 5.0. Scale = 5.0 / 10.0 = 0.5
            'w': np.array([0.0, 4.0]),
            'b': np.array([3.0]),
        },
    }
    expected_norms = {
        'layer1': np.array(5.0, dtype=np.float32),
        'layer2': np.array(10.0, dtype=np.float32),
    }
    clipped_output, norms = clipping.clip_pytree(pytree, clip_norm)
    chex.assert_trees_all_close(clipped_output, expected_output)
    chex.assert_trees_all_close(norms, expected_norms)


class ClipAndRoundToGridTest(parameterized.TestCase):

  @parameterized.parameters(
      (1.0, 1000, 100),
      (2.0, 500, 10000),
  )
  def test_adjusted_clip_norm(self, l2_clip_norm, grid_scale, num_params):
    adj = clipping._adjusted_clip_norm(l2_clip_norm, grid_scale, num_params)
    self.assertLess(adj, l2_clip_norm)
    self.assertGreater(adj, 0.0)

  def test_adjusted_clip_norm_too_small_grid_raises(self):
    with self.assertRaises(ValueError):
      # Grid scale 1 with 10000 params has max rounding
      # error 100 * 1.0 / 1 / 2 = 50 > 1.0
      clipping._adjusted_clip_norm(
          l2_clip_norm=1.0, grid_scale=1, num_params=10000
      )

  def test_clip_and_round_to_grid_shapes_and_types(self):
    grad = {
        'w': jnp.array([0.1, -0.3, 0.5]),
        'b': jnp.array(0.05),
    }
    with jax.enable_x64(True):
      rounded, _ = clipping.clip_and_round_to_grid(grad, 1.0, 1000)

      self.assertEqual(rounded['w'].dtype, jnp.int64)
      self.assertEqual(rounded['b'].dtype, jnp.int64)
      self.assertEqual(rounded['w'].shape, (3,))
      self.assertEqual(rounded['b'].shape, ())

  @parameterized.parameters(
      (1.0, 1000, 100),
      (5.0, 500, 1000),
      (0.5, 2000, 50),
  )
  def test_clipping_bound_obeyed_after_rounding(
      self, l2_clip_norm, grid_scale, num_params
  ):
    with jax.enable_x64(True):
      # Test 1: A uniform vector pointing in (1, 1, ..., 1) with
      # norm l2_clip_norm > adj_clip.
      grad_ones = jnp.ones(num_params) * (l2_clip_norm / np.sqrt(num_params))
      rounded_ones, _ = clipping.clip_and_round_to_grid(
          grad_ones, l2_clip_norm, grid_scale
      )
      self.assertLessEqual(optax.tree.norm(rounded_ones), grid_scale)
      self.assertLessEqual(
          optax.tree.norm(rounded_ones) * (l2_clip_norm / grid_scale),
          l2_clip_norm,
      )

      # Test 2: Random normal vectors with large norms.
      rng = np.random.default_rng(42)
      for _ in range(5):
        raw = rng.standard_normal(num_params)
        grad_random = jnp.array(raw / np.linalg.norm(raw) * l2_clip_norm * 2.0)
        rounded_random, _ = clipping.clip_and_round_to_grid(
            grad_random, l2_clip_norm, grid_scale
        )
        self.assertLessEqual(optax.tree.norm(rounded_random), grid_scale)
        self.assertLessEqual(
            optax.tree.norm(rounded_random) * (l2_clip_norm / grid_scale),
            l2_clip_norm,
        )

  def test_clip_and_round_to_grid_raises_without_x64(self):
    """Tests that clip_and_round_to_grid raises if x64 is disabled."""
    grad = jnp.array([0.1, -0.3, 0.5])
    with jax.enable_x64(False), self.assertRaises(ValueError):
      clipping.clip_and_round_to_grid(grad, 1.0, 1000)


class ClippedFunGridScaleTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='without_x64',
          enable_x64=False,
          extra_kwargs={},
      ),
      dict(
          testcase_name='incompatible_with_rescale',
          enable_x64=True,
          extra_kwargs={'rescale_to_unit_norm': True},
      ),
      dict(
          testcase_name='incompatible_with_normalize_by',
          enable_x64=True,
          extra_kwargs={'normalize_by': 100.0},
      ),
      dict(
          testcase_name='incompatible_with_pytree_clip_norm',
          enable_x64=True,
          extra_kwargs={'l2_clip_norm': {'a': np.array(1.0)}},
      ),
  )
  def test_clipped_fun_grid_scale_raises(self, enable_x64, extra_kwargs):
    """Tests that clipped_fun with grid_scale raises on invalid configs."""
    kwargs = dict(
        fun=lambda x: x,
        batch_argnums=0,
        l2_clip_norm=1.0,
        grid_scale=1000,
    )
    kwargs.update(extra_kwargs)
    with jax.enable_x64(enable_x64), self.assertRaises(ValueError):
      clipping.clipped_fun(**kwargs)

  def test_clipped_fun_grid_scale_properties(self):
    """Tests dtype, sensitivity, and per-example norm bound with grid_scale."""
    with jax.enable_x64(True):
      # Check int64 output dtype and that explicit dtype is ignored.
      for test_dtype in (None, jnp.int32, jnp.float16):
        cf = clipping.clipped_fun(
            lambda x: x * 2.0,
            batch_argnums=0,
            keep_batch_dim=False,
            l2_clip_norm=1.0,
            grid_scale=1000,
            dtype=test_dtype,
        )
        data = jax.random.normal(jax.random.key(0), (10, 4))
        result = cf(data)
        for leaf in jax.tree.leaves(result):
          self.assertEqual(leaf.dtype, jnp.int64)

      # Check that explicit dtype (e.g. int32) does not truncate fractional
      # values, and float16 does not overflow large values before grid rounding.
      cf_int = clipping.clipped_fun(
          lambda x: x,
          batch_argnums=0,
          keep_batch_dim=False,
          l2_clip_norm=1.0,
          grid_scale=1000,
          dtype=jnp.int32,
      )
      chex.assert_trees_all_close(
          cf_int(jnp.array([[0.5, 0.5]])),
          jnp.array([500, 500], dtype=jnp.int64),
      )

      cf_f16 = clipping.clipped_fun(
          lambda x: x,
          batch_argnums=0,
          keep_batch_dim=False,
          l2_clip_norm=1.0,
          grid_scale=1000,
          dtype=jnp.float16,
      )
      chex.assert_trees_all_close(
          cf_f16(jnp.array([[100000.0]])),
          jnp.array([1000], dtype=jnp.int64),
      )

      # Check sensitivity equals grid_scale.
      cf2 = clipping.clipped_fun(
          lambda x: x,
          batch_argnums=0,
          keep_batch_dim=False,
          l2_clip_norm=2.0,
          grid_scale=500,
      )
      self.assertEqual(cf2.l2_norm_bound, 500.0)
      self.assertEqual(
          cf2.sensitivity(dp_accounting.NeighboringRelation.REPLACE_SPECIAL),
          500.0,
      )

      # Check per-example norm is bounded by grid_scale.
      grid_scale = 1000
      cf3 = clipping.clipped_fun(
          lambda x: x * 3.0,
          batch_argnums=0,
          keep_batch_dim=False,
          l2_clip_norm=1.0,
          grid_scale=grid_scale,
      )
      for i in range(10):
        data = jax.random.normal(jax.random.key(i), (1, 20))
        result = cf3(data)
        self.assertLessEqual(float(optax.tree.norm(result)), grid_scale)

  def test_clipped_grad_grid_scale(self):
    """Tests clipped_grad with grid_scale end-to-end."""

    def loss_fn(params, data):
      return jnp.mean((data - params) ** 2)

    with jax.enable_x64(True):
      gf = clipping.clipped_grad(
          loss_fn,
          l2_clip_norm=1.0,
          batch_argnums=1,
          grid_scale=1000,
      )
      params = jnp.zeros(4)
      data = jax.random.normal(jax.random.key(0), (10, 4))
      result = gf(params, data)
      for leaf in jax.tree.leaves(result):
        self.assertEqual(leaf.dtype, jnp.int64)
      self.assertEqual(gf.l2_norm_bound, 1000.0)


def single_example_fn(data):
  assert data.ndim == 1
  return data + 2


def batched_fn(data):
  assert data.ndim == 2
  return data.mean(axis=0)


CLIP_SUM_INPUTS = [
    {'fun': single_example_fn, 'batch_argnums': 0, 'keep_batch_dim': False},
    {'fun': batched_fn, 'batch_argnums': 0},
]


class ClipTransformTest(parameterized.TestCase):

  @parameterized.parameters(CLIP_SUM_INPUTS)
  def test_clip_sum_sensitivity(self, **kwargs):
    data = jax.random.normal(jax.random.key(0), (15, 4))
    sum_clip_mean = clipping.clipped_fun(**kwargs)

    neighbor = data.at[3].set([-10, 10, 20, 20])
    diff = jnp.linalg.norm(sum_clip_mean(data) - sum_clip_mean(neighbor))
    relation = dp_accounting.NeighboringRelation.REPLACE_ONE
    self.assertLessEqual(diff, sum_clip_mean.sensitivity(relation) + 1e-6)

    is_padding_example_nbr = jnp.zeros(data.shape[0], jnp.bool_).at[3].set(True)
    relation = dp_accounting.NeighboringRelation.REPLACE_SPECIAL
    new = sum_clip_mean(data, is_padding_example=is_padding_example_nbr)
    diff = jnp.linalg.norm(sum_clip_mean(data) - new)
    self.assertLessEqual(diff, sum_clip_mean.sensitivity(relation) + 1e-6)

    neighbor = data[1:]
    relation = dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE
    diff = jnp.linalg.norm(sum_clip_mean(data) - sum_clip_mean(neighbor))
    self.assertLessEqual(diff, sum_clip_mean.sensitivity(relation) + 1e-6)

  @parameterized.parameters(CLIP_SUM_INPUTS)
  def test_nan_safe_by_default(self, **kwargs):
    data = jax.random.normal(jax.random.key(0), (15, 4))
    data = data.at[5].set(jnp.nan)
    sum_clip_mean = clipping.clipped_fun(**kwargs)

    neighbor = data.at[3].set([-10, 10, 20, 20])
    diff = jnp.linalg.norm(sum_clip_mean(data) - sum_clip_mean(neighbor))
    relation = dp_accounting.NeighboringRelation.REPLACE_ONE
    self.assertLessEqual(diff, sum_clip_mean.sensitivity(relation) + 1e-6)

  def test_return_norms_without_aux_matches_documented_signature(self):

    def fun(x):
      return x**2

    sum_clip = clipping.clipped_fun(
        fun,
        batch_argnums=0,
        keep_batch_dim=False,
        l2_clip_norm=jnp.inf,
        return_norms=True,
    )

    value, norms = sum_clip(jnp.array([1.0, 2.0, 3.0]))
    chex.assert_trees_all_close(value, 14.0)
    chex.assert_trees_all_close(norms, jnp.array([1.0, 4.0, 9.0]))

  @parameterized.parameters([None, 1, 2])
  def test_padding_examples_mask_aux_and_norms_with_nan(self, microbatch_size):
    def fun(x):
      aux = {
          'double': 2.0 * x,
          'vec': jnp.stack([x, -x]),
          'none_val': None,
      }
      return x**2, aux

    sum_clip = clipping.clipped_fun(
        fun,
        has_aux=True,
        batch_argnums=0,
        keep_batch_dim=False,
        l2_clip_norm=jnp.inf,
        return_norms=True,
        microbatch_size=microbatch_size,
    )
    data = jnp.array([1.0, 2.0, 3.0, 4.0])
    is_padding = jnp.array([False, True, True, True])
    value, (aux, norms) = sum_clip(data, is_padding_example=is_padding)

    chex.assert_trees_all_close(value, 1.0)
    np.testing.assert_allclose(norms[0], 1.0)
    self.assertTrue(np.all(np.isnan(norms[1:])))
    np.testing.assert_allclose(aux['double'][0], 2.0)
    self.assertTrue(np.all(np.isnan(aux['double'][1:])))
    np.testing.assert_allclose(aux['vec'][0], np.array([1.0, -1.0]))
    self.assertTrue(np.all(np.isnan(aux['vec'][1:])))
    self.assertIsNone(aux['none_val'])

  @parameterized.product(
      arg_dtype=[jnp.float16, jnp.bfloat16, jnp.float32],
      output_dtype=[jnp.float32, None],
      microbatch_size=[3, None],
  )
  def test_correct_dtype(self, arg_dtype, output_dtype, microbatch_size):
    data = jax.random.normal(jax.random.key(0), (15, 4), arg_dtype)
    sum_clip_mean = clipping.clipped_fun(
        **CLIP_SUM_INPUTS[0],
        dtype=output_dtype,
        microbatch_size=microbatch_size,
    )
    value = sum_clip_mean(data)
    expected_dtype = output_dtype or arg_dtype
    self.assertEqual(value.dtype, expected_dtype)

  def test_aux_shape_keep_batch_dim(self):
    def fun(batched_data):
      # With keep_batch_dim=True, data shape inside vmap is (1, D)
      scalar = jnp.sum(batched_data)
      array_1d = batched_data[:, 0, 0, 0]
      array_2d = batched_data[:, :, 0, 0]
      array_3d = batched_data[:, :, :, 1]
      return scalar, (scalar, array_1d, array_2d, array_3d, batched_data)

    data = jnp.zeros((15, 4, 9, 10))

    sum_clip_mean = clipping.clipped_fun(
        fun, has_aux=True, keep_batch_dim=True, batch_argnums=0
    )
    _, (aux_scalar, aux_1d, aux_2d, aux_3d, aux_4d) = sum_clip_mean(data)

    self.assertEqual(aux_scalar.shape, (15,))
    # In assertions below dimension of size 1 is dropped.
    self.assertEqual(aux_1d.shape, (15,))
    self.assertEqual(aux_2d.shape, (15, 4))
    self.assertEqual(aux_3d.shape, (15, 4, 9))
    self.assertEqual(aux_4d.shape, (15, 4, 9, 10))

  @parameterized.parameters(
      *cartesian_product(
          bad_value=[jnp.nan, jnp.inf, -jnp.inf],
          clip_norm=[0.0, 1.0],
          rescale_to_unit_norm=[True, False],
      )
  )
  def test_nan_safe_norm_check_produces_finite_output(
      self, bad_value, clip_norm, rescale_to_unit_norm
  ):
    """nan_safe zeros NaN/inf/overflow examples and obeys sensitivity."""
    data = jax.random.normal(jax.random.key(0), (8, 4))
    data_with_bad = data.at[3].set(bad_value)

    cf = clipping.clipped_fun(
        single_example_fn,
        batch_argnums=0,
        keep_batch_dim=False,
        l2_clip_norm=clip_norm,
        rescale_to_unit_norm=rescale_to_unit_norm,
        nan_safe=True,
    )

    # Clipped output should be finite even if input is not.
    chex.assert_tree_all_finite(cf(data_with_bad))

    # Clipped value for only the bad example should be zero.
    chex.assert_trees_all_close(cf(data_with_bad[3:4]), 0)

  def test_nan_safe_float16_sum_of_squares_overflow_not_zeroed(self):
    """Large float16 gradients whose sum of squares > 65504 are not zeroed."""
    # Gradient sum of squares = 100 * 10000 = 1,000,000 > 65504.
    # Accumulating squared norms in float32 prevents the norm from overflowing
    # to inf, ensuring nan_safe does not zero out valid finite gradients.
    data = jnp.full((2, 100), 100.0, dtype=jnp.float16)
    cf = clipping.clipped_fun(
        lambda x: x,
        batch_argnums=0,
        keep_batch_dim=False,
        l2_clip_norm=2000.0,
        nan_safe=True,
    )
    result = cf(data)
    chex.assert_tree_all_finite(result)
    expected = jnp.full((100,), 200.0, dtype=jnp.float16)
    chex.assert_trees_all_close(result, expected)


class ClippedFunPerLayerTest(parameterized.TestCase):

  def test_clip_per_layer_clipped_fun_full_norm_tree(self):
    """Tests clipped_fun with norm tree having same structure as pytree."""
    pytree = {'a': np.array([3.0, 4.0]), 'b': np.array([0.0, 10.0])}
    clip_norm = {'a': np.array(1.0), 'b': np.array(15.0)}
    expected_output = {
        # 'a' base norm is 5.0.
        # Scales = 1.0/5.0 (1x), 1.0/10.0 (2x), 1.0/15.0 (3x).
        # The scaled output is always [0.6, 0.8]. Sum: 3 * [0.6, 0.8]
        'a': np.array([1.8, 2.4]),
        # 'b' base norm is 10.0.
        # Batch 1 (norm 10): Unclipped. Scale = 1.0
        # Batch 2 (norm 20): Clipped. Scale = 15.0 / 20.0 = 0.75
        # Batch 3 (norm 30): Clipped. Scale = 15.0 / 30.0 = 0.5
        # Sum: ([0, 10] * 1.0) + ([0, 20] * 0.75) + ([0, 30] * 0.5)
        'b': np.array([0.0, 40.0]),
    }
    expected_norms = {
        'a': np.array([5.0, 10.0, 15.0]),
        'b': np.array([10.0, 20.0, 30.0]),
    }
    cf = clipping.clipped_fun(
        lambda x: jax.tree.map(lambda leaf: leaf * x, pytree),
        batch_argnums=0,
        keep_batch_dim=False,
        l2_clip_norm=clip_norm,
        rescale_to_unit_norm=False,
        return_norms=True,
    )
    cf_output, cf_norms = cf(jnp.array([1.0, 2.0, 3.0]))
    chex.assert_trees_all_close(cf_output, expected_output)
    chex.assert_trees_all_close(cf_norms, expected_norms)

  def test_clip_per_layer_clipped_fun_prefix_norm_tree(self):
    """Tests clipped_fun with norm tree as prefix of pytree."""
    pytree = {
        'layer1': {'w': np.array([0.0, 3.0]), 'b': np.array([4.0])},
        'layer2': {'w': np.array([0.0, 8.0]), 'b': np.array([6.0])},
    }
    clip_norm = {'layer1': np.array(2.0), 'layer2': np.array(5.0)}
    expected_output = {
        'layer1': {
            # 'layer1' base norm = 5.0; sqrt(0^2 + 3^2 + 4^2).
            # Batch 1 (norm 5.0): Clipped. Scale = 2.0 / 5.0
            # Batch 2 (norm 10.0): Clipped. Scale = 2.0 / 10.0
            # Batch 3 (norm 15.0): Clipped. Scale = 2.0 / 15.0
            # Sum of 3 scaled batches = 3 * (2.0 / 5.0) = 1.2
            'w': np.array([0.0, 3.0]) * 1.2,
            'b': np.array([4.0]) * 1.2,
        },
        'layer2': {
            # 'layer2' base norm = 10.0; sqrt(0^2 + 8^2 + 6^2)).
            # Batch 1 (norm 10.0): Clipped. Scale = 5.0 / 10.0
            # Batch 2 (norm 20.0): Clipped. Scale = 5.0 / 20.0
            # Batch 3 (norm 30.0): Clipped. Scale = 5.0 / 30.0
            # Sum of 3 scaled batches = 3 * (5.0 / 10.0) = 1.5
            'w': np.array([0.0, 8.0]) * 1.5,
            'b': np.array([6.0]) * 1.5,
        },
    }
    expected_norms = {
        'layer1': np.array([5.0, 10.0, 15.0], dtype=np.float32),
        'layer2': np.array([10.0, 20.0, 30.0], dtype=np.float32),
    }
    cf = clipping.clipped_fun(
        lambda x: jax.tree.map(lambda leaf: leaf * x, pytree),
        batch_argnums=0,
        keep_batch_dim=False,
        l2_clip_norm=clip_norm,
        rescale_to_unit_norm=False,
        return_norms=True,
    )
    cf_output, cf_norms = cf(jnp.array([1.0, 2.0, 3.0]))
    chex.assert_trees_all_close(cf_output, expected_output)
    chex.assert_trees_all_close(cf_norms, expected_norms)


class ClippedFunSensitivityContractTest(parameterized.TestCase):
  """Regression tests for clipped_fun sensitivity accounting.

  The Formal Guarantees section points callers at ``.sensitivity()`` rather
  than deriving bounds from kwargs. These tests lock the contract that
  ``normalize_by`` and per-layer clipping change ``l2_norm_bound``.
  """

  def test_normalize_by_scales_sensitivity(self):
    cf = clipping.clipped_fun(
        jnp.square,
        batch_argnums=0,
        keep_batch_dim=False,
        l2_clip_norm=2.0,
        rescale_to_unit_norm=False,
        normalize_by=4.0,
    )
    self.assertAlmostEqual(cf.l2_norm_bound, 2.0 / 4.0)
    self.assertAlmostEqual(
        cf.sensitivity(dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE),
        0.5,
    )
    self.assertAlmostEqual(
        cf.sensitivity(dp_accounting.NeighboringRelation.REPLACE_ONE),
        1.0,
    )

  def test_rescale_to_unit_norm_ignores_clip_magnitude(self):
    cf_small = clipping.clipped_fun(
        jnp.square,
        batch_argnums=0,
        keep_batch_dim=False,
        l2_clip_norm=0.25,
        rescale_to_unit_norm=True,
        normalize_by=1.0,
    )
    cf_large = clipping.clipped_fun(
        jnp.square,
        batch_argnums=0,
        keep_batch_dim=False,
        l2_clip_norm=100.0,
        rescale_to_unit_norm=True,
        normalize_by=1.0,
    )
    self.assertAlmostEqual(cf_small.l2_norm_bound, 1.0)
    self.assertAlmostEqual(cf_large.l2_norm_bound, 1.0)

  def test_per_layer_rescale_uses_sqrt_num_leaves(self):
    clip_norm = {'a': 1.0, 'b': 2.0, 'c': 3.0}
    cf = clipping.clipped_fun(
        lambda x: {'a': x, 'b': x, 'c': x},
        batch_argnums=0,
        keep_batch_dim=False,
        l2_clip_norm=clip_norm,
        rescale_to_unit_norm=True,
        normalize_by=5.0,
    )
    self.assertAlmostEqual(cf.l2_norm_bound, np.sqrt(3.0) / 5.0)

  def test_per_layer_without_rescale_uses_leaf_norm(self):
    clip_norm = {'a': 3.0, 'b': 4.0}
    cf = clipping.clipped_fun(
        lambda x: {'a': x, 'b': x},
        batch_argnums=0,
        keep_batch_dim=False,
        l2_clip_norm=clip_norm,
        rescale_to_unit_norm=False,
        normalize_by=2.0,
    )
    self.assertAlmostEqual(cf.l2_norm_bound, 5.0 / 2.0)


class ClippedGradSensitivityContractTest(parameterized.TestCase):
  """Regression tests for clipped_grad sensitivity accounting.

  The Args section used to claim sensitivity is always ``1.0`` or
  ``l2_clip_norm``. That ignores ``normalize_by``.
  """

  def test_normalize_by_scales_sensitivity(self):
    def loss(params, data):
      return 0.5 * jnp.mean((data - params) ** 2)

    cg = clipping.clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=2.0,
        rescale_to_unit_norm=False,
        normalize_by=4.0,
    )
    self.assertAlmostEqual(cg.l2_norm_bound, 2.0 / 4.0)
    self.assertAlmostEqual(
        cg.sensitivity(dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE),
        0.5,
    )


if __name__ == '__main__':
  absltest.main()

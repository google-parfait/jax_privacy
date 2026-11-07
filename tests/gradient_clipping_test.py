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
from jax_privacy import clipping as gradient_clipping
import optax


def mean_quadratic_loss(params: jax.Array, x: jax.Array) -> jax.Array:
  """1/n sum( (params - x_i)**2 )."""
  assert params.ndim == 1, params.shape
  assert x.ndim == 2, x.shape
  return 0.5 * jnp.mean((params - x) ** 2)


def mean_quadratic_loss_with_aux(
    params: jax.Array, x: jax.Array
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """Quadratic loss with auxiliary data."""
  assert params.ndim == 1, params.shape
  assert x.ndim == 2, x.shape
  loss = 0.5 * jnp.mean((params - x) ** 2)
  aux = {'key': jnp.mean(x, axis=0)}
  return loss, aux


def mean_logistic_regression_loss(
    params: dict[str, jax.Array], inputs: jax.Array, targets: jax.Array
) -> jax.Array:
  """Simple function to exercise pytree params and two args with batch axis."""
  assert inputs.ndim == 2 and targets.ndim == 2
  logits = inputs @ params['weights'] + params['bias']
  log_probs = jax.nn.log_softmax(logits)
  return -jnp.mean(jnp.sum(log_probs * targets, axis=-1))


class GradientClippingTest(parameterized.TestCase):

  def test_validate_args_overlap(self):
    with self.assertRaisesRegex(ValueError, 'overlap'):
      gradient_clipping.clipped_grad(
          mean_quadratic_loss,
          l2_clip_norm=1.0,
          argnums=0,
          batch_argnums=0,
      )

  def test_validate_args_empty_batch(self):
    with self.assertRaisesRegex(ValueError, 'Batch Argnums must not be empty'):
      gradient_clipping.clipped_grad(
          mean_quadratic_loss,
          l2_clip_norm=1.0,
          argnums=0,
          batch_argnums=(),
      )

  def test_validate_args_inconsistent_batch_size(self):
    params = jnp.zeros(3)
    x1 = jnp.zeros((10, 3))
    x2 = jnp.zeros(5)
    with self.assertRaisesRegex(ValueError, 'same size along axis 0'):
      gradient_clipping.clipped_grad(
          mean_quadratic_loss,
          l2_clip_norm=1.0,
          argnums=0,
          batch_argnums=(1, 2),
      )(params, x1, x2)

  def test_sum_clipped_grad_basic(self):
    params = jnp.array([2.0])
    x = jnp.array([[1.0], [4.0], [7.0]])
    clip_norm = 2.5
    rescale = False

    clipped_grad_fn = gradient_clipping.clipped_grad(
        mean_quadratic_loss,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=clip_norm,
        rescale_to_unit_norm=rescale,
    )
    sum_grads = clipped_grad_fn(params, x)

    # gradients = 1, -2, -5
    expected_grad = jnp.array([-3.5])
    chex.assert_trees_all_close(sum_grads, expected_grad, atol=1e-6)

  @parameterized.parameters([jnp.float16, jnp.bfloat16])
  def test_sum_clipped_grad_float16(self, dtype):
    params = jnp.array([2.0], dtype=dtype)
    x = jnp.array([[1.0], [4.0], [7.0]], dtype=dtype)
    clip_norm = 2.5
    rescale = False

    clipped_grad_fn = jax.jit(
        gradient_clipping.clipped_grad(
            mean_quadratic_loss,
            argnums=0,
            batch_argnums=1,
            l2_clip_norm=clip_norm,
            rescale_to_unit_norm=rescale,
        )
    )
    sum_grads = clipped_grad_fn(params, x)

    # gradients = 1, -2, -5
    expected_grad = jnp.array([-3.5])
    chex.assert_trees_all_close(sum_grads, expected_grad, atol=1e-6)

  @parameterized.parameters([None, 1, 3])
  def test_sum_clipped_grad_with_rescale(self, microbatch_size):
    params = jnp.array([2.0])
    x = jnp.array([[1.0], [4.0], [7.0]])
    clip_norm = 2.5
    rescale = True

    clipped_grad_fn = gradient_clipping.clipped_grad(
        mean_quadratic_loss,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=clip_norm,
        rescale_to_unit_norm=rescale,
        microbatch_size=microbatch_size,
    )
    sum_grads = clipped_grad_fn(params, x)
    expected_grad = jnp.array([-3.5 / 2.5])
    chex.assert_trees_all_close(sum_grads, expected_grad, atol=1e-6)

  @parameterized.parameters([None, 1, 3])
  def test_sum_clipped_grad_with_padding_examples(self, microbatch_size):
    params = jnp.array([2.0])
    x = jnp.array([[1.0], [4.0], [7.0]])
    clip_norm = 2.5
    rescale = True

    clipped_grad_fn = jax.jit(
        gradient_clipping.clipped_grad(
            mean_quadratic_loss,
            argnums=0,
            batch_argnums=1,
            l2_clip_norm=clip_norm,
            rescale_to_unit_norm=rescale,
            microbatch_size=microbatch_size,
        )
    )
    sum_grads = clipped_grad_fn(
        params, x, is_padding_example=jnp.array([False, False, False])
    )
    expected_grad = jnp.array([-3.5 / 2.5])
    chex.assert_trees_all_close(sum_grads, expected_grad, atol=1e-6)

    sum_grads = clipped_grad_fn(
        params, x, is_padding_example=jnp.array([True, True, True])
    )
    expected_grad = jnp.array([0.0])
    chex.assert_trees_all_close(sum_grads, expected_grad, atol=1e-6)

    sum_grads = clipped_grad_fn(
        params, x, is_padding_example=jnp.array([False, True, True])
    )
    expected_grad = jnp.array([1.0 / 2.5])
    chex.assert_trees_all_close(sum_grads, expected_grad, atol=1e-6)

  @parameterized.parameters([None, 1, 2])
  def test_sum_clipped_grad_padding_masks_aux_outputs_with_nan(
      self, microbatch_size
  ):
    params = jnp.array([2.0])
    x = jnp.array([[1.0], [4.0], [7.0], [10.0]])
    is_padding = jnp.array([False, True, True, True])

    clipped_grad_fn = jax.jit(
        gradient_clipping.clipped_grad(
            mean_quadratic_loss_with_aux,
            argnums=0,
            has_aux=True,
            return_values=True,
            return_grad_norms=True,
            batch_argnums=1,
            l2_clip_norm=2.5,
            rescale_to_unit_norm=False,
            microbatch_size=microbatch_size,
        )
    )
    sum_grads, aux_output = clipped_grad_fn(
        params, x, is_padding_example=is_padding
    )
    chex.assert_trees_all_close(sum_grads, jnp.array([1.0]), atol=1e-6)
    chex.assert_trees_all_close(aux_output.values[0], 0.5, atol=1e-6)
    self.assertTrue(jnp.all(jnp.isnan(aux_output.values[1:])))
    chex.assert_trees_all_close(aux_output.grad_norms[0], 1.0, atol=1e-6)
    self.assertTrue(jnp.all(jnp.isnan(aux_output.grad_norms[1:])))
    chex.assert_trees_all_close(
        aux_output.aux['key'][0], jnp.array([1.0]), atol=1e-6
    )
    self.assertTrue(jnp.all(jnp.isnan(aux_output.aux['key'][1:])))

    # Also verify AuxiliaryOutput when optional fields (grad_norms, aux) are
    # None.
    values_only_fn = jax.jit(
        gradient_clipping.clipped_grad(
            mean_quadratic_loss,
            argnums=0,
            has_aux=False,
            return_values=True,
            return_grad_norms=False,
            batch_argnums=1,
            l2_clip_norm=2.5,
            rescale_to_unit_norm=False,
            microbatch_size=microbatch_size,
        )
    )
    _, values_only_aux = values_only_fn(
        params, x, is_padding_example=is_padding
    )
    chex.assert_trees_all_close(values_only_aux.values[0], 0.5, atol=1e-6)
    self.assertTrue(jnp.all(jnp.isnan(values_only_aux.values[1:])))
    self.assertIsNone(values_only_aux.grad_norms)
    self.assertIsNone(values_only_aux.aux)

  @parameterized.parameters([None, 1, 3])
  def test_sum_clipped_grad_has_aux(self, microbatch_size):
    params = jnp.array([2.0])
    x = jnp.array([[1.0], [4.0], [7.0]])
    clip_norm = 10.0

    clipped_grad_fn = gradient_clipping.clipped_grad(
        mean_quadratic_loss_with_aux,
        argnums=0,
        has_aux=True,
        batch_argnums=1,
        l2_clip_norm=clip_norm,
        microbatch_size=microbatch_size,
    )
    output = clipped_grad_fn(params, x)
    sum_grads, aux_output = output
    actual_aux = jax.tree.map(lambda x: x.mean(axis=0), aux_output.aux)

    grad_fn = jax.grad(mean_quadratic_loss_with_aux, has_aux=True)
    expected_grad, expected_aux = grad_fn(params, x)
    chex.assert_trees_all_close(sum_grads / x.shape[0], expected_grad)
    chex.assert_trees_all_close(actual_aux, expected_aux)

  @parameterized.parameters([None, 1, 3])
  def test_sum_clipped_grad_return_metrics(self, microbatch_size):
    params = jnp.array([2.0])
    x = jnp.array([[1.0], [4.0], [7.0]])
    clip_norm = 10.0

    clipped_grad_fn = gradient_clipping.clipped_grad(
        mean_quadratic_loss,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=clip_norm,
        rescale_to_unit_norm=False,
        return_grad_norms=True,
        microbatch_size=microbatch_size,
    )
    output = clipped_grad_fn(params, x)
    sum_grads, aux_output = output
    grad_norms = aux_output.grad_norms

    expected_grad = jnp.array([1 - 2 - 5])
    chex.assert_trees_all_close(sum_grads, expected_grad)

    expected_norms = jnp.array([1, 2, 5])
    chex.assert_trees_all_close(grad_norms, expected_norms)

  @parameterized.parameters([None, 1, 2])
  def test_sum_clipped_grad_leaf_scales(self, microbatch_size):
    params = {'w': jnp.array([2.0]), 'b': jnp.array([1.0])}
    x = jnp.array([[1.0], [2.0]])

    def scaled_loss(p, data):
      pred = p['w'] * data + p['b']
      return jnp.sum((pred - data) ** 2)

    def leaf_scale_transform(grad):
      leaf_scales = {'w': jnp.array([0.1]), 'b': jnp.array([1.0])}
      return jax.tree.map(jnp.multiply, leaf_scales, grad)

    clip_norm = 1.0

    clipped_grad_fn = gradient_clipping.clipped_grad(
        scaled_loss,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=clip_norm,
        rescale_to_unit_norm=False,
        pre_clipping_transform=leaf_scale_transform,
        microbatch_size=microbatch_size,
    )
    sum_grads = clipped_grad_fn(params, x)

    expected_w = (0.4 / jnp.sqrt(0.4**2 + 4**2)) + (
        1.2 / jnp.sqrt(1.2**2 + 6**2)
    )
    expected_b = (4.0 / jnp.sqrt(0.4**2 + 4**2)) + (
        6.0 / jnp.sqrt(1.2**2 + 6**2)
    )
    chex.assert_trees_all_close(
        sum_grads,
        {'w': jnp.array([expected_w]), 'b': jnp.array([expected_b])},
        atol=1e-5,
    )

  @parameterized.parameters([None, 1, 2])
  def test_sum_clipped_grad_extra_batch_axis(self, microbatch_size):
    params = jnp.array([2.0])
    # x has shape (num_users, num_examples_per_user, example_length)
    x = jnp.array([[[1.0], [2.0], [3.0]], [[3.0], [4.0], [5.0]]])
    num_users = x.shape[0]
    self.assertEqual(num_users, 2)
    clip_norm = 1000
    rescale = False

    clipped_grad_fn = gradient_clipping.clipped_grad(
        mean_quadratic_loss_with_aux,
        has_aux=True,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=clip_norm,
        rescale_to_unit_norm=rescale,
        microbatch_size=microbatch_size,
        keep_batch_dim=False,
    )
    output = clipped_grad_fn(params, x)
    sum_grads, _ = output
    chex.assert_shape(sum_grads, (1,))
    expected_gradient = jax.grad(mean_quadratic_loss)(params, x.reshape(6, 1))
    # sum_grads is a sum over users, so we divide by num_users here.
    actual_gradient = sum_grads / num_users
    chex.assert_trees_all_close(actual_gradient, expected_gradient)

  def test_values_and_sum_per_example_clipped_grads(self):
    params = jnp.array([2.0])
    x = jnp.array([[1.0], [4.0], [7.0]])
    clip_norm = 10.0

    output = gradient_clipping.clipped_grad(
        mean_quadratic_loss_with_aux,
        argnums=0,
        has_aux=True,
        batch_argnums=1,
        l2_clip_norm=clip_norm,
        rescale_to_unit_norm=False,
        return_values=True,
    )(params, x)
    sum_grads, aux_output = output
    value = aux_output.values.mean(axis=0)
    aux = jax.tree.map(lambda x: x.mean(axis=0), aux_output.aux)

    expected_value, expected_aux = mean_quadratic_loss_with_aux(params, x)
    self.assertAlmostEqual(value, expected_value)
    chex.assert_trees_all_close(aux, expected_aux)

    expected_grad = jnp.array([1 - 2 - 5])
    chex.assert_trees_all_close(sum_grads, expected_grad)

  def test_values_and_sum_per_example_clipped_grads_no_aux(self):
    params = jnp.array([2.0])
    x = jnp.array([[1.0], [4.0], [7.0]])
    clip_norm = 10.0

    output = gradient_clipping.clipped_grad(
        mean_quadratic_loss,
        argnums=0,
        has_aux=False,
        batch_argnums=1,
        l2_clip_norm=clip_norm,
        rescale_to_unit_norm=False,
        return_values=True,
    )(params, x)
    sum_grads, aux_output = output

    chex.assert_trees_all_close(aux_output.values, jnp.array([0.5, 2.0, 12.5]))

    expected_grad = jnp.array([1 - 2 - 5])
    chex.assert_trees_all_close(sum_grads, expected_grad)

  def test_sum_clipped_grad_pytree_params_and_data(self):
    params = {'weights': jnp.array([[1.0], [2.0]]), 'bias': jnp.array([0.5])}
    inputs = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    targets = jnp.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    clip_norm = 1.0

    clipped_grad_fn = gradient_clipping.clipped_grad(
        mean_logistic_regression_loss,
        argnums=0,
        batch_argnums=(1, 2),
        l2_clip_norm=clip_norm,
        rescale_to_unit_norm=False,
    )
    sum_grads = jax.jit(clipped_grad_fn)(params, inputs, targets)
    chex.assert_trees_all_equal_structs(sum_grads, params)

  def test_sum_clipped_grad_prng_key(self):
    def fun(params, data, key):
      deterministic_output = params['w'] * data + params['b']
      noise = jax.random.normal(key, shape=deterministic_output.shape)
      combined_output = deterministic_output + noise
      return jnp.mean(combined_output)

    key = jax.random.key(0)
    params = {'w': 2.5, 'b': 1.0}
    data = jnp.array([10.0, 20.0, 30.0])

    clipped_grad_fn = gradient_clipping.clipped_grad(
        fun,
        l2_clip_norm=1.0,
        prng_argnum=2,
    )
    sum_grads = clipped_grad_fn(params, data, key)
    chex.assert_trees_all_equal_structs(sum_grads, params)


class SlackClippingTest(parameterized.TestCase):

  @staticmethod
  def _linear_loss(params, example):
    return jnp.sum(params * example)

  @staticmethod
  def _pytree_linear_loss(params, example):
    return sum(jnp.sum(params[key] * example[key]) for key in params)

  @parameterized.parameters(
      (0.0, 2.0, 4, False, [1.0, 1.0, 1.0, 1.0]),
      (0.75, 1.0, 10, False, [1.0, 1.0, 0.5] + [0.0] * 7),
      (1.5, 2.0, 4, True, [1.0, 0.0, 0.0, 0.0]),
      (3.0, 2.0, 4, False, [0.0, 0.0, 0.0, 0.0]),
      (0.0, 0.0, 4, False, [0.0, 0.0, 0.0, 0.0]),
  )
  def test_slack_from_norm(
      self, norm, clip_norm, slack, rescale, expected_fill
  ):
    output_bound = 1.0 if rescale else clip_norm
    slack_vector = gradient_clipping._slack_from_norm(
        jnp.asarray(norm), clip_norm, slack, rescale
    )
    expected = (
        jnp.asarray(expected_fill) * output_bound / jnp.sqrt(float(slack))
    )

    chex.assert_trees_all_close(slack_vector, expected, atol=1e-6)

  @parameterized.parameters(jnp.inf, jnp.nan, -1.0)
  def test_invalid_clip_norm_returns_zero_slack(self, clip_norm):
    slack_vector = gradient_clipping._slack_from_norm(
        jnp.asarray(0.0), clip_norm, slack=4, rescale_to_unit_norm=False
    )

    chex.assert_trees_all_equal(slack_vector, jnp.zeros(4))

  @parameterized.product(
      norm_ratio=(0.0, 0.25, 1.0, 2.0),
      slack=(1, 10),
      rescale=(False, True),
  )
  def test_joint_output_respects_clip_bound(self, norm_ratio, slack, rescale):
    clip_norm = 2.0
    kwargs = dict(
        l2_clip_norm=clip_norm,
        keep_batch_dim=False,
        rescale_to_unit_norm=rescale,
    )
    baseline = gradient_clipping.clipped_grad(
        self._pytree_linear_loss, **kwargs
    )
    with_slack = gradient_clipping.clipped_grad(
        self._pytree_linear_loss, slack=slack, **kwargs
    )
    params = {'a': jnp.zeros(1), 'b': jnp.zeros(1)}
    leaf_value = norm_ratio * clip_norm / jnp.sqrt(2.0)
    data = {key: jnp.array([[leaf_value]]) for key in params}

    output = with_slack(params, data)
    bound = 1.0 if rescale else clip_norm
    tolerance = 4 * jnp.finfo(jnp.float32).eps * bound

    chex.assert_trees_all_equal(output.gradient, baseline(params, data))
    self.assertLessEqual(float(optax.tree.norm(output)), bound + tolerance)
    self.assertEqual(with_slack.sensitivity(), baseline.sensitivity())
    self.assertEqual(with_slack.l2_norm_bound, bound)

  def test_slack_uses_post_transform_norm_but_aux_keeps_raw_norm(self):
    output, aux = gradient_clipping.clipped_grad(
        self._linear_loss,
        l2_clip_norm=1.0,
        keep_batch_dim=False,
        pre_clipping_transform=lambda grad: 2.0 * grad,
        return_grad_norms=True,
        slack=4,
    )(jnp.zeros(1), jnp.array([[0.25]]))

    chex.assert_trees_all_close(output.gradient, jnp.array([0.5]))
    chex.assert_trees_all_close(output.slack, jnp.array([0.5, 0.5, 0.0, 0.0]))
    chex.assert_trees_all_close(aux.grad_norms, jnp.array([0.25]))

  def test_microbatching_and_normalization(self):
    query = gradient_clipping.clipped_grad(
        self._linear_loss,
        l2_clip_norm=1.0,
        keep_batch_dim=False,
        normalize_by=4.0,
        microbatch_size=2,
        slack=4,
    )
    output = query(
        jnp.zeros(1),
        jnp.array([[0.0], [0.5], [2.0], [2.0]]),
    )

    chex.assert_trees_all_close(output.gradient, jnp.array([0.625]))
    chex.assert_trees_all_close(
        output.slack, jnp.array([0.25, 0.25, 0.125, 0.125])
    )
    self.assertEqual(query.sensitivity(), 0.25)

  @parameterized.parameters(jnp.nan, jnp.inf)
  def test_padding_and_nonfinite_zero_full_output(self, nonfinite):
    query = jax.jit(
        gradient_clipping.clipped_grad(
            self._linear_loss,
            l2_clip_norm=1.0,
            keep_batch_dim=False,
            microbatch_size=2,
            slack=4,
        )
    )
    output = query(
        jnp.zeros(1),
        jnp.array([[0.0], [0.5], [nonfinite], [0.0]]),
        is_padding_example=jnp.array([False, False, False, True]),
    )

    chex.assert_trees_all_close(output.gradient, jnp.array([0.5]))
    chex.assert_trees_all_close(output.slack, jnp.array([1.0, 1.0, 0.5, 0.5]))

  def test_none_preserves_legacy_output(self):
    kwargs = dict(l2_clip_norm=0.5, keep_batch_dim=False)
    legacy = gradient_clipping.clipped_grad(self._linear_loss, **kwargs)
    explicit_none = gradient_clipping.clipped_grad(
        self._linear_loss, slack=None, **kwargs
    )
    params = jnp.zeros(2)
    data = jnp.array([[0.3, 0.4], [3.0, 4.0]])

    chex.assert_trees_all_equal(
        jax.jit(legacy)(params, data), jax.jit(explicit_none)(params, data)
    )
    self.assertEqual(legacy.sensitivity(), explicit_none.sensitivity())

  @parameterized.parameters(
      (0, ValueError),
      (-1, ValueError),
      (True, TypeError),
      (1.5, TypeError),
  )
  def test_rejects_invalid_slack(self, slack, error_type):
    with self.assertRaises(error_type):
      gradient_clipping.clipped_grad(
          self._linear_loss,
          l2_clip_norm=1.0,
          slack=slack,
      )

  def test_rejects_unsupported_clipping_configurations(self):
    with self.assertRaisesRegex(ValueError, 'scalar global clipping'):
      gradient_clipping.clipped_grad(
          self._linear_loss,
          l2_clip_norm={'layer': 1.0},
          slack=4,
      )
    with self.assertRaisesRegex(ValueError, 'discrete grid clipping'):
      gradient_clipping.clipped_grad(
          self._linear_loss,
          l2_clip_norm=1.0,
          grid_scale=10,
          slack=4,
      )

  @parameterized.parameters(False, True)
  def test_dynamic_clip_norm_is_jittable(self, rescale):
    params = jnp.zeros(1)
    data = jnp.array([[0.75]])

    def run(clip_norm):
      query = gradient_clipping.clipped_grad(
          self._linear_loss,
          l2_clip_norm=clip_norm,
          keep_batch_dim=False,
          rescale_to_unit_norm=rescale,
          slack=4,
      )
      return query(params, data), query.l2_norm_bound

    compiled = jax.jit(run).lower(jnp.asarray(1.0)).compile()
    output_1, bound_1 = compiled(jnp.asarray(1.0))
    output_2, bound_2 = compiled(jnp.asarray(2.0))

    expected_gradient_2 = 0.375 if rescale else 0.75
    expected_slack_2 = (
        [0.5, 0.5, 0.25, 0.0] if rescale else [1.0, 1.0, 0.5, 0.0]
    )
    expected_bound_2 = 1.0 if rescale else 2.0
    chex.assert_trees_all_close(output_1.gradient, jnp.array([0.75]))
    chex.assert_trees_all_close(output_1.slack, jnp.array([0.5, 0.0, 0.0, 0.0]))
    chex.assert_trees_all_close(
        output_2.gradient, jnp.array([expected_gradient_2])
    )
    chex.assert_trees_all_close(output_2.slack, jnp.array(expected_slack_2))
    chex.assert_trees_all_close(bound_1, jnp.array(1.0))
    chex.assert_trees_all_close(bound_2, jnp.array(expected_bound_2))

  @parameterized.parameters(jnp.float16, jnp.bfloat16)
  def test_slack_is_float32_for_low_precision_gradient(self, dtype):
    output = gradient_clipping.clipped_grad(
        self._linear_loss,
        l2_clip_norm=1.0,
        keep_batch_dim=False,
        slack=1,
    )(
        jnp.zeros(1, dtype=dtype),
        jnp.array([[0.5]], dtype=dtype),
    )

    self.assertEqual(output.gradient.dtype, dtype)
    self.assertEqual(output.slack.dtype, jnp.float32)


if __name__ == '__main__':
  absltest.main()

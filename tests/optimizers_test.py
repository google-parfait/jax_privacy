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
from jax_privacy import clipping
from jax_privacy import optimizers
import optax


def _simple_loss(params, x):
  """Per-example quadratic loss: 0.5 * mean((params - x)^2)."""
  return 0.5 * jnp.mean((params - x) ** 2)


def _init_params():
  """Simple 1D params for tests."""
  return jnp.array([1.0, 2.0, 3.0])


class FindAdaptiveStateTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(optimizer=optax.adam(1.0)),
      dict(optimizer=optax.adamw(1.0)),
      dict(optimizer=optax.rmsprop(1.0)),
      dict(optimizer=optax.adagrad(1.0)),
      dict(optimizer=optax.amsgrad(1.0)),
  )
  def test_finds_adaptive_state(self, optimizer):
    """Should extract nu from a ScaleByRmsState."""
    params = {'w': jnp.ones(3)}
    state = optimizer.init(params)
    result = optimizers._find_adaptive_state(state)
    chex.assert_trees_all_equal_shapes_and_dtypes(result, params)

  def test_raises_on_sgd_state(self):
    """Should raise ValueError for non-adaptive optimizer (SGD)."""
    opt = optax.sgd(1e-3)
    params = _init_params()
    state = opt.init(params)
    with self.assertRaisesRegex(ValueError, 'adaptive optimizer'):
      optimizers._find_adaptive_state(state)

  def test_raises_on_empty_tuple(self):
    """Should raise ValueError for an empty tuple."""
    with self.assertRaisesRegex(ValueError, 'adaptive optimizer'):
      optimizers._find_adaptive_state(())


class ScaleThenPrivatizeTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(opt_fn=lambda: optax.adam(1e-3)),
      dict(opt_fn=lambda: optax.adamw(1e-3)),
  )
  def test_init_matches_base_optimizer(self, opt_fn):
    """Init should produce the same state as the base optimizer."""
    base_opt = opt_fn()
    augmented = optimizers.scale_then_privatize(base_opt)
    params = _init_params()
    base_state = base_opt.init(params)
    aug_state = augmented.init(params)
    chex.assert_trees_all_close(aug_state, base_state)

  def test_scaling_is_non_isotropic(self):
    """The scaling should be different across coordinates with different nu."""
    # Use a custom preconditioner to directly control nu and verify
    # that scaling is non-isotropic (different per coordinate).
    base_opt = optax.adam(1e-3)
    custom_nu = jnp.array([1.0, 10000.0])
    augmented = optimizers.scale_then_privatize(
        base_opt, extract_preconditioner_from_state_fn=lambda state: custom_nu
    )
    params = jnp.array([0.0, 0.0])
    state = augmented.init(params)

    pct = augmented.pre_clipping_transform(state)
    g = jnp.array([1.0, 1.0])
    scaled = pct(g)

    # s = 1/(sqrt(nu) + eps), so coordinate with larger nu gets scaled more.
    # scaled[0] = 1/(sqrt(1)+eps) ≈ 0.5, scaled[1] = 1/(sqrt(10000)+eps) ≈ 0.01
    self.assertGreater(float(scaled[0]), float(scaled[1]))

  def test_update_applies_inverse(self):
    """update should apply inverse transform before base optimizer update."""
    base_opt = optax.adam(1e-3)
    augmented = optimizers.scale_then_privatize(base_opt)
    params = _init_params()
    state = augmented.init(params)

    # Run a step with the base optimizer to populate nu.
    grad = jnp.array([0.1, 0.2, 0.3])
    _, state = base_opt.update(grad, state, params)

    # Now, if we forward-transform a gradient and pass it to augmented.update,
    # the update should produce the same result as calling base_opt.update
    # with the original gradient (since update applies inverse internally).
    pct = augmented.pre_clipping_transform(state)
    scaled_grad = pct(grad)
    aug_updates, _ = augmented.update(scaled_grad, state, params)
    base_updates, _ = base_opt.update(grad, state, params)
    chex.assert_trees_all_close(aug_updates, base_updates, atol=1e-6)

  def test_custom_preconditioner_fn(self):
    """Should use a custom preconditioner function when provided."""
    base_opt = optax.sgd(1e-3)  # SGD has no adaptive state.
    params = _init_params()

    # Custom extractor that returns a constant nu.
    custom_nu = jnp.array([4.0, 9.0, 16.0])
    custom_fn = lambda state: custom_nu  # pylint: disable=unnecessary-lambda

    augmented = optimizers.scale_then_privatize(
        base_opt, extract_preconditioner_from_state_fn=custom_fn
    )
    state = augmented.init(params)

    pct = augmented.pre_clipping_transform(state)
    g = jnp.array([1.0, 1.0, 1.0])
    scaled = pct(g)

    # s = 1 / (sqrt(nu) + eps) ≈ [1/2, 1/3, 1/4] for eps ≈ 0.
    expected = g / (jnp.sqrt(custom_nu) + 1e-8)
    chex.assert_trees_all_close(scaled, expected, atol=1e-6)

  def test_raises_on_sgd_without_custom_fn(self):
    """Should raise ValueError when using SGD without custom extractor."""
    base_opt = optax.sgd(1e-3)
    augmented = optimizers.scale_then_privatize(base_opt)
    params = _init_params()
    state = augmented.init(params)

    with self.assertRaisesRegex(ValueError, 'adaptive optimizer'):
      augmented.pre_clipping_transform(state)

  def test_eps_root_affects_scaling(self):
    """eps_root should affect the scaling computation."""
    base_opt = optax.adam(1e-3, b1=0.0, b2=1.0)
    params = jnp.array([0.0])
    state = base_opt.init(params)
    grad = jnp.array([1.0])
    _, state = base_opt.update(grad, state, params)

    aug_no_eps_root = optimizers.scale_then_privatize(base_opt, eps_root=0.0)
    aug_with_eps_root = optimizers.scale_then_privatize(base_opt, eps_root=1.0)

    pct_no = aug_no_eps_root.pre_clipping_transform(state)
    pct_with = aug_with_eps_root.pre_clipping_transform(state)

    g = jnp.array([1.0])
    scaled_no = pct_no(g)
    scaled_with = pct_with(g)

    # With eps_root > 0, the scaling should be smaller (more regularized).
    self.assertGreater(float(scaled_no[0]), float(scaled_with[0]))

  @parameterized.named_parameters(
      ('float32', jnp.float32),
      ('float16', jnp.float16),
  )
  def test_zero_second_moment_keeps_zero_grad_finite(self, dtype):
    """v=0 must not map a zero gradient to NaN via 0 * inf."""
    base_opt = optax.adam(1e-3)
    zero_nu = jnp.zeros(2, dtype=dtype)
    augmented = optimizers.scale_then_privatize(
        base_opt,
        extract_preconditioner_from_state_fn=lambda state: zero_nu,
    )
    params = jnp.zeros(2, dtype=dtype)
    state = augmented.init(params)
    pct = augmented.pre_clipping_transform(state)
    scaled = pct(jnp.zeros(2, dtype=dtype))
    self.assertTrue(jnp.all(jnp.isfinite(scaled)))
    chex.assert_trees_all_close(scaled, jnp.zeros(2, dtype=dtype))

  def test_adam_init_zero_grad_is_finite(self):
    """Adam's initial nu is 0; a zero gradient must stay finite in float16."""
    base_opt = optax.adam(1e-3)
    augmented = optimizers.scale_then_privatize(base_opt)
    params = jnp.array([1.0, 2.0], dtype=jnp.float16)
    state = augmented.init(params)
    pct = augmented.pre_clipping_transform(state)
    scaled = pct(jnp.zeros_like(params))
    self.assertTrue(jnp.all(jnp.isfinite(scaled)))
    chex.assert_trees_all_close(scaled, jnp.zeros_like(params))

  def test_zero_second_moment_default_eps_is_1_over_eps(self):
    """With v=0, s_t should be 1/eps rather than inf."""
    base_opt = optax.adam(1e-3)
    zero_nu = jnp.zeros(1)
    augmented = optimizers.scale_then_privatize(
        base_opt,
        eps=1e-8,
        extract_preconditioner_from_state_fn=lambda s: zero_nu,
    )
    params = jnp.zeros(1)
    state = augmented.init(params)
    pct = augmented.pre_clipping_transform(state)
    scaled = pct(jnp.ones(1))
    chex.assert_trees_all_close(scaled, jnp.array([1e8]), atol=1.0)

  def test_zero_second_moment_eps_zero_is_finite(self):
    """v=0 and eps=0 must still produce a finite scale, not inf/NaN."""
    base_opt = optax.adam(1e-3)
    zero_nu = jnp.zeros(1)
    augmented = optimizers.scale_then_privatize(
        base_opt,
        eps=0.0,
        extract_preconditioner_from_state_fn=lambda s: zero_nu,
    )
    params = jnp.zeros(1)
    state = augmented.init(params)
    pct = augmented.pre_clipping_transform(state)
    scaled = pct(jnp.ones(1))
    self.assertTrue(jnp.isfinite(scaled[0]))
    self.assertGreater(float(scaled[0]), 0.0)

  def test_inverse_recovers_gradient_when_nu_is_zero(self):
    """Forward then inverse scaling should be the identity at v=0."""
    base_opt = optax.adam(1e-3)
    zero_nu = jnp.zeros(3)
    augmented = optimizers.scale_then_privatize(
        base_opt,
        extract_preconditioner_from_state_fn=lambda s: zero_nu,
    )
    params = jnp.zeros(3)
    state = augmented.init(params)
    grad = jnp.array([0.1, -0.2, 0.3])
    scaled = augmented.pre_clipping_transform(state)(grad)
    recovered = augmented.pre_clipping_transform(state, inverse=True)(scaled)
    chex.assert_trees_all_close(recovered, grad, atol=1e-5)

  def test_positive_nu_matches_closed_form(self):
    """Well-conditioned v should still match 1 / (sqrt(v) + eps)."""
    base_opt = optax.adam(1e-3)
    nu = jnp.array([4.0, 9.0, 16.0])
    eps = 1e-8
    augmented = optimizers.scale_then_privatize(
        base_opt,
        eps=eps,
        extract_preconditioner_from_state_fn=lambda s: nu,
    )
    params = jnp.zeros(3)
    state = augmented.init(params)
    g = jnp.ones(3)
    scaled = augmented.pre_clipping_transform(state)(g)
    expected = g / (jnp.sqrt(nu) + eps)
    chex.assert_trees_all_close(scaled, expected, atol=1e-6)

  def test_jit_compatible(self):
    """The full pipeline should work under jax.jit."""
    base_opt = optax.adamw(1e-3)
    augmented = optimizers.scale_then_privatize(base_opt)
    params = _init_params()
    state = augmented.init(params)

    @jax.jit
    def step(params, state, grad):
      pct = augmented.pre_clipping_transform(state)
      scaled = pct(grad)
      updates, new_state = augmented.update(scaled, state, params)
      new_params = optax.apply_updates(params, updates)
      return new_params, new_state

    grad = jnp.array([0.1, 0.2, 0.3])
    new_params, _ = step(params, state, grad)
    # Check it produced valid output.
    chex.assert_shape(new_params, params.shape)
    self.assertFalse(jnp.any(jnp.isnan(new_params)))


class ScaleThenPrivatizeE2ETest(parameterized.TestCase):
  """End-to-end test integrating scale_then_privatize with clipped_grad."""

  @parameterized.parameters(dict(use_jit=False), dict(use_jit=True))
  def test_augmented_matches_base_with_no_clipping(self, use_jit):
    """With l2_clip_norm=inf, augmented training should match base training."""
    params = jnp.array([3.0, 1.0, -2.0])
    data = jnp.array([[1.0, 0.0, 0.5], [4.0, -1.0, 2.0], [7.0, 3.0, -1.0]])
    num_steps = 5

    base_optimizer = optax.adamw(1e-2)
    scale_first_optimizer = optimizers.scale_then_privatize(base_optimizer)

    def base_train_step(params, opt_state, data):
      grad_fn = clipping.clipped_grad(
          _simple_loss,
          argnums=0,
          batch_argnums=1,
          l2_clip_norm=jnp.inf,
      )
      grads = grad_fn(params, data)
      updates, new_state = base_optimizer.update(grads, opt_state, params)
      new_params = optax.apply_updates(params, updates)
      return new_params, new_state

    def augmented_train_step(params, opt_state, data):
      transform_fn = scale_first_optimizer.pre_clipping_transform(opt_state)
      grad_fn = clipping.clipped_grad(
          _simple_loss,
          argnums=0,
          batch_argnums=1,
          l2_clip_norm=jnp.inf,
          pre_clipping_transform=transform_fn,
      )
      grad = grad_fn(params, data)
      updates, new_state = scale_first_optimizer.update(grad, opt_state, params)
      new_params = optax.apply_updates(params, updates)
      return new_params, new_state

    if use_jit:
      base_train_step = jax.jit(base_train_step)
      augmented_train_step = jax.jit(augmented_train_step)

    base_params = params
    base_state = base_optimizer.init(params)
    aug_params = params
    aug_state = scale_first_optimizer.init(params)

    for _ in range(num_steps):
      base_params, base_state = base_train_step(base_params, base_state, data)
      aug_params, aug_state = augmented_train_step(aug_params, aug_state, data)

    chex.assert_trees_all_close(aug_params, base_params, atol=1e-5)


class MiscellaneousTest(parameterized.TestCase):

  def test_as_augmented_optimizer(self):
    """Should convert a GradientTransformation to an AugmentedOptimizer."""
    optimizer = optax.adam(1e-3)
    augmented = optimizers.as_augmented_optimizer(optimizer)
    self.assertIsInstance(augmented, optimizers.AugmentedGradientTransformation)
    augmented_v2 = optimizers.as_augmented_optimizer(augmented)
    self.assertIs(augmented, augmented_v2)


if __name__ == '__main__':
  absltest.main()

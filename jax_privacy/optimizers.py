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

"""Augmented gradient transformations for differentially private training.

Gradient-based DP training algorithms may need to specify pre-processing of the
per-example gradients before clipping and noising happens. Because this is
tightly linked to the optimizer, we provide an
:class:`~jax_privacy.optimizers.AugmentedGradientTransformation` that provides a
pre-processing function (the ``pre_clipping_transform`` that can be passed into
:func:`~jax_privacy.clipping.clipped_grad`) bound together with the core
optimizer update.

The primary use case is the "scale-then-privatize" technique from:
  Ganesh, McMahan, Thakurta. "On Design Principles for Private Adaptive
  Optimizers." arXiv:2507.01129.

Example Usage (clipping but no noise):
  >>> from jax_privacy import clipped_grad, noise_addition
  >>> import optax
  >>> import jax.numpy as jnp
  >>> loss_fn = lambda params, batch: 0.5 * jnp.mean((params - batch) ** 2)
  >>> optimizer = scale_then_privatize(optax.adamw(1e-3))
  >>> params = jnp.ones(3)
  >>> data = jnp.ones((10, 3))
  >>> state = optimizer.init(params)
  >>> noise_multiplier = 0.0
  >>> noise_state = noise_addition.gaussian_privatizer(stddev=0.0).init(params)
  >>> for _ in range(5):
  ...   grad_fn = clipped_grad(
  ...     loss_fn,
  ...     l2_clip_norm=1,
  ...     pre_clipping_transform=optimizer.pre_clipping_transform(state)
  ...   )
  ...   stddev = grad_fn.sensitivity() * noise_multiplier
  ...   noise_fn = noise_addition.gaussian_privatizer(stddev=stddev)
  ...   clipped_grads = grad_fn(params, data)
  ...   noisy_grads, noise_state = noise_fn.update(clipped_grads, noise_state)
  ...   updates, state = optimizer.update(noisy_grads, state, params)
  ...   params = optax.apply_updates(params, updates)
"""

from typing import Callable, NamedTuple, Protocol

import jax
from jax import numpy as jnp
import optax


class PreClippingTransform(Protocol):
  """A function that applies a transformation to a pytree of updates."""

  def __call__(
      self, updates: optax.Updates, inverse: bool = False
  ) -> optax.Updates:
    ...


class AugmentedGradientTransformation(NamedTuple):
  """A gradient transformation augmented with a pre-clipping transform.

  This extends the standard optax.GradientTransformation interface with a
  ``pre_clipping_transform`` field that, given the current optimizer state,
  returns a ``pre_clipping_transform`` specifying how to transform per-example
  gradients before and after the clipping/noising step.

  The ``update`` function expects to receive gradients that have already been
  transformed by ``pre_clipping_transform(...)``, clipped, summed,
  and noised. It will internally apply the inverse transform before delegating
  to the base optimizer's update. See the module docstring for an example usage.

  Attributes:
    init: Initializes the optimizer state given initial parameters. Matches the
      optax.GradientTransformation.init API: init(params) -> state
    update: Computes parameter updates from noisy gradients. The noisy gradients
      should be in the *scaled* space (transform -> clip -> aggregate -> noise).
      This function applies the inverse transform internally before calling the
      base optimizer's update: update(updates, state, params=None) -> (updates,
      new_state)
    pre_clipping_transform: Given the current optimizer state, returns a
      ``pre_clipping_transform`` function intended to be used with
      :func:`~jax_privacy.clipping.clipped_grad`. It consumes a pytree with
      structure matching the parameters and returns a transformed pytree. The
      transformed pytree may or may not have the same structure. The ``update``
      function is responsible for mapping the input back to the original
      structure.
  """

  init: Callable[[optax.Params], optax.OptState]
  update: Callable[..., tuple[optax.Updates, optax.OptState]]
  pre_clipping_transform: Callable[[optax.OptState], PreClippingTransform]


def _find_adaptive_state(state: optax.OptState) -> optax.Updates:
  """Recursively searches for a recognized adaptive optimizer state."""
  # These are all adaptive optimizers where the square root of the second
  # moments of gradients is the appropriate scaling.
  if isinstance(state, optax.ScaleByAdamState):
    return state.nu
  elif isinstance(state, optax.ScaleByAmsgradState):
    return state.nu_max
  elif isinstance(state, optax.ScaleByRmsState):
    return state.nu
  elif isinstance(state, optax.ScaleByRssState):
    return state.sum_of_squares

  # If the state is a tuple/list (e.g., from optax.chain), search recursively.
  if isinstance(state, (tuple, list)):
    for sub_state in state:
      result = _find_adaptive_state(sub_state)
      if result is not None:
        return result

  raise ValueError(
      f'Could not find an adaptive optimizer state in {type(state)}.'
      ' scale_then_privatize requires an adaptive optimizer (e.g.,'
      ' optax.adam, optax.adamw, optax.rmsprop, optax.adagrad). If you are'
      ' using a custom adaptive optimizer, pass a custom'
      ' ``extract_preconditioner_from_state_fn`` function to'
      ' scale_then_privatize.'
  )


def _preconditioner_scale(
    second_moment: jax.Array,
    eps: float,
    eps_root: float,
) -> jax.Array:
  """Adam-style coordinate scale ``1 / (sqrt(v + eps_root) + eps)``."""
  # Compute in at least float32 so the default eps=1e-8 does not underflow in
  # float16. A zero denominator (v = 0 and a vanishing eps) is floored so the
  # scale is large and finite instead of inf, which would turn a zero gradient
  # into NaN (0 * inf).
  compute_dtype = jnp.promote_types(second_moment.dtype, jnp.float32)
  v = jnp.astype(second_moment, compute_dtype)
  denom = jnp.sqrt(v + eps_root) + eps
  denom = jnp.maximum(denom, jnp.finfo(compute_dtype).tiny)
  return jnp.reciprocal(denom)


def scale_then_privatize(
    base_optimizer: optax.GradientTransformation,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    extract_preconditioner_from_state_fn: (
        Callable[[optax.OptState], optax.Updates] | None
    ) = None,
) -> AugmentedGradientTransformation:
  r"""Constructs an AugmentedGradientTransformation for scale-then-privatize.

  This implements Algorithm 8 from Ganesh, McMahan, Thakurta (2507.01129).
  The key idea is to use the optimizer's second-moment estimate :math:`v_{t-1}`
  from the previous step to define a non-isotropic geometry for clipping and
  noising per-example gradients. Specifically:

  .. math::

    s_t = 1 / (\sqrt{v_{t-1} + \text{eps\_root}} + \text{eps})

  Before clipping, each per-example gradient :math:`g` is transformed to
  :math:`s_t \odot g`. After clipping + aggregation + noise addition, the
  ``update`` function applies the inverse (divides by :math:`s_t`) before
  passing to the base optimizer's update.

  A large ``eps`` or ``eps_root`` passed here (but not in the adaptive
  optimizer's scaling) will cause all coordinates to be scaled
  nearly-identically, effectively retrieving no pre-clipping transform.
  ``eps`` or ``eps_root`` matching the adaptive optimizer may add large noise
  in coordinates where the gradient is small. Ideally, this parameter should
  be tuned to tradeoff between these two regimes.

  Args:
    base_optimizer: A standard :class:`optax.GradientTransformation`, typically
      an adaptive optimizer like ``optax.adamw(...)``, ``optax.adam(...)``, or
      any chained transformation containing a ``scale_by_adam`` (or similar)
      component.
    eps: A small constant added to the denominator outside the square root when
      computing the scaling vector :math:`s_t`. Analogous to the ``eps``
      parameter in Adam. This also acts as a stability constant to prevent
      excessively large scaling in coordinates where :math:`v` is very small.
      Corresponds to :math:`\varepsilon_{s_1}` in Algorithm 8 of the paper. See
      the note above on tuning this parameter. The scale is computed in at
      least float32 so this default remains representable in float16.
    eps_root: A small constant added to :math:`v` inside the square root,
      analogous to ``eps_root`` in :func:`optax.scale_by_adam`. See the note
      above on tuning this parameter.
    extract_preconditioner_from_state_fn: A function that takes the optimizer
      state and returns the second-moment estimate (:math:`v`) pytree. If
      ``None``, uses a default implementation that handles common optax adaptive
      optimizers (Adam, AMSGrad, RMSProp, AdaGrad).

  Returns:
    An :class:`~jax_privacy.optimizers.AugmentedGradientTransformation` for the
    scale-then-privatize technique.
  """
  if extract_preconditioner_from_state_fn is None:
    extract_preconditioner_from_state_fn = _find_adaptive_state

  def pre_clipping_transform(state, inverse=False):
    """Extracts ν̂ from the optimizer state and builds the scaling transform."""
    # Compute the scaling vector: s = 1 / (sqrt(ν + eps_root) + eps).
    # It is the same formula used in Adam for per-coordinate learning rates.
    nu = extract_preconditioner_from_state_fn(state)
    scaling = jax.tree.map(
        lambda v: _preconditioner_scale(v, eps, eps_root), nu
    )
    scale_fn = jnp.divide if inverse else jnp.multiply
    # Multiply/divide in the scale dtype (at least float32) so float16
    # gradients do not overflow before the result is cast back.
    return lambda updates: jax.tree.map(
        lambda u, s: jnp.astype(scale_fn(jnp.astype(u, s.dtype), s), u.dtype),
        updates,
        scaling,
    )

  def update(updates, state, params, **extra_args):
    """Applies the inverse scaling transform, then the base optimizer update."""
    unscaled_updates = pre_clipping_transform(state, inverse=True)(updates)
    return base_optimizer.update(unscaled_updates, state, params, **extra_args)

  return AugmentedGradientTransformation(
      init=base_optimizer.init,
      update=update,
      pre_clipping_transform=pre_clipping_transform,
  )


def as_augmented_optimizer(
    optimizer: AugmentedGradientTransformation | optax.GradientTransformation,
) -> AugmentedGradientTransformation:
  """Wraps a plain Optax optimizer with an identity pre-clipping transform."""
  if isinstance(optimizer, AugmentedGradientTransformation):
    return optimizer
  return AugmentedGradientTransformation(
      init=optimizer.init,
      update=optimizer.update,
      pre_clipping_transform=lambda opt_state: lambda x: x,
  )

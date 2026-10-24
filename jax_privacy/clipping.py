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

"""Utilities for clipping function outputs and aggregating across a batch."""

import collections
from collections.abc import Sequence
import dataclasses
import functools
import math
import numbers
from typing import Any, Callable, NamedTuple, TypeAlias

import dp_accounting
import jax
import jax.numpy as jnp
from jax_privacy import _validate
import optax
from optax import microbatching

PyTree: TypeAlias = optax.ArrayTree
AuxiliaryOutput = collections.namedtuple('Aux', ['values', 'grad_norms', 'aux'])
ClippedTreeAndNorm = collections.namedtuple(
    '_ClippedTreeAndNorm', ['clipped', 'norm']
)
_REPLACE_SPECIAL = dp_accounting.NeighboringRelation.REPLACE_SPECIAL


class ClippedGradOutput(NamedTuple):
  """Gradient and slack fields forming a single primary query.

  Both fields must be privatized together before ``slack`` is released.
  """

  gradient: PyTree
  slack: jax.Array


def _validate_slack_config(
    slack: int | None,
    l2_clip_norm: float | PyTree,
    grid_scale: int | None,
) -> None:
  """Validate the clipping configurations supported by slack outputs."""
  if slack is None:
    return
  if isinstance(slack, bool) or not isinstance(slack, int):
    raise TypeError('slack must be a Python integer or None.')
  if slack <= 0:
    raise ValueError('slack must be positive.')
  if grid_scale is not None:
    raise ValueError('slack is not supported with discrete grid clipping.')
  # We may want to support one slack vector per leaf in the future.
  if not jnp.isscalar(l2_clip_norm):
    raise ValueError('slack requires scalar global clipping.')


def _slack_from_norm(
    gradient_norm: jax.Array,
    clipping_bound: jax.typing.ArrayLike,
    slack: int,
    rescale_to_unit_norm: bool,
) -> jax.Array:
  """Construct prefix-filled slack from an already-computed norm."""
  gradient_norm = jnp.asarray(gradient_norm, dtype=jnp.float32)
  clip_norm = jnp.asarray(clipping_bound, dtype=jnp.float32)
  valid_clip_norm = jnp.isfinite(clip_norm) & (clip_norm > 0)
  safe_clip_norm = jnp.where(valid_clip_norm, clip_norm, 1.0)
  unused_fraction = jnp.where(
      valid_clip_norm,
      jnp.clip(1.0 - gradient_norm / safe_clip_norm, 0.0, 1.0),
      0.0,
  )
  unused_fraction = jnp.nan_to_num(unused_fraction, nan=0.0)
  k = jnp.asarray(slack, dtype=jnp.float32)
  fill = jnp.clip(
      k * unused_fraction - jnp.arange(slack, dtype=jnp.float32),
      0.0,
      1.0,
  )
  # sum(fill**2) <= sum(fill) = k * unused_fraction, which preserves
  # the joint gradient-and-slack L2 bound.
  output_bound = (
      1.0
      if rescale_to_unit_norm
      else jnp.where(valid_clip_norm, clip_norm, 0.0)
  )
  return fill * output_bound / jnp.sqrt(k)


@dataclasses.dataclass(frozen=True)
class BoundedSensitivityCallable:
  """Callable with a sensitivity property.

  If ``has_aux`` is ``False``, the sensitivity guarantee holds for the
  entire output which may be an arbitrary PyTree of JAX Arrays.  If
  ``has_aux`` is ``True``, the output of the function is a pair
  ``(value, aux)`` and the sensitivity guarantee only holds for the
  ``value`` PyTree. The ``aux`` PyTree is returned on a per-example
  basis (i.e., as a PyTree of arrays having a batch axis).  The caller
  should handle the ``aux`` output with care w.r.t. DP guarantees,
  should they be needed.
  """

  bounded_sensitivity_fun: Callable[..., Any]
  l2_norm_bound: float
  has_aux: bool

  def __call__(
      self,
      *args,
      is_padding_example: jax.Array | None = None,
      **kwargs,
  ):
    """Calls the underlying clipped function.

    Args:
      *args: The same positional arguments as the original ``fun`` passed to
        ``clipped_fun`` or ``clipped_grad``.
      is_padding_example: An optional 1-D boolean :class:`jax.Array` of shape
        ``(batch_size,)`` indicating which examples in the batch are synthetic
        padding (``True``) vs. real data (``False``).  Padding examples are
        zeroed out after clipping but before summation, so they contribute
        exactly zero to the aggregated output and do not affect the formal
        sensitivity guarantee.  Floating-point leaves of per-example auxiliary
        outputs and norms at padding positions are set to ``NaN``.  When
        omitted, all examples are treated as real. For a full guide, see `Using
        is_padding_example
        <https://jax-privacy.readthedocs.io/en/latest/sharp_edges_variable_batch_sizes.html#using-is-padding-example>`_.
      **kwargs: Forwards all keyword arguments to the underlying function.

    Returns:
      The clipped and aggregated output.  See the docstring of the
      factory function (``clipped_fun`` or ``clipped_grad``) that created
      this instance for the exact return signature.
    """
    return self.bounded_sensitivity_fun(
        *args, is_padding_example=is_padding_example, **kwargs
    )

  def sensitivity(
      self,
      neighboring_relation: dp_accounting.NeighboringRelation = _REPLACE_SPECIAL,  # pylint: disable=line-too-long
  ):
    """Returns the L2 sensitivity of the Callable.

    The L2 sensitivity is defined with respect to the given neighboring relation
    and the unit of privacy implied by the function that created this instance.

    Args:
      neighboring_relation: The neighboring relation to consider.

    Returns:
      The L2 sensitivity of the Callable.
    """
    match neighboring_relation:
      case dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE:
        return self.l2_norm_bound
      case dp_accounting.NeighboringRelation.REPLACE_ONE:
        return 2 * self.l2_norm_bound
      case dp_accounting.NeighboringRelation.REPLACE_SPECIAL:
        return self.l2_norm_bound
      case _:
        raise ValueError(f'Unsupported {neighboring_relation=}')


def _compute_pytree_l2_norm(
    pytree: PyTree,
    min_dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
  """Computes the L2 norm of a PyTree in at least `min_dtype` precision."""

  # Upcast per leaf to avoid float16 overflow/underflow without full-tree copy.
  def _leaf_squared_norm(x: jax.Array) -> jax.Array:
    compute_dtype = jnp.promote_types(x.dtype, min_dtype)
    x_promoted = x.astype(compute_dtype)
    return jnp.sum(x_promoted * x_promoted)

  squared_norms = [_leaf_squared_norm(x) for x in jax.tree.leaves(pytree)]
  return jnp.sqrt(sum(squared_norms))


def clip_pytree(
    pytree: PyTree,
    clip_norm: float | PyTree,
    rescale_to_unit_norm: bool = False,
):
  """Clips a PyTree of jax arrays.

    The clipping behavior is determined by the type of ``clip_norm``,
    like so:

    - If ``clip_norm`` is a Scalar (Global Clipping): Calculates the
      global L2 norm of the input PyTree; & if the norm exceeds
      ``clip_norm``, the entire PyTree is scaled down.
    - If ``clip_norm`` is a PyTree (Per-Layer Clipping): Calculates
      the L2 norm for each individual leaf; & if a leaf's norm exceeds
      its corresponding threshold in ``clip_norm``, that specific leaf
      is scaled down. The ``clip_norm`` PyTree structure must be a
      prefix of the input ``pytree``. Where ``clip_norm`` terminates
      early, its leaf values automatically broadcast down to match all
      corresponding sub-leaves of ``pytree``.

  If ``rescale_to_unit_norm`` is ``True``, the PyTree is additionally
  scaled by ``1.0 / clip_norm`` (resulting in a norm of at most 1.0
  no matter what ``clip_norm`` is). Handles cases where the original norm
  is zero, or the clip norm is 0 or infinity.

  Formal Guarantees:

  - When the input PyTree is all-finite, the output PyTree will have
    norm at most ``clip_norm`` if ``rescale_to_unit_norm`` is
    ``False``, and at most 1.0 if it is ``True``. For low-precision
    leaves (e.g., float16), the norm bound may be exceeded by up to
    the leaf dtype's machine epsilon due to rounding.
  - The output PyTree will have the same structure+dtypes as the input PyTree.
  - The output PyTree may contain NaN or inf values, but only when the
    returned L2 norm is non-finite (i.e., NaN or inf). If the returned L2
    norm is finite, the output is guaranteed to be all-finite.

  Edge Case Handling:

  ======================= ==================== =================================
  Case                    rescale_to_unit_norm Output
  ======================= ==================== =================================
  clip_norm = 0           False                Zero
  clip_norm = 0           True                 Input / norm, as clip_norm -> 0
  clip_norm = inf         False                Unchanged
  clip_norm = inf         True                 Zero
  clip_norm < 0 (static)  -                    Raises ValueError
  clip_norm < 0 (dynamic) -                    Zero
  pytree_norm = 0         -                    Unchanged
  ======================= ==================== =================================

  Args:
    pytree: The PyTree of arrays to clip.
    clip_norm: The maximum L2 norm allowed. Can be a single float or a PyTree;
      depending on whether Global / "Per-Layer" clipping is to be done (see
      details above).
    rescale_to_unit_norm: If ``True``, the output PyTree's norm is rescaled by
      ``1.0 / clip_norm`` after potential clipping. If ``False``, the output
      PyTree has norm at most ``clip_norm``.

  Returns:
    A tuple ``(clipped_pytree, original_l2_norm)``, where
    ``clipped_pytree`` is the processed PyTree and
    ``original_l2_norm`` is the L2 norm of the input PyTree. In case
    of per-layer clipping, ``original_l2_norm`` is a PyTree containing
    L2 norms of each leaf.
  """
  # -- Per-layer Norm --
  if not jnp.isscalar(clip_norm):
    recursive_clip = functools.partial(
        clip_pytree,
        rescale_to_unit_norm=rescale_to_unit_norm,
    )
    results = jax.tree.map(
        lambda c_norm, p_tree: ClippedTreeAndNorm(
            *recursive_clip(p_tree, c_norm)
        ),
        clip_norm,
        pytree,
    )
    is_leaf = lambda x: isinstance(x, ClippedTreeAndNorm)
    clipped = jax.tree.map(lambda x: x.clipped, results, is_leaf=is_leaf)
    norms = jax.tree.map(lambda x: x.norm, results, is_leaf=is_leaf)
    return ClippedTreeAndNorm(clipped=clipped, norm=norms)

  # -- Global Norm --
  if isinstance(clip_norm, numbers.Real) and clip_norm < 0:
    raise ValueError(f'clip_norm must be non-negative, got {clip_norm=}.')

  clip_norm = jnp.maximum(clip_norm, 0.0)
  l2_norm = _compute_pytree_l2_norm(pytree)
  scale = jnp.minimum(1.0, clip_norm / l2_norm)
  if rescale_to_unit_norm:
    scale = jax.lax.select(clip_norm > 0, scale / clip_norm, 1 / l2_norm)
  # If l2_norm is 0 or nan, set scale to 0.0.
  scale = jnp.nan_to_num(scale, nan=0.0, posinf=0.0)
  # Multiply in float32 (scale's dtype) to avoid float16 overflow, then cast.
  clipped = jax.tree.map(
      lambda x: (scale * x.astype(scale.dtype)).astype(x.dtype), pytree
  )
  return clipped, l2_norm.astype(jnp.float32)


def _maybe_zero(clipped, l2_norm, is_padding, nan_safe=True):
  """Zeros out padding and (optionally) non-finite examples."""
  should_zero = is_padding
  if nan_safe:
    # l2_norm may be a scalar or a pytree (per-layer clipping).
    norm_leaves = jax.tree.leaves(l2_norm)
    all_finite = functools.reduce(
        lambda a, b: a & b,
        (jnp.isfinite(n) for n in norm_leaves),
    )
    should_zero = should_zero | ~all_finite
  return jax.tree.map(lambda x: jnp.where(should_zero, 0, x), clipped)


def _mask_padding(pytree: PyTree, is_padding_example: jax.Array) -> PyTree:
  """Sets floating-point leaves at padding positions to NaN."""

  def _mask_leaf(x: jax.Array) -> jax.Array:
    if not jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating):
      return x
    mask = jnp.expand_dims(is_padding_example, tuple(range(1, x.ndim)))
    return jnp.where(mask, jnp.nan, x)

  return jax.tree.map(_mask_leaf, pytree)


def _adjusted_clip_norm(
    l2_clip_norm: float,
    grid_scale: int,
    num_params: int,
) -> float:
  """Computes a tighter clip norm to account for norm increase from rounding.

  When a real-valued vector is scaled by ``grid_scale / l2_clip_norm`` and
  rounded to the nearest integer, each coordinate changes by at most 0.5.
  This can increase the L2 norm by at most ``sqrt(num_params) * 0.5`` (in
  integer units), or equivalently ``sqrt(num_params) * l2_clip_norm /
  grid_scale / 2`` in the original units. We return a tighter clip norm so
  that after rounding, the integer L2 norm is at most ``grid_scale``.

  Args:
    l2_clip_norm: The desired post-rounding L2 norm bound (in original units).
    grid_scale: Number of integer grid steps corresponding to l2_clip_norm.
    num_params: Total number of scalar parameters in the gradient.

  Returns:
    The adjusted (tighter) clip norm in original units.

  Raises:
    ValueError: If the rounding error exceeds the clip norm.
  """
  grid_step = l2_clip_norm / grid_scale
  max_rounding_error = num_params**0.5 * grid_step * 0.5
  adjusted = l2_clip_norm - max_rounding_error
  # Only check when all values are concrete Python numbers (not JAX tracers).
  if isinstance(adjusted, numbers.Real) and adjusted <= 0:
    raise ValueError(
        f'Grid scale {grid_scale} is too small for {num_params} parameters. '
        f'The rounding error ({max_rounding_error:.4f}) exceeds the clip norm '
        f'({l2_clip_norm:.4f}). Increase grid_scale.'
    )
  return adjusted


def clip_and_round_to_grid(
    gradient: PyTree,
    l2_clip_norm: float,
    grid_scale: int,
) -> tuple[PyTree, jax.Array]:
  """Clips a gradient and rounds it to an integer grid.

  This function performs four operations on a single (unbatched) gradient
  pytree:

    1. Computes a tighter (adjusted) clip norm to account for the L2 norm
       increase caused by rounding.
    2. Clips the gradient to have L2 norm at most the adjusted clip norm.
    3. Scales the gradient by ``grid_scale / l2_clip_norm``.
    4. Rounds each coordinate to the nearest integer.

  After rounding, the integer vector has L2 norm at most ``grid_scale``.

  This function is designed to be used with :func:`jax.vmap` to process a batch
  of per-example gradients.

  Args:
    gradient: A pytree of gradient arrays for a single example.
    l2_clip_norm: The desired L2 clip norm (in original units).
    grid_scale: Number of integer grid steps corresponding to l2_clip_norm.

  Returns:
    A tuple ``(rounded, l2_norm)`` where ``rounded`` is a pytree of int64
    arrays representing the rounded gradient on the integer grid, and
    ``l2_norm`` is the L2 norm of the input gradient before clipping.

  Raises:
    ValueError: If grid_scale is too small relative to the number of
      parameters (the rounding error would exceed the clip norm).
  """
  _validate.x64_enabled()
  num_params = sum(x.size for x in jax.tree.leaves(gradient))
  adj_clip_norm = _adjusted_clip_norm(l2_clip_norm, grid_scale, num_params)
  clipped, l2_norm = clip_pytree(gradient, adj_clip_norm)
  scale = grid_scale / l2_clip_norm
  rounded = jax.tree.map(
      lambda g: jnp.round(g * scale).astype(jnp.int64), clipped
  )
  return rounded, l2_norm


# pylint: disable=g-bare-generic
def _with_extra_batch_axis(
    fun: Callable, batch_argnums: int | Sequence[int]
) -> Callable:
  """Wraps a function to add an extra batch axis to the batch_argnums."""
  if isinstance(batch_argnums, int):
    batch_argnums = (batch_argnums,)

  def wrapped_fun(*args, **kwargs):
    args_with_group_axis = list(args)
    for i in batch_argnums:
      args_with_group_axis[i] = jax.tree.map(
          lambda x: jnp.expand_dims(x, axis=1), args[i]
      )
    return fun(*args_with_group_axis, **kwargs)

  return wrapped_fun


def _normalize_fun_to_return_aux(fun, has_aux):
  if has_aux:
    return fun
  else:
    return lambda *args, **kwargs: (fun(*args, **kwargs), ())


def _num_real_microbatches(
    is_padding_example: jax.Array,
    microbatch_size: int | None,
) -> int | jax.Array:
  """Calculates the number of non-padding microbatches.

  The returned result is 1 + the index of the last microbatch that contains at
  least one non-padding example.  This means that microbatches consisting of
  all-padding examples that do not appear at the end will be treated as a real
  microbatch.

  Args:
    is_padding_example: A 1D array of shape (num_examples,).
    microbatch_size: Argument passed to ``microbatch``.

  Returns:
    The true batch size.
  """
  if microbatch_size is None:
    return is_padding_example.shape[0]
  reshaped = microbatching.reshape_batch_axis(
      is_padding_example, microbatch_size
  )
  # Ensure there is at least one True in the array.
  is_real_batch = jnp.append(True, ~reshaped.all(axis=1))
  # We want the last real microbatch, argmax returns the first True value,
  # so we add increasing numbers from 0 to 1 to each index.
  return jnp.argmax(is_real_batch + jnp.linspace(0, 1, is_real_batch.size))


def _maybe_squeeze_axis_1(x: jax.Array) -> jax.Array:
  """Squeezes the second axis if it is of size 1.

  We need a guard check because some leaves in the pytree might be scalars or
  have only 1 dimension (e.g., if the model reduces over the batch dimension of
  size 1 to return a single loss scalar). In JAX, calling jnp.squeeze(axis=1) on
  a scalar or 1D array throws an out-of-bounds error. Additionally, jnp.squeeze
  errors out if the specified axis is not of size 1. This check ensures we only
  squeeze valid 2D+ arrays that actually have size 1 at index 1.

  Note that this will also squeeze axis 1 for auxiliary outputs that naturally
  have a shape like ``(Batch, 1, ...)``. Callers should be aware of this
  potential
  side effect.

  Args:
    x: The input array.

  Returns:
    The input array with the second axis squeezed if it is of size 1.
  """
  if hasattr(x, 'shape') and len(x.shape) >= 2 and x.shape[1] == 1:
    return jnp.squeeze(x, axis=1)
  return x


def _validate_is_padding_example(is_padding_example, batch_size: int):
  """Validates the is_padding_example array."""
  if is_padding_example.ndim != 1:
    raise ValueError(
        'is_padding_example must be a 1-D array, got'
        f' ndim={is_padding_example.ndim}.'
    )
  if is_padding_example.shape[0] != batch_size:
    raise ValueError(
        f'is_padding_example has shape {is_padding_example.shape} but'
        f' batch has size {batch_size}. The leading dimensions must'
        ' match.'
    )
  if is_padding_example.dtype != jnp.bool_:
    raise ValueError(
        'is_padding_example must have dtype bool, got'
        f' {is_padding_example.dtype}.'
    )


def clipped_fun(
    fun: Callable,
    has_aux: bool = False,
    *,
    batch_argnums: int | Sequence[int] = 0,
    keep_batch_dim: bool = True,
    l2_clip_norm: float | PyTree = 1.0,
    rescale_to_unit_norm: bool = False,
    normalize_by: float = 1.0,
    return_norms: bool = False,
    microbatch_size: int | None = None,
    nan_safe: bool = True,
    dtype: jax.typing.DTypeLike | None = None,
    prng_argnum: int | None = None,
    spmd_axis_name: str | None = None,
    grid_scale: int | None = None,
    slack: int | None = None,
) -> BoundedSensitivityCallable:
  """Transforms a function to clip its output and sum across a batch.

  Example Usage:
    >>> data = jnp.array([0, 1, 2, 3, 4, 5])
    >>> clipped_mean = clipped_fun(jnp.mean, l2_clip_norm=1.0)
    >>> clipped_mean(data)
    Array(5., dtype=float32)
    >>> is_padding_example = jnp.array([False, False, False, False, True, True])
    >>> clipped_mean(data, is_padding_example=is_padding_example)
    Array(3., dtype=float32)

  Formal Guarantees:
    For the first function output:
      The L2 sensitivity of the returned callable is a complex function of
      the input parameters (``l2_clip_norm``, ``rescale_to_unit_norm``,
      ``normalize_by``, ``grid_scale``, per-layer clipping settings, etc.)
      and may change as new features are added. Rather than reasoning about
      the sensitivity from the parameters, callers should query the returned
      callable's ``.sensitivity()`` method (or equivalently its
      ``.l2_norm_bound`` attribute for add-or-remove-one / zero-out DP)
      directly. Under replace-one DP, ``.sensitivity()`` doubles the
      add/remove bound.

    Extra auxiliary outputs (aux, norms) are per-example.
      This function guarantees that per-example outputs only depend on the data
      for the same example. This allows maximum flexibility for the caller
      to aggregate these as desired (possibly with a DP mean, median, quantile,
      or histogram mechanism).

  Args:
    fun: The function to be clipped.
    has_aux: If ``True``, ``fun`` is expected to return a tuple ``(value,
      aux)``. Only the value will be clipped + aggregated, ``aux`` will be
      returned on a per-example basis. Exercise caution when using this as the
      sensitivity guarantees of the returned Callable are only provided w.r.t.
      ``value``.
    batch_argnums: Specifies which argument(s) of ``fun`` contain the batch
      dimension. All arguments specified here must have the same size along the
      0th axis.
    keep_batch_dim: If ``True``, batch inputs will be passed to ``fun`` with a
      leading batch axis of size 1.  If ``False``, this size 1 axis will be
      dropped (reducing the rank of the batch args by 1 before passing to
      ``fun``).
    l2_clip_norm: The maximum L2 norm allowed. Can be a single float or a
      PyTree; depending on whether Global / "Per-Layer" clipping is to be done.
    rescale_to_unit_norm: If ``True``, the output PyTree's norm is rescaled by
      ``1.0 / clip_norm`` after potential clipping. If ``False``, the output
      PyTree has norm at most ``clip_norm``.
    normalize_by: Divide the clipped output by this value before returning.
    return_norms: If ``True``, the returned Callable will return the l2_norms of
      the per-example values before clipping. These values should be handled
      with care, see the formal guarantees above.
    microbatch_size: If set, the batch is split up into microbatches of this
      size. These microbatches are then processed sequentially, with operations
      on the groups within each microbatch being vectorized using
      :func:`jax.vmap`. This can be used to reduce peak memory usage at the cost
      of increased sequential computation.
    nan_safe: If ``True``, per-example outputs with non-finite L2 norms (NaN or
      inf) are zeroed out before aggregation, preserving the formal guarantees.
    dtype: Optional dtype for the clipped+aggregated PyTree. If ``None``, the
      dtype will be the same as the dtypes of the function output. Can be useful
      to avoid overflow issues when using low-precision dtypes as the
      transformed function computes a sum over a potentially large batch.
    prng_argnum: If set, specifies which argument of ``fun`` is a PRNG key. The
      PRNG will be split to have a batch dimension and vmapped over.
    spmd_axis_name: See :func:`jax.vmap`.
    grid_scale: If set, per-example outputs are additionally scaled and rounded
      to an integer grid after clipping.  Specifically, each clipped output is
      multiplied by ``grid_scale / l2_clip_norm``, rounded to the nearest
      integer, and cast to ``jnp.int64``.  The clipping norm is tightened
      automatically so that the integer L2 norm of each rounded output is at
      most ``grid_scale``.  This option is designed for use with the discrete
      Gaussian mechanism.  Incompatible with ``rescale_to_unit_norm=True`` and
      ``normalize_by != 1.0``.  When set, ``dtype`` is ignored (output is always
      ``jnp.int64``).
    slack: Optional number of SlaClip slack coordinates. Must be a positive
      Python integer. When set, the returned value is a
      :class:`ClippedGradOutput` whose fields must be privatized together. Slack
      coordinates have float32 dtype. Only scalar global clipping is currently
      supported.

  Returns:
    A :class:`~jax_privacy.clipping.BoundedSensitivityCallable`
    wrapping a new function ``clip_fn`` that clips the output of
    ``fun`` and sums across the batch. ``clip_fn`` takes the same
    positional arguments as ``fun``, and additionally accepts an
    optional keyword argument ``is_padding_example`` (see
    :meth:`~jax_privacy.clipping.BoundedSensitivityCallable.__call__`
    for details). The exact output signature depends on ``has_aux``
    and ``return_norms``:

    .. list-table::
       :header-rows: 1

       * - ``has_aux``
         - ``return_norms``
         - ``clipped_fn`` returns
       * - ``False``
         - ``False``
         - ``value``
       * - ``True``
         - ``False``
         - ``value, aux``
       * - ``False``
         - ``True``
         - ``value, norms``
       * - ``True``
         - ``True``
         - ``value, (aux, norms)``
  """
  if isinstance(batch_argnums, int):
    batch_argnums = (batch_argnums,)
  _validate_slack_config(slack, l2_clip_norm, grid_scale)
  if grid_scale is not None:
    _validate.discrete_clipping(
        grid_scale,
        rescale_to_unit_norm,
        normalize_by,
        l2_clip_norm,
    )

  fun = _normalize_fun_to_return_aux(fun, has_aux)

  def clipped_fn(*args, is_padding_example=None, **kwargs):
    batch_size = _validate.batch([args[i] for i in batch_argnums])
    if is_padding_example is None:
      is_padding_example = jnp.zeros(batch_size, dtype=jnp.bool_)
    else:
      _validate_is_padding_example(is_padding_example, batch_size)

    def clipped_fun_one_group(*args, is_padding_example, **kwargs):
      value, aux = fun(*args, **kwargs)
      if grid_scale is not None:
        # dtype is intentionally ignored here; the output is always int64.
        clipped, norm = clip_and_round_to_grid(value, l2_clip_norm, grid_scale)
      else:
        value = optax.tree.cast(value, dtype)
        clipped, norm = clip_pytree(value, l2_clip_norm, rescale_to_unit_norm)
      if slack is not None:
        slack_vector = _slack_from_norm(
            norm, l2_clip_norm, slack, rescale_to_unit_norm
        )
        clipped = ClippedGradOutput(clipped, slack_vector)
      clipped = _maybe_zero(clipped, norm, is_padding_example, nan_safe)
      return clipped, aux, norm

    num_real_mb = _num_real_microbatches(is_padding_example, microbatch_size)
    sum_ = microbatching.AccumulationType.SUM
    concat = microbatching.AccumulationType.CONCAT
    axes = [0 if i in batch_argnums else None for i in range(len(args))]
    if prng_argnum is not None:
      args = list(args)
      rngs = args[prng_argnum]
      split_rngs = jax.tree.map(lambda x: jax.random.split(x, batch_size), rngs)
      args[prng_argnum] = split_rngs
      axes[prng_argnum] = 0

    microbatched_vmap_fun = microbatching.micro_vmap(
        clipped_fun_one_group,
        in_axes=axes,
        microbatch_size=microbatch_size,
        accumulator=(sum_, concat, concat),
        num_real_microbatches=num_real_mb,
        vmap_fn=functools.partial(jax.vmap, spmd_axis_name=spmd_axis_name),
    )

    clipped_values, aux, norms = microbatched_vmap_fun(
        *args, is_padding_example=is_padding_example, **kwargs
    )
    if keep_batch_dim:
      # If keep_batch_dim is True, we artificially added a dimension of size 1
      # to the batch arguments before passing them to the vmap'ed function.
      # While vmap and micro_vmap take a single example's slice, it preserves
      # the output shapes of the inner function. If the inner function (e.g.
      # your model) returns something that still contains this size 1 batch
      # dimension, vmap will stack these outputs (not concatenate), resulting
      # in a shape like (B, 1, ...). If the caller expects standard shapes
      # (B, ...), this extra axis at index 1 can cause broadcasting issues
      # downstream (e.g. in Keras metrics). To fix this, we squeeze the axis 1
      # of aux outputs.
      aux = jax.tree.map(_maybe_squeeze_axis_1, aux)
    aux, norms = _mask_padding((aux, norms), is_padding_example)
    if normalize_by != 1.0:
      clipped_values = jax.tree.map(lambda x: x / normalize_by, clipped_values)

    match has_aux, return_norms:
      case False, False:
        return clipped_values
      case False, True:
        return clipped_values, norms
      case True, False:
        return clipped_values, aux
      case True, True:
        return clipped_values, (aux, norms)

  if grid_scale is not None:
    norm_bound = float(grid_scale)
  elif rescale_to_unit_norm:
    # If every leaf is rescaled to unit norm, the worst-case global norm is
    # the square root of the total number of leaves in the PyTree.
    num_leaves = len(jax.tree.leaves(l2_clip_norm))
    norm_bound = math.sqrt(num_leaves) / normalize_by
  else:
    norm_bound = (
        sum(leaf**2 for leaf in jax.tree.leaves(l2_clip_norm)) ** 0.5
    ) / normalize_by

  if keep_batch_dim:
    clipped_fn = _with_extra_batch_axis(clipped_fn, batch_argnums)
  callable_has_aux = has_aux or return_norms
  return BoundedSensitivityCallable(clipped_fn, norm_bound, callable_has_aux)


def _validate_static_args(argnums, batch_argnums, normalize_by):
  """Validates the argnums and batch_argnums inputs are compatible."""
  if normalize_by <= 0.0:
    raise ValueError(f'normalize_by must be > 0, got {normalize_by}.')
  if isinstance(argnums, int):
    argnums = (argnums,)
  if isinstance(batch_argnums, int):
    batch_argnums = (batch_argnums,)
  if not batch_argnums:
    raise ValueError('Batch Argnums must not be empty.')
  if min(argnums + batch_argnums) < 0:
    raise ValueError(
        f'argnums={argnums} and batch_argnums={batch_argnums} must be >= 0.'
    )
  shared_argnums = set(argnums) & set(batch_argnums)
  if shared_argnums:
    raise ValueError(
        'Cannot compute clipped gradients for argnums that have a batch axis. '
        f'{argnums=} and {batch_argnums=} with overlap {list(shared_argnums)}.'
    )


def clipped_grad(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    *,
    l2_clip_norm: float | PyTree,
    rescale_to_unit_norm: bool = False,
    normalize_by: float = 1.0,
    batch_argnums: int | Sequence[int] = 1,
    keep_batch_dim: bool = True,
    return_values: bool = False,
    return_grad_norms: bool = False,
    pre_clipping_transform: Callable[[PyTree], PyTree] = lambda x: x,
    microbatch_size: int | None = None,
    nan_safe: bool = True,
    dtype: jax.typing.DTypeLike | None = None,
    prng_argnum: int | None = None,
    spmd_axis_name: str | None = None,
    grid_scale: int | None = None,
    slack: int | None = None,
) -> BoundedSensitivityCallable:
  """Create a function to compute the sum of clipped gradients of fun.

  This function acts as a transformation similar to :func:`jax.grad`,
  but with added functionality for gradient clipping applied on a
  per-example (or per-group) basis before summation. It computes the
  gradient of ``fun`` with respect to ``argnums``, calculates the L2
  norm of the gradient for each example slice along the first axis of
  the ``batch_argnums`` args, clips each per-example gradient to have
  a norm of at most ``l2_clip_norm``, and finally sums these clipped
  gradients.

  Non-grad outputs of the returned function (aux, values, grad_norms)
  may optionally be returned by setting the arguments ``has_aux``,
  ``return_values``, and/or ``return_grad_norms`` to ``True``.  These
  outputs are per-example, and hence have a batch axis. It is up to
  the caller to handle these as necessary. See the ``DP Sensitivity
  Guarantee`` below for more details on this design choice.

  Example Usage:
    >>> import jax.numpy as jnp
    >>> f = lambda param, data: 0.5 * jnp.mean((data - param)**2)
    >>> g = clipped_grad(f, l2_clip_norm=jnp.inf)
    >>> g(3.0, jnp.array([0, 7, -2]))
    Array(4., dtype=float32)

  Example Usage (with Auxiliary Output):
    >>> g = clipped_grad(
    ...   f, l2_clip_norm=jnp.inf, return_values=True, return_grad_norms=True
    ... )
    >>> _, aux = g(3.0, jnp.array([0, 7, -2]))
    >>> aux.values
    Array([ 4.5,  8. , 12.5], dtype=float32)
    >>> aux.grad_norms
    Array([3., 4., 5.], dtype=float32)

  Example Usage (with Per-User Clipping):
    >>> f = lambda param, data: 0.5 * jnp.mean((data - param)**2)
    >>> g = clipped_grad(f, l2_clip_norm=jnp.inf, keep_batch_dim=False)
    >>> userA = jnp.array([1, -1])
    >>> userB = jnp.array([2, 2])
    >>> userC = jnp.array([0, 3])
    >>> g(3.0, jnp.array([userA, userB, userC]))
    Array(5.5, dtype=float32)

  Formal Guarantees:
    For the primary output:
      When ``slack`` is set, the primary output contains both the gradient
      and slack fields of :class:`ClippedGradOutput`; otherwise it is the
      gradient PyTree.
      The L2 sensitivity of the returned callable is a complex function of
      the input parameters (``l2_clip_norm``, ``rescale_to_unit_norm``,
      ``normalize_by``, ``grid_scale``, per-layer clipping settings, etc.)
      and may change as new features are added. Rather than reasoning about
      the sensitivity from the parameters, callers should query the returned
      callable's ``.sensitivity()`` method (or equivalently its
      ``.l2_norm_bound`` attribute for add-or-remove-one DP) directly.
    All auxiliary outputs (aux, values, grad_norms) are per-example. This
      function guarantees that per-example outputs only depend on the data for
      the same example. This allows maximum flexibility for the caller to
      aggregate these as desired (possibly with a DP mean, median, quantile,
      or histogram mechanism).

  Args:
    fun: The function to be differentiated, which should return a scalar loss
      value. If ``has_aux`` is ``True``, it should return a tuple ``(value,
      aux)``.
    argnums: Specifies which argument(s) of ``fun`` to differentiate with
      respect to. Can be an integer or a sequence of integers. These arguments
      should *not* have a batch dimension.
    has_aux: If ``True``, ``fun`` is expected to return a tuple ``(value,
      aux)``. The auxiliary data ``aux`` will be returned by the transformed
      function. Exercise caution when using this as no DP sensitivity guarantees
      are provided for the auxiliary data.
    l2_clip_norm: The maximum L2 norm for each per-example gradient. Gradients
      with a norm larger than this value will be scaled down. Can be a single
      float or a PyTree. If Scalar: It does global clipping across all
      gradients. If PyTree: It does per-layer clipping and contains l2 norm
      associated with each leaf of the PyTree. The ``clip_norm`` PyTree
      structure must be a prefix of the input ``pytree``. Where ``clip_norm``
      terminates early, its leaf values automatically broadcast down to match
      all corresponding sub-leaves of ``pytree``.
    rescale_to_unit_norm: If ``True``, the clipped gradient PyTree's norm is
      rescaled by ``1.0 / l2_clip_norm`` after potential clipping. If ``False``,
      each per-example gradient has norm at most ``l2_clip_norm``. Sensitivity
      also depends on ``normalize_by`` and per-layer clipping; use
      ``.sensitivity()`` / ``.l2_norm_bound`` rather than deriving it from these
      flags. The motivation for setting this to ``True`` is to decouple the
      clipping norm from the learning rate for non-adaptive optimizers, as
      described in https://arxiv.org/abs/2204.13650
    normalize_by: Divide the clipped output by this value before returning.
    batch_argnums: Specifies which argument(s) of ``fun`` contain the batch
      dimension (usually the data and labels). Can be an integer or a sequence
      of integers. All arguments specified here must have the same size along
      their first dimension (the batch dimension). The default value of 1
      assumes the signature of fun is ``fun(params, batch)``.
    keep_batch_dim: If ``True``, batch inputs will be passed to ``fun`` with a
      leading batch axis of size 1.  If ``False``, this size 1 axis will be
      dropped (reducing the rank of the batch args by 1 before passing to
      ``fun``). The default value of ``True`` assumes that ``fun`` expects
      inputs with a batch axis. Overriding this default can be useful if fun
      defines the loss function for a single example, or if clipping should be
      applied at the group or user level (in which case an extra batch axis is
      added to the inputs).
    return_values: If ``True``, the transformed function will also return the
      per-example values, before clipping.
    return_grad_norms: If ``True``, the transformed function will also return
      the per-example gradient norms, before clipping. If ``l2_clip_norm`` is a
      scalar, it returns a single global norm for the entire gradient PyTree. If
      ``l2_clip_norm`` is a PyTree, it returns a matching PyTree structure where
      each leaf contains the L2 norm of the corresponding subtree in the
      gradient PyTree.
    pre_clipping_transform: An optional function to apply to the per-example
      gradients before clipping. The function should consume the gradient pytree
      for a single example and return a new pytree (possibly with different
      structure). Can be used to e.g., scale the leaves of the pytree to
      accommodate preconditioner clipping. Does not affect the sensitivity
      guarantee.
    microbatch_size: If set, input groups are formed into microbatches of this
      size. These microbatches are then processed sequentially, with operations
      on the groups within each microbatch being vectorized using
      :func:`jax.vmap`. This can be used to reduce peak memory usage at the cost
      of increased sequential computation. Microbatching will be at the level of
      users/groups.  E.g., if there are 500 users, with 7 examples per user, and
      ``microbatch_size=100``, then the input will be broken into 5 microbatches
      of 100 users, and when processing a microbatch, ``fun`` will be invoked
      100 times (in parallel with vmap) on groups of 7 examples.
    nan_safe: If ``True``, per-example gradients with non-finite L2 norms (NaN
      or inf) are zeroed out before aggregation, preserving the formal
      guarantees.
    dtype: Optional dtype for the returned gradient. If ``None``, the dtype will
      be the same as the dtypes of the gradient function. Can be useful to avoid
      overflow issues when using low-precision dtypes as the returned function
      computes a sum over a potentially large batch.
    prng_argnum: If set, specifies which argument of ``fun`` is a PRNG key. The
      PRNG will be split to have a batch dimension and vmapped over.
    spmd_axis_name: See :func:`jax.vmap`. Only relevant in distributed settings.
    grid_scale: If set, per-example grads are additionally scaled and rounded to
      an integer grid after clipping.  Specifically, each clipped grad is
      multiplied by ``grid_scale / l2_clip_norm``, rounded to the nearest
      integer, and cast to ``jnp.int64``.  The clipping norm is tightened
      automatically so that the integer L2 norm of each rounded output is at
      most ``grid_scale``.  This option is designed for use with the discrete
      Gaussian mechanism.  Incompatible with ``rescale_to_unit_norm=True`` and
      ``normalize_by != 1.0``.  When set, ``dtype`` is ignored (output is always
      ``jnp.int64``).
    slack: Optional number of SlaClip slack coordinates. Must be a positive
      Python integer. When set, the sensitivity-bounded primary output is a
      :class:`ClippedGradOutput`. Its gradient and slack fields must be
      privatized together. The float32 slack is built from the same norm used
      for clipping after ``pre_clipping_transform``. Only scalar global clipping
      is currently supported.

  Returns:
    A :class:`~jax_privacy.clipping.BoundedSensitivityCallable`
    wrapping a new function ``values_and_clipped_grad_fn`` that
    computes the sum of clipped per-group gradients of ``fun``. The
    returned callable accepts the same positional arguments as
    ``fun``, plus an optional ``is_padding_example`` keyword argument
    (see
    :meth:`~jax_privacy.clipping.BoundedSensitivityCallable.__call__`
    for details).

    The returned function returns ``grad`` if ``return_values``,
    ``return_grad_norms``, and ``has_aux`` are all ``False``.
    Otherwise, it returns a tuple ``(grad, AuxiliaryOutput)``, where
    ``AuxiliaryOutput`` is a namedtuple with fields ``(values,
    grad_norms, aux)``.  Each field is set to ``None`` when its
    corresponding flag is ``False``: ``values`` is ``None`` unless
    ``return_values=True``, ``grad_norms`` is ``None`` unless
    ``return_grad_norms=True``, and ``aux`` is ``None`` unless
    ``has_aux=True``.  When present, these outputs are per-example
    (i.e., they retain a batch axis).

    When ``slack`` is set, `grad` is a :class:`ClippedGradOutput`;
    otherwise it retains the original gradient structure.
  """
  _validate_static_args(argnums, batch_argnums, normalize_by)
  fun = _normalize_fun_to_return_aux(fun, has_aux)
  value_and_grad_fn = jax.value_and_grad(fun, argnums, has_aux=True)

  def grad_fn(*args, **kwargs):
    value_and_aux, grad = value_and_grad_fn(*args, **kwargs)
    result = pre_clipping_transform(grad)
    grad_norms_val = jax.tree.map(
        lambda _, g: optax.tree.norm(g), l2_clip_norm, grad
    )
    if has_aux or return_values or return_grad_norms:
      aux = AuxiliaryOutput(
          values=value_and_aux[0] if return_values else None,
          grad_norms=grad_norms_val if return_grad_norms else None,
          aux=value_and_aux[1] if has_aux else None,
      )
      return result, aux
    return result

  return clipped_fun(
      grad_fn,
      has_aux=has_aux or return_values or return_grad_norms,
      batch_argnums=batch_argnums,
      l2_clip_norm=l2_clip_norm,
      keep_batch_dim=keep_batch_dim,
      rescale_to_unit_norm=rescale_to_unit_norm,
      normalize_by=normalize_by,
      microbatch_size=microbatch_size,
      nan_safe=nan_safe,
      dtype=dtype,
      prng_argnum=prng_argnum,
      spmd_axis_name=spmd_axis_name,
      grid_scale=grid_scale,
      slack=slack,
  )

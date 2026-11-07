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

"""Utilities for sharding in multi-machine settings.

This file houses helper functions related to sharding. Users of JAX Privacy
should not need to use this file directly, but the utilities implemented here
are leveraged by higher-level APIs elsewhere in the library. The functions
defined here assume that input arrays are enriched with type-level sharding
information, as described in `JAX Explicit Sharding
<https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html>`_.

This file contains primitives needed for "distributed noise generation"
as described in `Scaling up the Banded Matrix Factorization Mechanism for
Differentially Private ML, McKenna (2024)
<https://arxiv.org/abs/2405.15913>`_
and in `Correlated Noise Mechanisms for Differentially Private Learning,
Pillutla et al. (2025) <https://arxiv.org/abs/2506.08201>`_.
"""

import math
from typing import Any, TypeAlias, TypeVar

import jax
from jax_privacy import _validate
from jax_privacy import optimizers
import numpy as np
import optax


PyTree: TypeAlias = Any
PartitionSpecPyTree: TypeAlias = Any
TransformT = TypeVar(
    'TransformT',
    optax.GradientTransformation,
    optimizers.AugmentedGradientTransformation,
)


def _ceiling_to_multiple(size: int, multiple: int) -> int:
  """Return the smallest multiple of multiple that is >= size."""
  remainder = size % multiple
  return size + multiple - remainder if remainder != 0 else size


def flatten_with_zero_redundancy(
    abstract_array: jax.ShapeDtypeStruct | jax.Array,
) -> jax.ShapeDtypeStruct:
  """Return a flattened, padded, and ZeRo-sharded abstract version of x.

  Specifically, the returned object will describe a 1D array that is
  partitioned across all mesh axes. This is precisely the concept of
  "zero-redundancy" sharding as described in https://arxiv.org/pdf/1910.02054,
  i.e., no redundant copies of the partitioned value exist anywhere.

  Args:
    abstract_array: Abstract (or concrete) array with shape, dtype, and sharding
      attributes.

  Returns:
    A zero-redundancy abstract flattened+padded version of the input value.
  """
  mesh = jax.typeof(abstract_array).sharding.mesh
  # As of JAX 0.7.1, we can use ShapeDtypeStruct with sharding preserved.
  return jax.ShapeDtypeStruct(
      shape=(_ceiling_to_multiple(abstract_array.size, mesh.size),),
      dtype=abstract_array.dtype,
      sharding=jax.sharding.NamedSharding(mesh, jax.P(mesh.axis_names)),
  )


def _flatten_pspec(p: jax.sharding.PartitionSpec) -> jax.sharding.PartitionSpec:
  """Flatten a PartitionSpec from a nD sharding to an "equivalent" 1D sharding.

  Example Usage:
    >>> p = jax.sharding.PartitionSpec(None, ('x', 'y'), None, 'z')
    >>> tuple(_flatten_pspec(p))
    (('x', 'y', 'z'),)

  Example Usage:
    >>> p = jax.sharding.PartitionSpec('data', None, ('replica', 'mdl'))
    >>> tuple(_flatten_pspec(p))
    (('data', 'replica', 'mdl'),)

  Args:
    p: A PartitionSpec defined over a nD array.

  Returns:
    A PartitionSpec defined over the same devices / mesh axes as p, but defined
    wrt a flattened version of the original array.
  """

  result = []
  for item in p:
    if isinstance(item, tuple):
      result.extend(item)
    elif isinstance(item, str):
      result.append(item)
    elif item is None:
      continue
    else:
      raise ValueError(f'Unexpected item in PartitionSpec: {item}.')
  return jax.sharding.PartitionSpec(tuple(result))


def local_reshape_add(x: jax.Array, y: jax.Array) -> jax.Array:
  """Reshapes y[:x.size] into x.shape and x.sharding and adds to x.

  This function expects both x and y to have type-level sharding information,
  See https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html for more
  details.

  Internally, this function uses JAX's shard_map primitive to perform a local
  reshape on each device, avoiding the need to transfer data between devices
  that would arise if using jnp.reshape.

  Note that this function is not quite equivalent to
  x + y[:x.size].reshape(x.shape).  In particular, the precise mapping between
  the entries of y and it's reshaped version is not in general the same
  as jnp.reshape.  We do guarantee (and test) that each entry of y will be
  mapped to at most one entry of the reshaped version.  Since y are typically
  i.i.d. random bits, the precise mapping is not important for our purposes.

  Formal Guarantees:
    - For each index tuple i and j, z_i = x_i + y_j for some j. y_j is only
      mapped to at most one entry of z.
    - z has the same shape, dtype, and sharding as x.

  Args:
    x: The first array.
    y: The second array.  Should be 1D, and sharded across all devices in the
      out_sharding.mesh, and have size greater than or equal to x.size.

  Returns:
    x + reshape(y[:x.size], x.shape), with sharding equal to out_sharding.
  """
  out_sharding = jax.typeof(x).sharding
  per_device_shape = out_sharding.shard_shape(x.shape)
  per_device_size = math.prod(per_device_shape)

  reshape = jax.shard_map(
      lambda v: v[:per_device_size].reshape(per_device_shape),
      mesh=out_sharding.mesh,
      # Replicates input across mesh axes not in out_sharding.spec (as desired).
      in_specs=_flatten_pspec(out_sharding.spec),
      out_specs=out_sharding.spec,
  )
  y = jax.reshard(y, _flatten_pspec(out_sharding.spec))
  return (x + reshape(y)).astype(x.dtype)


def compute_early_stopping_order(
    batch_size: int,
    microbatch_size: int | None,
) -> np.ndarray:
  """Return index permutation so data is processed in order with microbatching.

  To avoid communication in distributed environments with microbatching, data
  data is reshaped from (batch_size, *dims) to (num_microbatches,
  microbatch_size, *dims) using a Fortran-order reshape.

  This is a helper function to reorder data so that they get processed in the
  same order by ``microbatch`` as they would be processed
  without microbatching. This can be particularly helpful when the last elements
  of the batch are padding examples, in which case if they appear in the
  same microbatch we can avoid processing them.  This function is only useful
  if using the "is_padding_example" keyword argument with
  ``microbatch``.

  Example Usage:
    >>> order = compute_early_stopping_order(batch_size=10, microbatch_size=2)
    >>> order
    array([0, 2, 4, 6, 8, 1, 3, 5, 7, 9])

  When permuting the input data to ``microbatch`` according
  to the above permutation, the examples will be split up into 5 microbatches:
  [0, 1], [2, 3], [4, 5], [6, 7], [8, 9] and processed sequentially.

    >>> from optax import microbatching
    >>> microbatching.reshape_batch_axis(order, microbatch_size=2)
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]])

  We can see how this is directly useful in the context of padding below.
  Because the last two microbatches consist of only padding examples,
  ``microbatch`` will skip them, saving compute.

    >>> is_padding = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    >>> microbatching.reshape_batch_axis(is_padding[order], microbatch_size=2)
    array([[0, 0],
           [0, 0],
           [0, 0],
           [1, 1],
           [1, 1]])

  Args:
    batch_size: The size of the batch axis.
    microbatch_size: The target microbatch size that will be used with
      ``microbatch``.

  Returns:
    A permutation of the example indices, where padding examples are evenly
    distributed across the microbatch indices and appear in the last k
    microbatches.  This is useful for early stopping when the true batch size is
    less than the size of the batch axis.
  """
  indices = np.arange(batch_size)
  if microbatch_size is None:
    return indices
  elif batch_size % microbatch_size != 0:
    raise ValueError(
        f'batch_size={batch_size} is not divisible by {microbatch_size=}'
    )
  return indices.reshape(-1, microbatch_size).T.flatten()


def with_sharding_specs(
    transform: TransformT,
    param_specs: PartitionSpecPyTree,
) -> TransformT:
  """Binds parameter PartitionSpecs to a transformation for Auto-axes meshes.

  Requires an active ``AxisType.Auto`` mesh set via :func:`jax.set_mesh`.

  Args:
    transform: An :class:`optax.GradientTransformation` that relies on
      type-level sharding (``AxisType.Explicit``), such as a privatizer
      configured with ``SupportedStrategies.ZERO``.
    param_specs: A PyTree of :class:`jax.sharding.PartitionSpec` or
      :class:`jax.sharding.NamedSharding` matching the parameter/gradient tree.

  Returns:
    An :class:`optax.GradientTransformation` whose ``init`` and ``update``
    temporarily enter ``AxisType.Explicit`` mode with the provided specs.

  Raises:
    ValueError: If ``param_specs`` contains leaves that are neither
      ``PartitionSpec`` nor ``NamedSharding``, or if no ``AxisType.Auto`` mesh
      is active via :func:`jax.set_mesh` when ``init`` or ``update`` is called.
  """

  def _to_pspec(s):
    if isinstance(s, jax.sharding.NamedSharding):
      return s.spec
    if isinstance(s, jax.sharding.PartitionSpec):
      return s
    raise ValueError(
        'Expected PartitionSpec or NamedSharding leaves in param_specs, got'
        f' {type(s)}.'
    )

  specs = jax.tree.map(_to_pspec, param_specs)

  @jax.jit
  def init_fn(params):
    _validate.explicit_mesh(status=False)
    return jax.sharding.explicit_axes(transform.init, in_sharding=(specs,))(
        params
    )

  @jax.jit
  def update_fn(updates, state, params=None):
    del params
    _validate.explicit_mesh(status=False)

    def _update(updates):
      # Trace transform.init inside explicit_axes to read the Explicit state
      # PartitionSpecs from types; @jax.jit DCEs the unused init ops.
      state_specs = jax.tree.map(
          lambda x: jax.typeof(x).sharding.spec, transform.init(updates)
      )
      return transform.update(updates, jax.sharding.reshard(state, state_specs))

    return jax.sharding.explicit_axes(_update, in_sharding=(specs,))(updates)

  return transform._replace(init=init_fn, update=update_fn)

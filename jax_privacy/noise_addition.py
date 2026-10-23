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

"""Implementations of optax.GradientTransformations that add noise to gradients.

This module implements optax.GradientTransformations, which we informally call
``privatizers``, that are responsible for taking clipped + aggregated
gradients and adding noise to them. These noise-addition schemes are *stateful*,
meaning the noise added to one gradient may depend on the noise that was added
to previous gradients in various ways. In the simplest case, where i.i.d.
gaussian noise is added to each gradient, this state is nothing more than a
pseudo-random key, each call to ``update`` uses this key to generate fresh
noise, and splits it into a new key for future steps.

Example Usage:
  >>> import jax
  >>> privatizer = gaussian_privatizer(stddev=1.0, prng_key=jax.random.key(0))
  >>> model = grad = jax.numpy.zeros(10)
  >>> noise_state = privatizer.init(model)
  >>> for _ in range(4):
  ...   noisy_grad, noise_state = privatizer.update(
  ...     sum_of_clipped_grads=grad, noise_state=noise_state
  ...   )

More powerful privatizers, like those based on matrix factorization have
richer state representations, but this is abstracted away from the user via
the optax.GradientTransformation interface. Different privatizers are fully
swappable with each other using the above pattern with only one line of code
changed.

As optax.GradientTransformations, these privatizers can be composed with other
transformations, via optax.chain(privatizer, optimizer). These transformed
privatizers enjoy the same privacy properties by the post-processing property.
"""

import enum
import functools
from typing import NamedTuple, Protocol

import jax
from jax import numpy as jnp
import numpy as np
import optax

from . import sharding_utils
from .matrix_factorization import streaming_matrix


class _NoiseStructureFn(Protocol):
  """A function that returns the intermediate shape/sharding of noise.

  (Expected) Formal Guarantees of y = get_noise_structure(x):
    - x.size <= y.size (i.e., math.prod(x.shape) <= math.prod(y.shape)). See
    property (2) of _NoiseAdder.
  """

  def __call__(self, value_info: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
    """Return the abstract info of the intermediate noise from the value."""


class _NoiseAdder(Protocol):
  """Protocol specifying the semantics of _IntermediateShardingStrategy.add.

  Will be called as add(value, noise), where the first argument represents the
  input and can have arbitrary shape/sharding while the second arguments
  represents the noise and will have shape/sharding
  specified by get_noise_structure (subject to the formal guarantees above).

  (Expected) Formal Guarantees for z = add(x, y):
    - Each Element z_i is equal to x_i + y_j for some j.
    - Each element y_j is added to at most one element x_i.
    - z has the same shape/dtype/sharding as x.
  """

  def __call__(self, value: jax.Array, noise: jax.Array) -> jax.Array:
    """Add the noise to the input tensor, possibly via a reshape operation."""


class _IntermediateShardingStrategy(NamedTuple):
  """Specifies an implementation strategy for adding noise to values.

  Noise may be generated with a different shape and/or sharding than the
  inputs. This class holds implementations to determine the shape/sharding
  of the intermediates, and a function to add two arrays with potentially
  different shapes/shardings. For example usage, see
  ``_dense_matrix_factorization_privatizer`` below.
  """

  get_noise_structure: _NoiseStructureFn
  add: _NoiseAdder


# Our public API expects an Enum to prevent users from passing in a custom
# implementation that breaks the formal guarantees. Custom implementations
# can be used, but should be checked in and reviewed by JAX Privacy authors.
class ShardingStrategies(enum.Enum):
  """Sharding strategies for generating intermediate noise."""

  DEFAULT = _IntermediateShardingStrategy(
      get_noise_structure=lambda value_info: value_info,
      add=lambda value, noise: (value + noise).astype(value.dtype),
  )
  """Basic approach for single-machine training scenarios."""
  ZERO = _IntermediateShardingStrategy(
      get_noise_structure=sharding_utils.flatten_with_zero_redundancy,
      add=sharding_utils.local_reshape_add,
  )
  """Zero-redundancy approach suitable for multi-machine scenarios. Requires
  inputs to have explicit sharding annotations (or wrap the privatizer with
  :func:`~jax_privacy.sharding_utils.with_sharding_specs` on ``AxisType.Auto``
  meshes)."""


# Deprecated alias for ShardingStrategies.
SupportedStrategies = ShardingStrategies


def matrix_factorization_privatizer(
    noising_matrix: jax.typing.ArrayLike | streaming_matrix.StreamingMatrix,
    *,
    stddev: float,
    prng_key: jax.Array | int | None = None,
    dtype: jax.typing.DTypeLike | None = None,
    intermediate_strategy: ShardingStrategies = ShardingStrategies.DEFAULT,
) -> optax.GradientTransformation:
  """Creates a gradient privatizer that adds correlated noise to gradients.

  This implementation is described in Section 4.4 of `Correlated Noise
  Mechanisms for Differentially Private Learning
  <https://arxiv.org/abs/2506.08201>`_. A different implementation will be used
  depending on whether the ``noising_matrix`` is a :class:`jax.Array` or a
  :class:`~jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix`.
  The dtype of the noise generated by this privatizer will be determined by the
  ``noising_matrix`` and the input gradients according to standard jax type
  promotion rules. The output of the privatize transformation will always match
  the input dtype.

  For naming of these parameters, see the `matrix_factorization README
  <https://github.com/google-parfait/jax_privacy/blob/main/jax_privacy/matrix_factorization/README.md>`_.

  Args:
    noising_matrix: A matrix used to generate correlated noise. Noise samples
      will be distributed according to a multivariate Gaussian with covariance
      matrix ``noising_matrix @ noising_matrix.T``.
    stddev: Standard deviation to use for the noise of this privatizer.
    prng_key: An optional PRNGKey array representing the source of randomness.
    dtype: The dtype to use for intermediate noise. If specified, noise will be
      generated with this dtype, added to the input gradient according to normal
      jax type promotion rules, and then cast back to the gradient dtype.
    intermediate_strategy: Strategy to use for generating intermediate noise.

  Returns:
    An :class:`optax.GradientTransformation` which adds samples from Gaussian
    correlated by ``noising_matrix`` (i.e., samples from a Gaussian with
    covariance ``noising_matrix @ noising_matrix.T``), keyed by ``prng_key``,
    to its stream of gradients.
  """
  if prng_key is None:
    prng_key = jax.random.key(np.random.randint(0, 2**31))
  elif isinstance(prng_key, int):
    prng_key = jax.random.key(prng_key)
  if isinstance(noising_matrix, jax.typing.ArrayLike):
    impl = _dense_matrix_factorization_privatizer
  elif isinstance(noising_matrix, streaming_matrix.StreamingMatrix):
    impl = _streaming_matrix_factorization_privatizer
  else:
    raise NotImplementedError(f'Unsupported noising_matrix: {noising_matrix}')

  return impl(
      noising_matrix,
      prng_key=prng_key,
      stddev=stddev,
      strategy=intermediate_strategy.value,
      dtype=dtype,
  )


gaussian_privatizer = functools.partial(
    matrix_factorization_privatizer,
    streaming_matrix.identity(),
)
gaussian_privatizer.__doc__ = (
    """Constructs a gradient privatizer that adds isotropic Gaussian noise."""
)


_MAX_REPLAY_CHUNKS = 8


@jax.jit(static_argnames=['transform'])
def _init_for_step(transform, structure, step):
  """Initializes ``transform`` and advances its state to ``step``."""
  zeros = optax.tree.zeros_like(structure)
  body = lambda _, s: transform.update(zeros, s)[1]
  leaves, treedef = jax.tree.flatten(transform.init(structure))
  # Partition state leaves into byte-balanced chunks so XLA's while-loop DCE
  # prunes inactive leaves and bounds temporary roll scratch.
  num_chunks = max(1, min(_MAX_REPLAY_CHUNKS, len(leaves)))
  chunk_bytes = [0] * num_chunks
  leaf_chunk = []
  for leaf in leaves:
    c = min(range(num_chunks), key=chunk_bytes.__getitem__)
    leaf_chunk.append(c)
    chunk_bytes[c] += leaf.size * leaf.dtype.itemsize

  out_leaves = list(leaves)
  for c in range(num_chunks):
    state = jax.lax.fori_loop(
        jnp.zeros_like(step), step, body, transform.init(structure)
    )
    for i, leaf in enumerate(jax.tree.leaves(state)):
      if leaf_chunk[i] == c:
        out_leaves[i] = leaf
  return treedef.unflatten(out_leaves)


def init_for_step(
    transform: optax.GradientTransformation,
    structure: optax.ArrayTree,
    step: jax.typing.ArrayLike = 0,
) -> optax.OptState:
  """Initializes ``transform`` and advances its state to ``step``.

  Args:
    transform: A stateful noise-addition :class:`optax.GradientTransformation`.
    structure: Example gradient PyTree passed to ``transform.init``.
    step: Non-negative step index to fast-forward the state to.

  Returns:
    The state of ``transform`` after ``step`` updates.
  """
  return _init_for_step(transform, structure, jnp.asarray(step, jnp.int32))


def _compute_loop_bounds(matrix_row):
  """Computes bounds for jax.lax.fori_loop version of row-column reduction."""
  assert matrix_row.ndim == 1
  nonzero_entries = (matrix_row != 0).astype(jnp.int32)
  first_nonzero = jnp.argmax(nonzero_entries)
  # We reverse to get the last index.
  last_nonzero = matrix_row.shape[0] - jnp.argmax(nonzero_entries[::-1])
  return first_nonzero, last_nonzero


def _gaussian_linear_combination(
    matrix_row: jax.Array,
    key: jax.Array,
    shape: tuple[int, ...],
    dtype: jax.typing.DTypeLike,
    out_sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
  """Computes a linear combination of standard Gaussian random variables."""
  assert matrix_row.ndim == 1

  def loop_body(idx, partial):
    coef = matrix_row[idx].astype(dtype)
    sub_key = jax.random.fold_in(key, idx)
    # We pass sharding because sharding-in-types will replicate otherwise.
    noise = jax.random.normal(sub_key, shape, dtype, out_sharding=out_sharding)
    return partial + coef * noise

  lower_bound, upper_bound = _compute_loop_bounds(matrix_row)
  loop_state = jnp.zeros(shape, dtype)
  return jax.lax.fori_loop(lower_bound, upper_bound, loop_body, loop_state)


def _dense_matrix_factorization_privatizer(
    noising_matrix: jax.typing.ArrayLike,
    *,
    stddev: float,
    prng_key: jax.Array,
    strategy: _IntermediateShardingStrategy,
    dtype: jax.typing.DTypeLike | None = None,
) -> optax.GradientTransformation:
  """Creates a gradient privatizer from a dense matrix C^{-1}."""
  # See Section 4.4.5 of https://arxiv.org/pdf/2506.08201 (Approach 2)
  noising_matrix = jnp.asarray(noising_matrix)

  if noising_matrix.ndim != 2:
    raise ValueError(f'Expected 2D matrix, found {noising_matrix.shape=}.')

  def update(sum_of_clipped_grads, noise_state, params=None):
    del params  # Unused, but expected by optax.GradientTransformation API.
    index = noise_state
    matrix_row = noising_matrix[index] * stddev

    target = jax.tree.map(strategy.get_noise_structure, sum_of_clipped_grads)
    noise = optax.tree.random_like(
        rng_key=prng_key,
        target_tree=target,
        sampler=functools.partial(_gaussian_linear_combination, matrix_row),
        dtype=dtype,
    )
    noisy_grads = jax.tree.map(strategy.add, sum_of_clipped_grads, noise)
    return noisy_grads, index + 1

  init = lambda _: jnp.array(0)
  return optax.GradientTransformation(init, update)


def _iid_normal_noise(prng_key, target_tree, stddev, dtype=None):
  standard_normal = optax.tree.random_like(
      rng_key=prng_key,
      target_tree=target_tree,
      sampler=jax.random.normal,
      dtype=dtype,
  )
  return optax.tree.scale(stddev, standard_normal)


def _streaming_matrix_factorization_privatizer(
    noising_matrix: streaming_matrix.StreamingMatrix,
    *,
    stddev: float,
    prng_key: jax.Array,
    strategy: _IntermediateShardingStrategy,
    dtype: jax.typing.DTypeLike | None = None,
) -> optax.GradientTransformation:
  """Creates a gradient privatizer from a StreamingMatrix C^{-1}."""

  def init(model):
    intermediate = jax.tree.map(strategy.get_noise_structure, model)
    if dtype is not None:
      to_dtype = lambda x: jax.ShapeDtypeStruct(
          x.shape, dtype, sharding=jax.ShapeDtypeStruct.like(x).sharding
      )
      intermediate = jax.tree.map(to_dtype, intermediate)
    # downstream users may donate this state to JIT, so we copy prng_key here.
    return jnp.copy(prng_key), noising_matrix.init_multiply(intermediate)

  def update(sum_of_clipped_grads, noise_state, params=None):
    del params  # Unused, but expected by optax.GradientTransformation API.
    prng_key, inner_state = noise_state
    new_key, sub_key = jax.random.split(prng_key)

    target = jax.tree.map(strategy.get_noise_structure, sum_of_clipped_grads)
    iid_noise = _iid_normal_noise(sub_key, target, stddev, dtype)
    corr_noise, new_state = noising_matrix.multiply_next(iid_noise, inner_state)

    noisy_grads = jax.tree.map(strategy.add, sum_of_clipped_grads, corr_noise)
    return noisy_grads, (new_key, new_state)

  return optax.GradientTransformation(init, update)

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

"""Randomness, seeding, and sampling utilities for jax_privacy."""

import copy
import math
import secrets
from typing import Protocol

from absl import logging
import jax
from jax.experimental import multihost_utils
from jax_privacy import _validate
from jax_privacy import sharding_utils
import numpy as np
import optax


def secure_seed(bits: int = 31) -> int:
  """Returns a cryptographically secure random integer shared across hosts."""
  _validate.positive(bits=bits)
  raw = secrets.randbits(bits).to_bytes(math.ceil(bits / 8), 'big')
  arr = np.frombuffer(raw, dtype=np.uint8)
  shared = multihost_utils.broadcast_one_to_all(arr)
  return int.from_bytes(np.asarray(shared, dtype=np.uint8).tobytes(), 'big')


def default_rng(
    rng: np.random.Generator | int | None = None,
) -> np.random.Generator:
  """Returns a Generator, drawing a shared secure seed if None."""
  if rng is None:
    rng = secure_seed()
  rng = np.random.default_rng(rng)
  _validate.equal_across_hosts(rng=copy.deepcopy(rng).integers(2**31))
  return rng


def default_prng_key(
    rng: np.random.Generator | int | None = None,
) -> jax.Array:
  """Returns a JAX PRNGKey from a Generator, seed, or None."""
  seed = default_rng(rng).integers(2**31)
  return jax.random.key(int(seed))


def resolve_rng(
    rng: np.random.Generator | int | None = None,
    *,
    noise_seed: int | None = None,
    prng_key: jax.Array | int | None = None,
) -> np.random.Generator | int | None:
  """Resolves ``rng`` while handling deprecated RNG keyword arguments.

  Centralizes backwards-compatibility handling for public functions migrating
  from legacy RNG arguments to ``rng``. At most one argument may be
  non-``None``. When a deprecated argument is provided, logs a deprecation
  warning, converts a ``jax.Array`` PRNG key outside JIT to an integer seed,
  and returns the resolved value (typically forwarded to :func:`default_rng` or
  :func:`default_prng_key`). Once the deprecated arguments are removed in
  ``jax_privacy`` 2.6, callers can drop the ``resolve_rng`` call and pass
  ``rng`` directly.

  Example::

    rng = _random.resolve_rng(rng, noise_seed=noise_seed)
    rng = _random.default_rng(rng)

  Args:
    rng: Preferred RNG argument (a ``numpy.random.Generator``, integer seed, or
      ``None``).
    noise_seed: Deprecated legacy integer seed argument.
    prng_key: Deprecated legacy JAX PRNG key or integer seed argument.

  Returns:
    The resolved ``numpy.random.Generator``, integer seed, or ``None``.
  """
  legacy = dict(noise_seed=noise_seed, prng_key=prng_key)
  active = [(k, v) for k, v in legacy.items() if v is not None]
  if len(active) + (rng is not None) > 1:
    names = (['rng'] if rng is not None else []) + [k for k, _ in active]
    raise ValueError(f'At most one RNG argument may be set, got {names}.')
  if not active:
    return rng
  name, value = active[0]
  logging.warning(
      '%s is deprecated and will be removed in jax_privacy 2.6 (12/26);'
      ' use rng instead.',
      name,
  )
  if isinstance(value, jax.Array):
    _validate.jitted(False)
    # Preserve every key word, using a fixed byte order across platforms.
    key_data = np.asarray(jax.random.key_data(value), dtype='>u4')
    return int.from_bytes(key_data.tobytes(), 'big')
  return value


class Sampler(Protocol):
  """Protocol for NumPy random sampling functions."""

  def __call__(
      self,
      rng: np.random.Generator,
      *,
      size: int | tuple[int, ...],
      dtype: jax.typing.DTypeLike | None = None,
  ) -> np.ndarray:
    ...


def _parallel_sample(
    rng: np.random.Generator,
    shape: int | tuple[int, ...],
    sharding: jax.sharding.NamedSharding | None,
    sampler: Sampler = np.random.Generator.standard_normal,
    dtype: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Helper function for parallel_sample_pytree that works on a single leaf."""

  _validate.jitted(False)

  if sharding is None:
    return jax.numpy.asarray(sampler(rng, size=shape, dtype=dtype))

  _validate.explicit_mesh(status=True, device_mesh=sharding.mesh)

  if isinstance(shape, int):
    shape = (shape,)
  num_devices = sharding.mesh.size
  per_device_size = math.ceil(math.prod(shape) / num_devices)
  size = per_device_size * num_devices

  full_sharding = jax.sharding.NamedSharding(
      sharding.mesh, jax.sharding.PartitionSpec(sharding.mesh.axis_names)
  )

  # In multi-machine settings, all CPUs will run the same code in parallel.
  # If the rng implementation is `pure` (only depends on the seed), then all
  # machines have consistent states. If it is not (e.g., depends on system
  # randomness), then rngs[i] might be different across machines. However, each
  # rng is only used once since we are generating an array that is fully sharded
  # across all devices in the mesh, so the lack of cross-machine consistency
  # is not an issue.
  rngs = rng.spawn(num_devices)

  def local_fn(s):
    start, _, _ = s[0].indices(size)
    i = start // per_device_size if per_device_size else 0
    return sampler(rngs[i], size=per_device_size, dtype=dtype)

  fully_sharded_result = jax.make_array_from_callback(
      (size,), full_sharding, local_fn
  )
  # TODO: b/415360727 - Materialization of this array of zeros is suboptimal.
  result = jax.numpy.zeros(shape, dtype=dtype, out_sharding=sharding)
  return sharding_utils.local_reshape_add(result, fully_sharded_result)


def parallel_sample_pytree(
    rng: np.random.Generator,
    struct: optax.ArrayTree,
    sampler: Sampler = np.random.Generator.standard_normal,
) -> optax.ArrayTree:
  """Sample a pytree with numpy and convert it to a sharded JAX pytree.

  This function is intended to bridge the gap between JAX programs and
  external sources of randomness, e.g., generating on CPU with numpy. This
  function is defined in terms of a np.random.Generator input, such as those
  defined by the ``randomgen`` python library. This function parallelizes the
  sampling across all devices defined by the mesh, then communicates the
  samples between devices as necessary to produce the desired output sharding.

  Args:
    rng: The random number generator to use.
    struct: A PyTree of jax.Array or jax.ShapeDtypeStruct objects, defining the
      shape, sharding, and dtype of the output pytree.
    sampler: The random sampling function to use.

  Returns:
    A PyTree with the same structure as ``struct``, with the sampled values.
  """
  _validate.jitted(False)

  treedef = jax.tree.structure(struct)
  rngs = jax.tree.unflatten(treedef, rng.spawn(treedef.num_leaves))

  def leaf_fn(leaf, rng):
    return _parallel_sample(rng, leaf.shape, leaf.sharding, sampler, leaf.dtype)

  return jax.tree.map(leaf_fn, struct, rngs)

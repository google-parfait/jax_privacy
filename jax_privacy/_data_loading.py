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

"""Data loading and batch iteration for jax_privacy training loops."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import concurrent.futures
import copy
import functools
import itertools
from typing import Any, TYPE_CHECKING, TypeAlias

from absl import logging
import jax
from jax_privacy import _validate
from jax_privacy import batch_selection
import numpy as np
import optax

# Guarded by TYPE_CHECKING so grain remains an optional runtime dependency.
if TYPE_CHECKING:
  import grain.python as grain  # pylint: disable=g-import-not-at-top

PyTree: TypeAlias = Any
BatchAndPadding: TypeAlias = tuple[optax.ArrayTree, jax.Array]


def is_map_dataset(dataset: Any) -> bool:
  """Returns whether ``dataset`` is a PyGrain MapDataset or data source."""
  names = [c.__name__ for c in type(dataset).__mro__]
  return any(x in ("MapDataset", "RandomAccessDataSource") for x in names)


def _from_pytree(
    dataset: optax.ArrayTree,
    strategy: batch_selection.BatchSelectionStrategy,
    *,
    rng: np.random.Generator,
    padding_example: Any = None,
    pad_to_multiple_of: int = 1,
    microbatch_size: int | None = None,
    initial_step: int = 0,
    sharding: jax.sharding.NamedSharding | None = None,
    max_workers: int | None = None,
) -> Iterator[BatchAndPadding]:
  """Yields padded (batch, is_padding_example) pairs from an array PyTree."""
  del max_workers
  num_examples = _validate.batched(dataset, padding_example)
  _validate.equal_across_hosts(dataset=jax.tree.map(lambda x: x[0], dataset))
  index_iterator = strategy.batch_iterator(num_examples, rng=rng)

  # When sharding across a mesh (including multi-controller JAX),
  # make_array_from_callback slices and transfers only the process-addressable
  # shards to HBM. In multi-controller setups, every process must hold the
  # full dataset and use the same rng so each process samples the same global
  # batch indices before slicing its local device shards.
  def _to_device(x: np.ndarray) -> jax.Array:
    if sharding is None:
      return jax.device_put(x)
    return jax.make_array_from_callback(x.shape, sharding, lambda idx: x[idx])

  def _index_and_zero(x, indices):
    mask = np.expand_dims(indices == -1, tuple(range(1, x.ndim)))
    return _to_device(np.where(mask, 0, x[indices]))

  # Fast-forward the index iterator to initial_step when resuming from a
  # checkpoint, without materializing or transferring skipped batches.
  # TODO: Investigate stateful iterator instead of rng replay.
  for indices in itertools.islice(index_iterator, initial_step, None):
    indices = batch_selection.pad_to_multiple_of(
        indices, pad_to_multiple_of, microbatch_size=microbatch_size
    )
    fn = functools.partial(_index_and_zero, indices=indices)
    yield jax.tree.map(fn, dataset), _to_device(indices == -1)


def _read_batch(
    indices: np.ndarray,
    get_element: Callable[[int], PyTree],
    sharding: jax.sharding.NamedSharding | None,
    executor: concurrent.futures.Executor,
) -> BatchAndPadding:
  """Fetches local batch shards in parallel and places them on device.

  When sharding is provided, uses jax.make_array_from_callback so each JAX
  process only fetches and transfers the batch-axis slices addressed by its
  local devices (including multi-controller setups where multiple processes
  share a replicated batch shard). Slice reads are cached per call so multi-leaf
  PyTrees and local devices sharing a replicated axis read and stack each slice
  at most once.

  Formal Guarantees:
    - Assuming indices is identical across all JAX processes, position j along
      axis 0 of batch equals get_element(int(indices[j])) across all PyTree
      leaves, and is_padding_example[j] equals (indices[j] == -1).
    - get_element(-1) is used to obtain the padding template and for any -1
      entries in indices, and each unique addressable batch slice is fetched via
      executor.map at most once per call.

  Args:
    indices: 1D integer array of global example indices (with -1 denoting
      padding positions). Must match across all JAX processes.
    get_element: Callable mapping an example index to a single-example PyTree of
      arrays; get_element(-1) must return a data-independent zero padding
      example defining the expected PyTree structure, shapes, and dtypes.
    sharding: Optional NamedSharding partitioning at most axis 0 (the batch
      dimension), leaving feature dimensions unsharded.
    executor: Executor used to parallelize get_element calls within each local
      batch slice.

  Returns:
    A (batch, is_padding_example) pair on device with leading dimension
    len(indices) and sharding (if provided).
  """
  n = len(indices)
  is_padding = indices == -1
  padding = get_element(-1)
  # Flatten PyTree leaves because make_array_from_callback builds one jax.Array
  # leaf at a time, whereas get_element returns all leaves of an example.
  pad_leaves, treedef = jax.tree.flatten(padding)

  @functools.cache
  def _load_slice(batch_slice: slice) -> list[np.ndarray]:
    # Determine which batch elements this process needs and read them from disk.
    local_indices = indices[batch_slice]
    local_elements = list(executor.map(get_element, local_indices))
    # Prepend padding and slice [1:] to handle empty slices cleanly.
    stack = lambda *xs: np.stack(xs)[1:]
    stacked: PyTree = jax.tree.map(stack, padding, *local_elements)
    return jax.tree.leaves(stacked)

  if sharding is None:
    leaves = _load_slice(slice(None))
    batch: PyTree = jax.tree.unflatten(treedef, leaves)
    return jax.device_put((batch, is_padding))

  def _make_leaf(i: int, pad: np.ndarray) -> jax.Array:
    global_shape = (n, *pad.shape)
    # _load_slice loads the full PyTree while we select only leaf i here;
    # because _load_slice is cached, the slice is only read once across leaves.
    callback_fn = lambda shard_slices: _load_slice(shard_slices[0])[i]
    # Omit dtype so 64-bit NumPy leaves narrow silently like jax.device_put.
    return jax.make_array_from_callback(global_shape, sharding, callback_fn)

  leaves = [_make_leaf(i, pad) for i, pad in enumerate(pad_leaves)]
  batch = jax.tree.unflatten(treedef, leaves)
  callback_fn = lambda shard_slices: is_padding[shard_slices]
  is_pad = jax.make_array_from_callback((n,), sharding, callback_fn)
  return batch, is_pad


def _from_pygrain(
    dataset: grain.RandomAccessDataSource,
    strategy: batch_selection.BatchSelectionStrategy,
    *,
    rng: np.random.Generator,
    padding_example: Any = None,
    pad_to_multiple_of: int = 1,
    microbatch_size: int | None = None,
    initial_step: int = 0,
    sharding: jax.sharding.NamedSharding | None = None,
    max_workers: int | None = None,
) -> Iterator[BatchAndPadding]:
  """Yields padded (batch, is_padding_example) pairs from a MapDataset."""
  if padding_example is None:
    # Assumes uniform record shapes; mismatched shapes fail downstream anyway.
    logging.info("Inferring padding_example from dataset[0].")
    padding_example = dataset[0]
  _validate.equal_across_hosts(dataset=dataset[0])
  index_iterator = strategy.batch_iterator(len(dataset), rng=copy.deepcopy(rng))
  padding_element = jax.tree.map(
      lambda x: np.zeros(x.shape, dtype=x.dtype), padding_example
  )

  # TODO: Skip or zero out MapDataset elements whose structure,
  # shape, or dtype does not match padding_example, and document this as a
  # formal guarantee.
  def _get_element(idx: int) -> Any:
    return padding_element if idx == -1 else dataset[idx]

  with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
    for indices in itertools.islice(index_iterator, initial_step, None):
      indices = batch_selection.pad_to_multiple_of(
          indices, pad_to_multiple_of, microbatch_size=microbatch_size
      )
      yield _read_batch(indices, _get_element, sharding, executor)


def batch_iterator(
    dataset: Any,
    strategy: batch_selection.BatchSelectionStrategy,
    *,
    rng: np.random.Generator,
    padding_example: Any = None,
    pad_to_multiple_of: int = 1,
    microbatch_size: int | None = None,
    initial_step: int = 0,
    sharding: jax.sharding.NamedSharding | None = None,
    max_workers: int | None = None,
) -> Iterator[BatchAndPadding]:
  """Returns an iterator of padded (batch, is_padding_example) pairs.

  Formal Guarantees:
    - Non-padding examples are sampled according to ``strategy.batch_iterator``,
      starting from ``initial_step``.
    - Padding positions (where ``indices == -1``) are zeroed out in ``batch``
      (having no dependence on ``dataset`` when ``padding_example`` is
      specified) and marked ``True`` in ``is_padding_example``.

  Args:
    dataset: A PyTree of arrays where the first axis of each leaf is the batch /
      example dimension, or a ``grain.MapDataset`` (or
      ``grain.RandomAccessDataSource``).
    strategy: Batch selection strategy producing per-step example indices.
    rng: Random number generator for batch sampling.
    padding_example: Optional single-example PyTree specification (arrays or
      ``jax.ShapeDtypeStruct``) defining the structure, shapes, and dtypes of
      zero-padded examples. If omitted on a ``MapDataset``, it is inferred from
      ``dataset[0]``.
    pad_to_multiple_of: Pads each batch size to a multiple of this integer.
    microbatch_size: Optional microbatch size used to reorder padding examples
      for early stopping.
    initial_step: Step index to fast-forward the batch iterator to when resuming
      from a checkpoint.
    sharding: Optional ``NamedSharding`` for placing batches across a device
      mesh.
    max_workers: Optional thread pool size for parallel element retrieval when
      ``dataset`` is a ``MapDataset``.

  Returns:
    An iterator yielding ``(batch, is_padding_example)`` pairs on device.
  """
  kwargs = dict(
      padding_example=padding_example,
      rng=rng,
      pad_to_multiple_of=pad_to_multiple_of,
      microbatch_size=microbatch_size,
      initial_step=initial_step,
      sharding=sharding,
      max_workers=max_workers,
  )
  fn = _from_pygrain if is_map_dataset(dataset) else _from_pytree
  return fn(dataset, strategy, **kwargs)

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

from collections.abc import Iterator
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

BatchAndPadding: TypeAlias = tuple[optax.ArrayTree, jax.Array]


def is_map_dataset(dataset: Any) -> bool:
  """Returns whether ``dataset`` is a PyGrain MapDataset or data source."""
  names = [c.__name__ for c in type(dataset).__mro__]
  return any(x in ("MapDataset", "RandomAccessDataSource") for x in names)


def _from_pytree(
    dataset: optax.ArrayTree,
    strategy: batch_selection.BatchSelectionStrategy,
    *,
    padding_example: Any = None,
    rng: batch_selection.RngType = None,
    pad_to_multiple_of: int = 1,
    microbatch_size: int | None = None,
    initial_step: int = 0,
    sharding: jax.sharding.NamedSharding | None = None,
    max_workers: int | None = None,
) -> Iterator[BatchAndPadding]:
  """Yields padded (batch, is_padding_example) pairs from an array PyTree."""
  del max_workers
  num_examples = _validate.batch(dataset, padding_example)
  _validate.multihost(jax.tree.map(lambda x: x[0], dataset), rng)
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


def _from_pygrain(
    dataset: grain.RandomAccessDataSource,
    strategy: batch_selection.BatchSelectionStrategy,
    *,
    padding_example: Any = None,
    rng: batch_selection.RngType = None,
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
  _validate.multihost(dataset[0], rng)
  index_iterator = strategy.batch_iterator(len(dataset), rng=copy.deepcopy(rng))
  padding_element = jax.tree.map(
      lambda x: np.zeros(x.shape, dtype=x.dtype), padding_example
  )

  # TODO: Skip or zero out MapDataset elements whose structure,
  # shape, or dtype does not match padding_example, and document this as a
  # formal guarantee.
  def _get_element(idx: int) -> Any:
    return padding_element if idx == -1 else dataset[idx]

  # Wrap process-local batch slices into a batch-sharded global jax.Array via
  # safe_sharding first, then reshard to the caller's target sharding (which
  # may replicate across some mesh axes).
  def _to_device(x: np.ndarray) -> jax.Array:
    if sharding is None:
      return jax.device_put(x)
    safe_sharding = jax.sharding.NamedSharding(
        sharding.mesh, jax.P(sharding.mesh.axis_names)
    )
    x = jax.make_array_from_process_local_data(safe_sharding, x)
    return jax.device_put(x, sharding)

  with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
    for indices in itertools.islice(index_iterator, initial_step, None):
      indices = batch_selection.pad_to_multiple_of(
          indices, pad_to_multiple_of, microbatch_size=microbatch_size
      )
      # In multi-controller JAX, every process samples the same global indices
      # from the shared rng, then slices its disjoint 1 / process_count share so
      # each host only fetches and transforms its local records in parallel.
      if sharding is not None:
        indices = np.split(indices, jax.process_count())[jax.process_index()]
      elements = list(executor.map(_get_element, indices))
      # Prepend padding_element and slice [1:] to handle empty batches.
      batch = jax.tree.map(
          lambda *leaves: _to_device(np.stack(leaves)[1:]),
          padding_element,
          *elements,
      )
      yield batch, _to_device(indices == -1)


def batch_iterator(
    dataset: Any,
    strategy: batch_selection.BatchSelectionStrategy,
    *,
    padding_example: Any = None,
    rng: batch_selection.RngType = None,
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
    padding_example: Optional single-example PyTree specification (arrays or
      ``jax.ShapeDtypeStruct``) defining the structure, shapes, and dtypes of
      zero-padded examples. If omitted on a ``MapDataset``, it is inferred from
      ``dataset[0]``.
    rng: Optional random seed or ``numpy.random.Generator`` for batch sampling.
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

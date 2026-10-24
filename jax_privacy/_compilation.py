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

"""Ahead-of-time compilation helpers for the DP training loop.

This module owns the machinery that warms up the ``train_step`` JIT cache by
asynchronously ahead-of-time (AOT) compiling it for the batch sizes a run will
encounter.  It is deliberately kept separate from :mod:`jax_privacy.training`
so the core training loop stays independent of compilation concerns.  The
runtime dependency is one-directional (``training`` imports this module, not the
reverse); the type-only dependency on :class:`~jax_privacy.training.DPTrainer`
is guarded by ``TYPE_CHECKING`` to avoid an import cycle.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import copy
import dataclasses
import functools
import math
from typing import Any, TYPE_CHECKING, TypeAlias

from absl import logging
import jax
from jax_privacy import _data_loading
from jax_privacy import _validate
from jax_privacy import batch_selection
import numpy as np

if TYPE_CHECKING:
  from jax_privacy import training  # pylint: disable=g-import-not-at-top

PrecompiledFuture: TypeAlias = concurrent.futures.Future[jax.stages.Compiled]

# Shared thread pool for background ahead-of-time compilation.
_COMPILE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# JAX config flag that lowers large closed-over constants as arguments to the
# compiled executable instead of baking them into the HLO as literals.
_SIMPLIFIED_JAXPR_CONSTANTS_FLAG = "jax_use_simplified_jaxpr_constants"
_EMBEDDED_CONSTANTS_MAX_BYTES_FLAG = "jax_embedded_constants_max_bytes"


@contextlib.contextmanager
def hoist_closed_over_constants():
  """Lowers large closed-over constants as arguments rather than HLO literals.

  DP fine-tuning loss functions typically close over a large frozen base model
  (this is the documented :class:`~jax_privacy.training.LossFn` contract). By
  default JAX materializes such closed-over arrays into the compiled HLO as
  constants, which inflates compilation time and peak memory, discards input
  sharding, and can OOM for large models. Enabling
  ``jax_use_simplified_jaxpr_constants`` instead hoists them into the compiled
  executable's signature as regular arguments.
  """
  previous = getattr(jax.config, _SIMPLIFIED_JAXPR_CONSTANTS_FLAG)
  previous_max_bytes = getattr(jax.config, _EMBEDDED_CONSTANTS_MAX_BYTES_FLAG)
  # LoweringParameters evaluates its default hoist_constants_as_args=False once
  # at import time, so Traced.lower() ignores the runtime config flag unless the
  # dataclass __init__ defaults are also updated.
  init_fn = jax.interpreters.mlir.LoweringParameters.__init__
  previous_defaults = getattr(init_fn, "__defaults__")
  jax.config.update(_SIMPLIFIED_JAXPR_CONSTANTS_FLAG, True)
  jax.config.update(_EMBEDDED_CONSTANTS_MAX_BYTES_FLAG, 0)
  setattr(init_fn, "__defaults__", (*previous_defaults[:-1], True))
  try:
    yield
  finally:
    setattr(init_fn, "__defaults__", previous_defaults)
    jax.config.update(_EMBEDDED_CONSTANTS_MAX_BYTES_FLAG, previous_max_bytes)
    jax.config.update(_SIMPLIFIED_JAXPR_CONSTANTS_FLAG, previous)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PadToMultiple:
  """Pads each batch to a multiple of ``multiple`` to bound recompilations.

  If ``multiple`` is ``None`` (the default), batches are padded to the maximum
  batch size encountered across the run (rounded up to a multiple of
  ``microbatch_size`` if set), so ``train_step`` is compiled once.

  Note: Because the compiled batch size(s) depend on the sampled batches, the
  compilation cache (as well as step timing and memory usage) can leak
  information about the secret batch selection if exposed to an adversary.
  """

  multiple: int | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class AutotuneMicrobatch:
  """Auto-selects the largest microbatch that fits, then compiles once."""

  hbm_safety_fraction: float = 0.9


CompilationStrategy: TypeAlias = PadToMultiple | AutotuneMicrobatch


def batch_sharding(
    batch_axis_name: str | tuple[str, ...] | None = None,
) -> jax.sharding.NamedSharding | None:
  """Returns the batch NamedSharding for the active mesh, or None."""
  mesh = jax.sharding.get_mesh()
  if batch_axis_name is None or mesh.empty:
    return None
  spec = jax.sharding.PartitionSpec(batch_axis_name)
  return jax.sharding.NamedSharding(mesh, spec)


def _abstract_batch_and_padding(
    dataset, size, sharding=None, padding_example=None
):
  """Returns abstract ``(batch, is_padding)`` inputs of the given size."""
  if padding_example is not None:
    dataset, dim = padding_example, 0
  elif _data_loading.is_map_dataset(dataset):
    dataset, dim = dataset[0], 0
  else:
    dim = 1

  def _to_abstract(x):
    return jax.ShapeDtypeStruct(
        (size, *x.shape[dim:]), x.dtype, sharding=sharding
    )

  return jax.tree.map(_to_abstract, dataset), _to_abstract(np.bool_(False))


def _dry_run_state(
    dataset: training.Dataset,
    state: training.TrainingState,
    rng_or_seed: np.random.Generator | int | None,
    padding_example: Any = None,
) -> tuple[np.random.Generator, int, training.TrainingState, jax.Array]:
  """Eval-shape setup; draws the same rng as training for JIT cache hits."""
  rng = copy.deepcopy(np.random.default_rng(rng_or_seed))
  seed = rng.integers(2**63)
  if _data_loading.is_map_dataset(dataset):
    n = len(dataset)
  else:
    n = _validate.batch(dataset, padding_example)
  state = jax.tree.map(jax.ShapeDtypeStruct.like, state)
  key = jax.ShapeDtypeStruct.like(jax.random.key(seed))
  return rng, n, state, key


def _max_padded_batch_size(
    strategy: batch_selection.BatchSelectionStrategy,
    num_examples: int,
    rng: np.random.Generator | int | None = None,
    microbatch_size: int | None = None,
) -> int:
  """Returns the max batch size rounded up to a multiple of microbatch_size."""
  # Dry-run the batch iterator on a copy of rng so the caller's generator state
  # is preserved, then round the max size up to a multiple of microbatch_size.
  batches = strategy.batch_iterator(num_examples, rng=copy.deepcopy(rng))
  max_batch = max(idx.size for idx in batches)
  mb = microbatch_size or 1
  return max(mb, math.ceil(max_batch / mb) * mb)


def precompile(
    trainer: training.DPTrainer,
    dataset: training.Dataset,
    state: training.TrainingState,
    *,
    rng_or_seed: np.random.Generator | int | None = None,
) -> tuple[training.DPTrainer, dict[int, PrecompiledFuture]]:
  """[ADVANCED] Resolve config and AOT-compile the step(s) ``fit`` will run.

  Args:
    trainer: The trainer whose ``train_step`` will be compiled.
    dataset: The training dataset batches are sampled from.
    state: The initial ``TrainingState``.
    rng_or_seed: Optional RNG or seed controlling batch sampling; must match the
      value later passed to training so cached batches align.

  Returns:
    ``(trainer, futures)``: the (possibly resolved) trainer and a mapping from
    padded batch size to its background compilation future.
  """
  strategy = trainer.compilation_strategy
  if isinstance(strategy, AutotuneMicrobatch):
    return _autotune(
        trainer,
        dataset,
        state,
        strategy.hbm_safety_fraction,
        rng_or_seed=rng_or_seed,
    )

  # This rng matches the one used in training, so even though the batch iterator
  # is stochastic, the batch sizes and hence compiled functions will match.
  rng, n, state, key = _dry_run_state(
      dataset, state, rng_or_seed, trainer.padding_example
  )
  sharding = batch_sharding(trainer.performance_flags.batch_axis_name)
  batch_strategy = trainer.plan.batch_selection_strategy
  microbatch_size = trainer.performance_flags.microbatch_size
  if strategy.multiple is None:
    multiple = _max_padded_batch_size(batch_strategy, n, rng, microbatch_size)
    strategy = PadToMultiple(multiple=multiple)
    trainer = dataclasses.replace(trainer, compilation_strategy=strategy)
  futures: dict[int, PrecompiledFuture] = {}
  with hoist_closed_over_constants():
    for idx in batch_strategy.batch_iterator(n, rng=rng):
      padded = batch_selection.pad_to_multiple_of(
          idx,
          strategy.multiple,
          microbatch_size=microbatch_size,
      )
      batch_size = padded.size
      if batch_size in futures:
        continue
      batch, pad = _abstract_batch_and_padding(
          dataset, batch_size, sharding, trainer.padding_example
      )

      # The compilation cache leaks the compiled batch size(s), which depend on
      # the sampled batches -- a technical DP violation if it is part of output.
      lowered = trainer.train_step.lower(trainer, state, batch, pad, key)
      logging.info("AOT-compiling train_step for batch size %d", batch_size)
      # We asyncronously ahead-of-time (AOT) compile the lowered function in a
      # background thread to avoid blocking the training loop. Currently, the
      # rest of this function (batch simulation + lowering) happens on the main
      # thread. This could potentially be improved in the future.
      futures[batch_size] = _COMPILE_POOL.submit(lowered.compile)

  return trainer, futures


def _compile_and_peak(
    trainer: training.DPTrainer,
    state: training.TrainingState,
    batch: training.Batch,
    is_padding_example: jax.Array,
    prng_key: jax.Array,
) -> tuple[jax.stages.Compiled | None, float]:
  """Compiles the step; returns (compiled, peak_bytes) or (None, inf)."""
  try:
    # Hoist constants so the measured peak matches the (hoisted) training run.
    with hoist_closed_over_constants():
      compiled = trainer.train_step.lower(
          trainer, state, batch, is_padding_example, prng_key
      ).compile()
  except jax.errors.JaxRuntimeError:
    return None, float("inf")
  stats = compiled.memory_analysis()
  peak = stats.peak_memory_in_bytes if stats is not None else 0
  return compiled, peak


def _extrapolate_seed(peak1, peak2, budget, powers):
  """Largest power of two whose extrapolated peak fits the budget."""
  seed = powers[0]
  for b in powers:
    if peak1 + (peak2 - peak1) * (b - 1) <= budget:
      seed = b
  return seed


def _device_hbm_limit() -> int | None:
  """Sources the per-chip HBM budget from the first local device."""
  try:
    device = jax.local_devices()[0]
  except IndexError:
    return None
  try:
    limit = device.memory_stats().get("bytes_limit")
    if limit is not None:
      return limit
  except (RuntimeError, AttributeError):
    pass
  return getattr(device, "device_memory_bytes_limit", None)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Candidate:
  """Result of compile-verifying a single microbatch size."""

  trainer: training.DPTrainer
  pad: int
  compiled: jax.stages.Compiled | None
  peak: float
  fits: bool


def _autotune(
    trainer: training.DPTrainer,
    dataset: training.Dataset,
    state: training.TrainingState,
    hbm_safety_fraction: float,
    *,
    rng_or_seed: np.random.Generator | int | None = None,
) -> tuple[training.DPTrainer, dict[int, PrecompiledFuture]]:
  """Selects the largest microbatch that fits, then compiles it once."""
  # Largest microbatch is capped by the per-chip HBM budget (None if unknown,
  # in which case we fall back to compile-success as the fit test below).
  limit = _device_hbm_limit()
  logging.info("[JAX-Privacy] Found per-chip HBM limit: %s", limit)
  budget = limit * hbm_safety_fraction if limit is not None else None

  rng, n, state, key = _dry_run_state(
      dataset, state, rng_or_seed, trainer.padding_example
  )
  sharding = batch_sharding(trainer.performance_flags.batch_axis_name)
  batch_strategy = trainer.plan.batch_selection_strategy
  max_batch = max(idx.size for idx in batch_strategy.batch_iterator(n, rng=rng))
  powers = [2**i for i in range(max_batch.bit_length())]

  def _resolve(microbatch, pad_size):
    performance_flags = dataclasses.replace(
        trainer.performance_flags, microbatch_size=microbatch
    )
    return dataclasses.replace(
        trainer,
        performance_flags=performance_flags,
        compilation_strategy=PadToMultiple(multiple=pad_size),
    )

  @functools.cache
  def probe(microbatch: int) -> _Candidate:
    """Compile-verifies a microbatch size and measures its peak (memoized)."""
    pad = int(-(-max_batch // microbatch) * microbatch)
    resolved = _resolve(microbatch, pad)
    batch, padding = _abstract_batch_and_padding(
        dataset, pad, sharding, trainer.padding_example
    )
    compiled, peak = _compile_and_peak(resolved, state, batch, padding, key)
    fits = compiled is not None and (budget is None or peak <= budget)
    return _Candidate(
        trainer=resolved, pad=pad, compiled=compiled, peak=peak, fits=fits
    )

  # Seed from an affine peak(microbatch) fit at 1 and 2, then compile-verify and
  # step one power of two at a time to the largest fitting size (never OOMs).
  seed = powers[0]
  if len(powers) > 1 and budget is not None:
    seed = _extrapolate_seed(probe(1).peak, probe(2).peak, budget, powers)

  i = powers.index(seed)
  if probe(powers[i]).fits:
    while i + 1 < len(powers) and probe(powers[i + 1]).fits:
      i += 1
  else:
    while i > 0 and not probe(powers[i]).fits:
      i -= 1

  best = probe(powers[i])
  if not best.fits:
    logging.warning("No microbatch size fit device memory; using 1.")
    return _resolve(1, max_batch), {}

  logging.info("Autotune microbatch=%d fits; pad=%d.", powers[i], best.pad)
  future: PrecompiledFuture = concurrent.futures.Future()
  future.set_result(best.compiled)
  return best.trainer, {best.pad: future}

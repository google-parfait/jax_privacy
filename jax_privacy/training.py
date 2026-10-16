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

"""End-to-end training loop for differentially private training.

This module provides :class:`~jax_privacy.training.DPTrainer`, a class that
encapsulates the static configuration for a DP training loop (execution plan,
loss function, optimizer) and exposes a reusable ``train_step`` that can be
independently JIT-compiled or ahead-of-time compiled.

Design notes: This module aims to provide a framework-agnostic and model
agnostic DP training loop that can be easily configured to use a variety of
mechanism variants and performance knobs. Since it is defined in terms of an
ExecutionPlanConfig object, it automatically works with any mechanism plan
supported by our Tier 2 API.

Because it is designed to be lightweight and framework-agnostic, ``jax_privacy``
does not take dependencies on checkpointing libraries (such as Orbax) or metrics
logging backends, and leaves those concerns out of scope for this module.
Instead, :class:`~jax_privacy.training.DPTrainer` is designed to compose
naturally with external infrastructure: callers can inject custom checkpointing
and logging logic via the ``callback`` argument to
:meth:`~jax_privacy.training.DPTrainer.fit`, and resume from a saved checkpoint
by passing the restored :class:`~jax_privacy.training.TrainingState` directly to
:meth:`~jax_privacy.training.DPTrainer.fit`.

Example Usage:

  >>> def loss_fn(params, batch, prng):
  ...   return jax.numpy.mean((params - batch) ** 2), {}
  >>> cfg = BandMFConfig.default(1, 5, 5, noise_multiplier=0.0)
  >>> trainer = DPTrainer(config=cfg, loss_fn=loss_fn, optimizer=optax.sgd(0.5))
  >>> data, in_c, out_c = np.zeros((8, 2)), np.ones((4, 2)), -np.ones((4, 2))
  >>> params0 = np.zeros(2)
  >>> state0 = trainer.init(params0)
  >>> state = trainer.fit(np.vstack([data, in_c]), state0, rng_or_seed=0)
  >>> auditor = audit(loss_fn, state.params, in_c, out_c, params0)
  >>> float(auditor.attack_auroc())
  1.0
"""

from collections.abc import Callable
import dataclasses
import functools
from typing import Any, Protocol, TypeAlias

from absl import logging
import jax
import jax_privacy
from jax_privacy import _compilation
from jax_privacy import _data_loading
from jax_privacy import _validate
from jax_privacy import auditing
from jax_privacy import execution_plan
from jax_privacy import optimizers as aug_optimizers
import numpy as np
import optax
from optax import microbatching

# Re-export key symbols so users can access them via jax_privacy.training.
BandMFConfig = execution_plan.BandMFConfig
DPExecutionPlan = execution_plan.DPExecutionPlan
ExecutionPlanConfig = execution_plan.ExecutionPlanConfig
PerformanceFlags = execution_plan.PerformanceFlags

Loss: TypeAlias = jax.Array
Aux: TypeAlias = optax.ArrayTree
PerExampleAux: TypeAlias = jax_privacy.clipping.AuxiliaryOutput
Batch: TypeAlias = optax.ArrayTree
Dataset: TypeAlias = optax.ArrayTree | Any
Params: TypeAlias = optax.ArrayTree
OptState: TypeAlias = optax.ArrayTree
NoiseState: TypeAlias = optax.ArrayTree
# Re-exported so callers can keep using training.PrecompiledFuture.
PrecompiledFuture: TypeAlias = _compilation.PrecompiledFuture
# Compilation strategies, re-exported for the public API.
CompilationStrategy = _compilation.CompilationStrategy
PadToMultiple = _compilation.PadToMultiple
AutotuneMicrobatch = _compilation.AutotuneMicrobatch


class LossFn(Protocol):
  """Expected contract for loss functions used in DP training.

  Loss functions must accept ``params``, ``data``, and ``prng`` and return
  ``(loss, aux)``.

  Any additional context the loss function needs — frozen parameters,
  model configuration, label smoothing constants, etc. — should be closed
  over before passing the function to :class:`~jax_privacy.training.DPTrainer`::

      frozen = model.freeze(some_params)
      def my_loss(params, data, prng):
          all_params = {**frozen, **params}
          logits = model.apply(all_params, data['x'], rngs={'dropout': prng})
          return cross_entropy(logits, data['y']), {'logits': logits}

      trainer = DPTrainer(..., loss_fn=my_loss, ...)

  NOTE: This signature does not support mutable model state that persists
  across steps (e.g., batch-norm running statistics), as such state is
  generally incompatible with DP-SGD or very difficult to handle correctly.

  Example signature::

      def loss_fn(params, data, prng):
          ...
          return loss, aux
  """

  def __call__(
      self,
      params: Params,
      data: Batch,
      prng: jax.Array,
  ) -> tuple[Loss, Aux]:
    ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class TrainingState:
  """Container for the state of the training loop."""

  step: int
  params: Params
  opt_state: OptState
  noise_state: NoiseState | None = None


CallbackFn: TypeAlias = Callable[[int, TrainingState, PerExampleAux], None]


# DPTrainer contains static configuration that defines the training step, but
# not all fields are hashable (e.g., BandMFConfig contains an `array` field).
# To get jax.jit to work, we use `eq=False` below to hash based on object ID.
# TODO: Refactor to reduce unnecessary recompilations.
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class DPTrainer:
  """Stateless trainer encapsulating the static configuration of a DP loop.

  ``DPTrainer`` separates *configuration* (plan, loss, optimizer) from
  *per-run state* (data, initial params, RNG seed).  This makes the
  ``train_step`` method available as a standalone callable that can be
  compiled or used independently of the training loop.

  **Sharding**: For multi-device training inside a :func:`jax.set_mesh`
  context, provide ``params`` sharded on the active mesh and configure
  ``batch_axis_name`` through ``performance_flags``. When ``batch_axis_name``
  is set and a mesh is active, :meth:`fit` shards the batch dimension of each
  batch and padding mask along ``PartitionSpec(batch_axis_name)``. In
  multi-controller setups (``jax.process_count() > 1``), every process must
  pass the full ``dataset`` and an identical ``rng_or_seed`` to :meth:`fit`.

  Attributes:
    config: An :class:`~jax_privacy.execution_plan.ExecutionPlanConfig` (e.g.
      :class:`~jax_privacy.execution_plan.BandMFConfig`) specifying the DP
      mechanism.
    performance_flags: Performance-only flags (numerical precision, sharding,
      memory/compute trade-offs) that do not affect the privacy guarantee.
    loss_fn: The per-example loss function.  See
      :class:`~jax_privacy.training.LossFn`.
    optimizer: An
      :class:`~jax_privacy.optimizers.AugmentedGradientTransformation` or a
      plain :class:`optax.GradientTransformation`.
    compilation_strategy: Selects which ``train_step`` programs are compiled for
      training. :class:`~jax_privacy.training.PadToMultiple` (default) pads
      batches to a multiple to bound recompilations;
      :class:`~jax_privacy.training.AutotuneMicrobatch` picks the largest
      microbatch size that fits device memory and compiles once. See
      :class:`~jax_privacy.training.CompilationStrategy`.
    padding_example: Optional single-example PyTree specification (arrays or
      :class:`jax.ShapeDtypeStruct`) defining the structure, shapes, and dtypes
      of zero-padded examples. If omitted on a ``grain.MapDataset``, it is
      inferred from ``dataset[0]``, which requires a non-empty dataset with
      fixed element shapes.
    noise_seed: Optional seed for the random number generator used for noise
      addition. If ``None``, a fresh random seed is drawn; pass an explicit seed
      for multi-host coordination, stateless checkpoint replay
      (``allow_missing_noise_state=True``), or reproducible tests (and keep it
      secret in production deployments).
    allow_missing_noise_state: Whether to allow ``noise_state=None`` on resume.
  """

  config: execution_plan.ExecutionPlanConfig
  performance_flags: execution_plan.PerformanceFlags = dataclasses.field(
      default_factory=execution_plan.PerformanceFlags
  )
  loss_fn: LossFn
  optimizer: (
      aug_optimizers.AugmentedGradientTransformation
      | optax.GradientTransformation
  )
  compilation_strategy: _compilation.CompilationStrategy = dataclasses.field(
      default_factory=_compilation.PadToMultiple
  )
  padding_example: Any = None
  noise_seed: int | None = None
  allow_missing_noise_state: bool = False

  def __post_init__(self):
    # noise_addition_transform closes over noise_seed (or a random key if None).
    if self.allow_missing_noise_state and self.noise_seed is None:
      raise ValueError("allow_missing_noise_state=True requires noise_seed.")
    _ = self.plan  # Build untraced so cached PRNG key isn't a leaked tracer.

  @functools.cached_property
  def plan(self) -> execution_plan.DPExecutionPlan:
    """``DPExecutionPlan`` built from ``config`` and ``performance_flags``."""
    return self.config.make(self.performance_flags, noise_seed=self.noise_seed)

  @jax.jit(static_argnames=["self"])
  def init(self, params: Params) -> TrainingState:
    """Initialize a ``TrainingState`` at step 0."""
    optimizer = aug_optimizers.as_augmented_optimizer(self.optimizer)
    # jax_privacy will sometimes upcast gradients to a higher dtype for
    # numerical stability, regardless of the dtype of params.
    grads_like = optax.tree.cast(params, self.performance_flags.dtype)
    noise_state = self.plan.noise_addition_transform.init(grads_like)
    return TrainingState(
        step=jax.numpy.zeros((), dtype=jax.numpy.int32),
        params=params,
        opt_state=optimizer.init(grads_like),
        noise_state=noise_state,
    )

  @jax.jit(static_argnames=["self"], donate_argnames=["state"])
  def train_step(
      self,
      state: TrainingState,
      batch: Batch,
      is_padding_example: jax.Array,
      prng_key: jax.Array,
  ) -> tuple[TrainingState, PerExampleAux]:
    """Executes a single DP training step.

    This method is a pure function of its inputs and is safe to wrap with
    ``jax.jit``, ``jax.jit(...).lower()``, or any other JAX transformation.

    Args:
      state: Current ``TrainingState``.
      batch: A PyTree of arrays representing the current mini-batch.
      is_padding_example: A boolean array indicating which examples in ``batch``
        are padding (and should be ignored).
      prng_key: Base PRNG key; a step-specific key is derived via
        ``jax.random.fold_in(prng_key, state.step)``.

    Returns:
      A tuple ``(new_state, aux)`` where ``new_state`` is the updated
      ``TrainingState`` and ``aux`` is the per-example auxiliary output.
    """
    optimizer = aug_optimizers.as_augmented_optimizer(self.optimizer)
    pre_clip_fn = optimizer.pre_clipping_transform(state.opt_state)
    grad_fn = self.plan.clipped_grad(
        self.loss_fn,
        has_aux=True,
        return_values=True,
        return_grad_norms=True,
        pre_clipping_transform=pre_clip_fn,
        prng_argnum=2,
    )
    loss_prng = jax.random.fold_in(prng_key, state.step)
    clipped_grad_sum, aux = grad_fn(
        state.params, batch, loss_prng, is_padding_example=is_padding_example
    )
    dp_grad, new_noise_state = self.plan.noise_addition_transform.update(
        clipped_grad_sum, state.noise_state
    )
    updates, new_opt_state = optimizer.update(
        dp_grad, state.opt_state, state.params
    )
    new_params = optax.apply_updates(state.params, updates)

    new_state = TrainingState(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        noise_state=new_noise_state,
    )
    return new_state, aux

  def _precompile(
      self,
      dataset: Dataset,
      state: TrainingState,
      *,
      rng_or_seed: np.random.Generator | int | None = None,
  ) -> tuple["DPTrainer", dict[int, PrecompiledFuture]]:
    """[ADVANCED] Resolve config and AOT-compile the steps ``fit`` runs."""
    if self.allow_missing_noise_state and state.noise_state is None:
      init_fn = jax.jit(self.plan.noise_addition_transform.init)
      grads_like = optax.tree.cast(state.params, self.performance_flags.dtype)
      noise_state = init_fn.eval_shape(grads_like)
      state = dataclasses.replace(state, noise_state=noise_state)
    return _compilation.precompile(
        self, dataset, state, rng_or_seed=rng_or_seed
    )

  def fit(
      self,
      dataset: Dataset,
      state: TrainingState,
      *,
      callback: CallbackFn | None = None,
      rng_or_seed: np.random.Generator | int | None = None,
      precompile: bool = True,
  ) -> TrainingState:
    """Runs an end-to-end differentially private training loop.

    Args:
      dataset: The training dataset, either as a PyTree of arrays where the
        first axis of each leaf is the batch / example dimension, or as a
        ``grain.MapDataset`` (or ``grain.RandomAccessDataSource``) where each
        element is a PyTree of arrays for a single example.
      state: Initial or resumable ``TrainingState`` (e.g., from :meth:`init`).
        Training starts or continues from ``state.step``, and the batch
        selection iterator safely fast-forwards to maintain deterministic DP
        properties. If ``state.noise_state`` is ``None``, it is reconstructed by
        replaying the noise addition transform from step 0 (requires
        ``self.allow_missing_noise_state=True``). Note that ``state`` buffers
        are donated during training (:meth:`train_step` uses
        ``donate_argnames=["state"]``), so copy any arrays you need to retain
        after ``fit`` returns.
      callback: Called after each step as ``callback(step, state, aux)``.
        ``step`` is a Python int.
      rng_or_seed: Optional random seed or ``numpy.random.Generator``, used for
        sampling batches (impacting privacy) and initializing the loss PRNG key
        (potentially impacting utility). Does not influence the noise addition
        transform, which is configured via the
        :class:`~jax_privacy.execution_plan.DPExecutionPlan`.
      precompile: A boolean indicating whether to asynchronously precompile
        ``train_step`` for the batch sizes encountered, instead of just-in-time
        compiling on the fly, which can idle accelerators during training.
        Strategies that resolve before training (e.g. ``AutotuneMicrobatch``)
        run even when this is ``False``.

    Returns:
      Final ``TrainingState``.
    """
    if (n := getattr(self.config, "num_examples", None)) is not None:
      is_map = _data_loading.is_map_dataset(dataset)
      m = len(dataset) if is_map else _validate.batch(dataset)
      _validate.equal(n, num_examples=m)
    # Precompilation resolves the trainer and compiles the required train_steps.
    # It may return a *different* trainer (e.g. AutotuneMicrobatch picks a
    # concrete microbatch size), so the loop below runs on ``trainer`` rather
    # than ``self``.
    trainer: DPTrainer = self
    futures: dict[int, PrecompiledFuture] = {}

    if self.allow_missing_noise_state and state.noise_state is None:
      transform = self.plan.noise_addition_transform
      grads_like = optax.tree.cast(state.params, self.performance_flags.dtype)
      noise_state = jax_privacy.noise_addition.init_for_step(
          transform, grads_like, int(state.step)
      )
      state = dataclasses.replace(state, noise_state=noise_state)

    if (
        precompile
        or getattr(self.compilation_strategy, "multiple", None) is None
    ):
      trainer, futures = self._precompile(
          dataset, state, rng_or_seed=rng_or_seed
      )
    assert isinstance(trainer.compilation_strategy, PadToMultiple)
    assert trainer.compilation_strategy.multiple is not None
    warn_on_cache_miss = bool(futures)

    # We need tight alignment between how rng is used here and in precompile().
    rng = np.random.default_rng(rng_or_seed)
    prng_key = jax.random.key(int(rng.integers(2**63)))

    with _compilation.hoist_closed_over_constants():
      axis = trainer.performance_flags.batch_axis_name
      batches = _data_loading.batch_iterator(
          dataset,
          trainer.plan.batch_selection_strategy,
          padding_example=trainer.padding_example,
          rng=rng,
          pad_to_multiple_of=trainer.compilation_strategy.multiple,
          microbatch_size=trainer.performance_flags.microbatch_size,
          initial_step=int(state.step),
          sharding=_compilation.batch_sharding(axis),
      )

      step = int(state.step)
      for batch, is_padding_example in batches:
        batch_size = is_padding_example.size
        step_fn = trainer.train_step
        if batch_size in futures:
          step_fn = futures[batch_size].result()
        elif warn_on_cache_miss:
          logging.info("JIT-compiling train_step for batch size %d", batch_size)
          logging.warning("Cache Miss! Precompile is not working as intended.")

        state, aux = step_fn(state, batch, is_padding_example, prng_key)
        step += 1

        del batch, is_padding_example

        if callback is not None:
          callback(step, state, aux)

    return state


@jax.jit(static_argnames=["loss_fn", "microbatch_size"])
def _score_canaries(
    loss_fn: LossFn,
    trained: Params,
    canaries: Batch,
    untrained: Params | None = None,
    *,
    microbatch_size: int | None = None,
    prng_key: jax.Array,
) -> jax.Array:
  """Computes per-canary loss reductions ``loss(untrained) - loss(trained)``."""
  n = _validate.batch(canaries)
  mb = microbatch_size or 1
  idx = jax_privacy.batch_selection.pad_to_multiple_of(np.arange(n), mb)

  def score_one(x, key):
    b = jax.tree.map(lambda a: a[None], x)
    # Reuse key across both evaluations to cancel stochastic loss variance.
    l0 = 0.0 if untrained is None else loss_fn(untrained, b, key)[0]
    return l0 - loss_fn(trained, b, key)[0]

  vmap_fn = microbatching.micro_vmap(score_one, (0, 0), microbatch_size=mb)
  padded = jax.tree.map(lambda x: x[idx], canaries)
  keys = jax.random.split(prng_key, idx.size)
  return vmap_fn(padded, keys)[:n]


def audit(
    loss_fn: LossFn,
    trained: Params,
    in_canaries: Batch,
    out_canaries: Batch,
    untrained: Params | None = None,
    *,
    microbatch_size: int | None = None,
    prng_key: jax.Array | int = 0,
) -> auditing.CanaryScoreAuditor:
  """Scores held-in and held-out canaries and returns an auditor.

  Args:
    loss_fn: Per-example loss function matching :class:`LossFn`.
    trained: Trained model parameters.
    in_canaries: Held-in canary batch included in the training dataset.
    out_canaries: Held-out canary batch excluded from the training dataset.
    untrained: Optional untrained model parameters used to center canary scores
      as ``loss(untrained) - loss(trained)``. If ``None``, scores are
      ``-loss(trained)``.
    microbatch_size: Optional microbatch size for evaluating ``loss_fn``.
    prng_key: Random seed or PRNG key for loss_fn evaluation.

  Returns:
    A :class:`~jax_privacy.auditing.CanaryScoreAuditor` initialized with the
    held-in and held-out canary scores.
  """
  if isinstance(prng_key, int):
    prng_key = jax.random.key(prng_key)
  in_key, out_key = jax.random.split(prng_key)
  args = (loss_fn, trained)
  kw = dict(untrained=untrained, microbatch_size=microbatch_size)
  in_scores = _score_canaries(*args, in_canaries, prng_key=in_key, **kw)
  out_scores = _score_canaries(*args, out_canaries, prng_key=out_key, **kw)
  return auditing.CanaryScoreAuditor(in_scores, out_scores)

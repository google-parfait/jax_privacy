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

"""API for adding DP-SGD to a Keras model.

Example Usage:

.. code-block:: python

   import os
   os.environ["KERAS_BACKEND"] = "jax"
   import keras
   from jax_privacy import keras_api

   model = keras.Sequential([
       keras.Input(shape=(1,)),
       keras.layers.Dense(1),
   ])
   params = keras_api.DPKerasConfig(
       epsilon=1.0,
       delta=1e-5,
       clipping_norm=1.0,
       batch_size=8,
       gradient_accumulation_steps=1,
       train_steps=10,
       train_size=80,
       noise_multiplier=1.0,
   )
   private_model = keras_api.make_private(model, params)
   private_model.get_noise_multiplier()
"""

from collections.abc import Callable
import dataclasses
import functools
import inspect
import math
import types
from typing import Any

import dp_accounting
import jax
import jax.numpy as jnp
import keras
import numpy as np
import optax

from . import _validate
from . import accounting
from . import batch_selection
from . import clipping
from . import noise_addition


@dataclasses.dataclass(frozen=True)
class DPKerasConfig:
  """Parameters for adding DP-SGD to a Keras model.

  Attributes:
      epsilon: The epsilon that defines the differential-privacy budget. It
        should be in (0; +infinity) range. 0 means perfect privacy guarantee
        (not achievable in practice due to infinite noise), +infinity means no
        privacy guarantee. A commonly used value is ln(3) (smaller value more
        noise). You should set this value before training and only based on the
        privacy guarantees you have to achieve. You should not increase the
        epsilon only because of poor model performance.
      delta: The delta that defines the differential-privacy budget. The value
        of it means the probability of full disclosure, no-privacy. It should be
        in (0, 1] and be as small as possible (e.g. 1e-5, smaller value more
        noise). You should set this value before training and only based on the
        privacy guarantees you have to achieve. You should not increase the
        delta only because of poor model performance.
      clipping_norm: The clipping norm for the per-example gradients.
      batch_size: The batch size used by the DP optimizer. When
        ``poisson_sampling_in_fit=True``, this is the expected batch size of the
        internal Poisson sampler. Otherwise it must match the batch size
        supplied to ``fit()``.
      gradient_accumulation_steps: The number of gradient accumulation steps.
        This is the number of batches to accumulate before adding noise and
        performing an optimizer step. 1 means that there is no gradient
        accumulation, each optimizer step is performed after a single batch.
        This parameter defines the effective batch size = (physical) batch_size
        * gradient_accumulation_steps, i.e. the real accumulated batch size used
        for the model update. Usually DP training provides better accuracy with
        larger effective batch size, therefore it is recommended to set
        gradient_accumulation_steps to a value larger than 1. In many cases, you
        won't be able to set the physical batch size to a large enough value due
        to memory constraints, therefore gradient accumulation technique is very
        useful during DP training.
      train_steps: The number of training steps (optimizer update steps). If you
        try to train the model for more steps, it will fail. Because one
        optimizer update consumes ``gradient_accumulation_steps`` batches, this
        is the number of optimizer updates, not the number of (micro-)batches.
        If you train by epochs, then it is epochs * (train_size //
        effective_batch_size), where effective_batch_size = batch_size *
        gradient_accumulation_steps (this reduces to epochs * (train_size //
        batch_size) only when gradient_accumulation_steps == 1). If you train
        while the dataset iterator is not over, then it is the number of
        optimizer updates the iterator produces, i.e. its (micro-)batch length
        // gradient_accumulation_steps.
      train_size: The number of training examples in the dataset. If you repeat
        the examples in your dataset iterator, it should be the number of
        training examples in the original dataset before repeating.
      poisson_sampling_in_fit: Whether ``fit()`` should internally resample
        random-access array inputs using Poisson sampling. Leave this as
        ``False`` for backwards-compatible behavior or when the user supplies a
        dataset iterator that already handles sampling.
      noise_multiplier: The noise multiplier for the gradients. If ``None``
        (recommended), the noise multiplier will be automatically calculated
        based on epsilon, delta, effective_batch_size, train_steps and
        train_size. The noise added to the average of gradients per total batch
        is normal with mean 0 and stddev = noise_multiplier * clipping_norm /
        effective_batch_size.
      rescale_to_unit_norm: Whether to rescale the gradients to unit norm.
        Simplifies learning-rate tuning, see https://arxiv.org/abs/2204.13650.
      seed: The seed for the random number generator. If ``None``, a random seed
        is used. It must be an int64. Useful for reproducibility.
      microbatch_size: The size of each microbatch. The device batch size will
        be split up into microbatches of this size and processed sequentially on
        the forward/backward pass. By setting ``microbatch_size=batch_size``,
        the forward/backward pass is performed once on the entire batch using
        :func:`jax.vmap`. By setting ``microbatch_size=1``, the forward/backward
        pass is performed on each batch element individually, with the gradients
        accumulated sequentially using :func:`jax.lax.scan`. Setting to
        ``batch_size`` gives the largest degree of parallelism, while setting to
        1 gives the least memory consumption. Any value in between can be used
        to trade-off memory consumption vs. parallel computation. This parameter
        is similar to ``gradient_accumulation_steps``, but it works fully inside
        of device memory under a single jitted function, while
        ``gradient_accumulation_steps`` operates outside of the jit boundary.
        The default value is ``None``, which means that no microbatching is
        used, and is equivalent to ``microbatch_size=batch_size``.
  """

  epsilon: float
  delta: float
  clipping_norm: float
  batch_size: int
  gradient_accumulation_steps: int
  train_steps: int
  train_size: int
  poisson_sampling_in_fit: bool = False
  noise_multiplier: float | None = None
  rescale_to_unit_norm: bool = True
  microbatch_size: int | None = None
  seed: int | None = None

  # Smaller values result in higher precision but slower computation.
  _pld_discretization = 1e-3

  @property
  def effective_batch_size(self) -> int:
    """The effective batch size which is used for the model update.

    It equals ``batch_size * gradient_accumulation_steps``.
    """
    return self.batch_size * self.gradient_accumulation_steps

  def update_with_calibrated_noise_multiplier(self) -> 'DPKerasConfig':
    """Calculates the noise multiplier for the given DP training parameters.

    Returns:
      A copy (new instance) of DPKerasConfig with the noise multiplier set to
      the calibrated value.
    """
    print(
        f'Calculating noise multiplier for: {self.epsilon=},'
        f' {self.delta=}, {self.effective_batch_size=}, {self.train_steps=},'
        f' {self.train_size=}. This might take a few minutes.'
    )
    sampling_prob = self.effective_batch_size / self.train_size
    make_event = lambda nm: accounting.dpsgd_event(
        noise_multiplier=nm,
        iterations=self.train_steps,
        sampling_prob=sampling_prob,
    )
    calculated_noise_multiplier = dp_accounting.calibrate_dp_mechanism(
        make_fresh_accountant=functools.partial(
            dp_accounting.pld.PLDAccountant,
            value_discretization_interval=self._pld_discretization,
        ),
        make_event_from_param=make_event,
        target_epsilon=self.epsilon,
        target_delta=self.delta,
    )
    print(
        'Finished calculating noise multiplier:'
        f' {calculated_noise_multiplier=}.'
    )
    return dataclasses.replace(
        self, noise_multiplier=calculated_noise_multiplier
    )

  def __post_init__(self):
    self._validate_params()

  def _validate_params(self) -> None:
    """Validates the parameters for DP-SGD training."""
    _validate.positive(
        epsilon=self.epsilon,
        clipping_norm=self.clipping_norm,
        batch_size=self.batch_size,
        train_size=self.train_size,
    )
    if not 0 < self.delta <= 1:
      raise ValueError(f'Expected delta={self.delta} in (0, 1].')
    if self.batch_size > self.train_size:
      raise ValueError(
          f'Batch size {self.batch_size} must be less than or equal to train'
          f' size {self.train_size}.'
      )
    _validate.positive(
        train_steps=self.train_steps,
        gradient_accumulation_steps=self.gradient_accumulation_steps,
    )
    if self.microbatch_size is not None:
      _validate.positive(microbatch_size=self.microbatch_size)
      if self.microbatch_size > self.batch_size:
        raise ValueError(
            f'Microbatch size {self.microbatch_size} must be less than or'
            f' equal to batch size {self.batch_size}.'
        )
    if self.noise_multiplier is not None:
      _validate.positive(noise_multiplier=self.noise_multiplier)
      try:
        sampling_prob = self.effective_batch_size / self.train_size
        event = accounting.dpsgd_event(
            noise_multiplier=self.noise_multiplier,
            iterations=self.train_steps,
            sampling_prob=sampling_prob,
        )
        pld_accountant = dp_accounting.pld.PLDAccountant(
            value_discretization_interval=self._pld_discretization,
        )
        resulting_epsilon = pld_accountant.compose(event).get_epsilon(
            target_delta=self.delta,
        )
      except ValueError as e:
        raise ValueError(
            'Value error occurred while calculating epsilon based on the'
            f' provided {self.noise_multiplier=}. Maybe the noise multiplier is'
            f' too small? Original error: {e}'
        ) from e
      tolerance = 1e-1
      if resulting_epsilon > self.epsilon + tolerance:
        raise ValueError(
            f'Provided {self.noise_multiplier=} will lead to privacy'
            ' budget exceed because the resulting epsilon will be'
            f' {resulting_epsilon=} > target_epsilon={self.epsilon}. You need'
            ' to set a greater noise multiplier (greater epsilon means more'
            ' noise and more budget). Or you can leave noise multiplier unset'
            ' at all and let the API to automatically calculate the optimal'
            ' one.'
        )


def make_private(model: keras.Model, params: DPKerasConfig) -> keras.Model:
  """Adds DP-SGD training to a Keras model without modifying its API.

  This function mutates ``model`` in place, installs the DP-SGD hooks, and
  returns the same model instance. When
  ``params.poisson_sampling_in_fit`` is enabled, the wrapped ``fit()`` path
  expects random-access per-example arrays or pytrees of arrays so it can
  perform Poisson sampling internally.

  Args:
    model: The Keras model to add DP-SGD training to.
    params: The :class:`~jax_privacy.keras_api.DPKerasConfig` parameters for
      DP-SGD training.

  Returns:
    The input model with overloaded methods for DP-SGD training.
  """
  _validate_model(model)

  # Adding DP-SGD to the model works in the following way:
  # 1. We add attributes to the model:
  #    - _dp_params: The parameters for DP-SGD training.
  #    - _gradient_computer: The gradient computer for DP-SGD training.
  #    - _rng: The random number generator for DP-SGD training.
  # 2. We replace the model.fit with a new method that performs validation for
  #    DP-SGD training and calls the original fit method.
  # 3. We replace the model.train_step method with a new method that performs
  #    DP-SGD training. This method differs from the original, only in the
  #    gradient computation (clipped and noised).
  # 4. We replace the model._update_metrics_variables method with a new method
  #    that updates the metrics variables for DP-SGD training.

  _add_dp_sgd_attributes(model, params)
  model.get_noise_multiplier = types.MethodType(get_noise_multiplier, model)
  model.fit = types.MethodType(
      _create_fit_fn_with_validation(model.fit, params), model
  )
  model.train_step = types.MethodType(_dp_train_step, model)
  if not hasattr(model, '_update_metrics_variables'):
    # _update_metrics_variables was extracted from train_step recently in
    # https://github.com/keras-team/keras/pull/20805/. Since in our train_step
    # we use it, we need to add it if it's not present. In the future, when
    # we stop supporting old versions of Keras, we can remove this.
    model._update_metrics_variables = types.MethodType(  # pylint: disable=protected-access
        _update_metrics_variables, model
    )
  return model


def get_noise_multiplier(model: keras.Model) -> float:
  """Returns the noise multiplier used for DP-SGD training.

  If the noise multiplier is not set in
  :class:`~jax_privacy.keras_api.DPKerasConfig`, this will calibrate it once and
  cache the value on the model.

  Args:
    model: A Keras model previously wrapped with
      :func:`~jax_privacy.keras_api.make_private`.

  Returns:
    The configured or calibrated noise multiplier.
  """
  if not hasattr(model, '_dp_params'):
    raise ValueError(
        'Model does not appear to be a DP-SGD Keras model. '
        'Call make_private() first.'
    )
  return _resolve_noise_multiplier(
      model._dp_params, model  # pylint: disable=protected-access
  )


def _validate_model(model: keras.Model) -> None:
  if not isinstance(model, keras.Model):
    raise ValueError(f'Model {model} is not a Keras model.')
  if keras.config.backend() != 'jax':
    raise ValueError(f'Model {model} must use Jax backend.')
  # TODO: b/415360727 - Add validation that the model does not contain layers
  # that are not compatible with DP-SGD, e.g. batch norm.


def _validate_optimizer(model: keras.Model, params: DPKerasConfig) -> None:
  optimizer_gradient_accumulation_steps = (
      model.optimizer.gradient_accumulation_steps or 1
  )
  dp_params_gradient_accumulation_steps = params.gradient_accumulation_steps
  if (
      optimizer_gradient_accumulation_steps
      != dp_params_gradient_accumulation_steps
  ):
    raise ValueError(
        'optimizer.gradient_accumulation_steps ='
        f' {optimizer_gradient_accumulation_steps} must be equal to'
        ' DPKerasConfig.gradient_accumulation_steps ='
        f' {dp_params_gradient_accumulation_steps}.'
    )


def _add_dp_sgd_attributes(model: keras.Model, params: DPKerasConfig) -> None:
  """Adds DP-SGD training attributes to the Keras model."""
  model._dp_params = params  # pylint: disable=protected-access
  model._dp_noise_multiplier = params.noise_multiplier  # pylint: disable=protected-access
  seed = _get_random_int64() if params.seed is None else params.seed
  model.add_weight(
      name='_rng',
      shape=(2,),
      dtype='uint32',
      initializer=lambda shape, dtype: jax.random.PRNGKey(seed),
      trainable=False,
  )
  model.add_weight(
      name='_optimizer_steps',
      shape=(1,),
      initializer=jnp.zeros,
      dtype='uint32',
      trainable=False,
  )


_FitFnReturnType = keras.callbacks.History
_DEFAULT_POISSON_PADDING_MULTIPLE = 32


class _PoissonSampledTrainingDataset(keras.utils.PyDataset):
  """Keras dataset serving Poisson-sampled batches from random-access data."""

  def __init__(
      self,
      x: optax.ArrayTree,
      y: optax.ArrayTree | None,
      sample_weight: optax.ArrayTree | None,
      *,
      dp_params: DPKerasConfig,
      steps_per_epoch: int,
  ):
    super().__init__()
    self._x = x
    self._y = y
    self._sample_weight = sample_weight
    self._train_size = _tree_batch_size(x)
    self._steps_per_epoch = steps_per_epoch
    self._gradient_accumulation_steps = dp_params.gradient_accumulation_steps
    self._sampling_prob = dp_params.effective_batch_size / float(
        self._train_size
    )
    self._padding_multiple = _get_poisson_padding_multiple(dp_params)
    seed = _get_random_int64() if dp_params.seed is None else dp_params.seed
    self._rng = np.random.default_rng(seed)
    self._epoch_batches = []
    self.on_epoch_end()

  def __len__(self) -> int:
    return self._steps_per_epoch

  def __getitem__(
      self, index: int
  ) -> tuple[optax.ArrayTree, optax.ArrayTree | None, optax.ArrayTree]:
    padded_indices = self._epoch_batches[index]
    batched_x = _take_batch_from_tree(self._x, padded_indices)
    batched_y = _take_batch_from_tree(self._y, padded_indices)
    batched_sample_weight = _build_batch_sample_weight(
        self._sample_weight, padded_indices
    )
    return batched_x, batched_y, batched_sample_weight

  def on_epoch_end(self) -> None:
    """Precomputes one epoch of padded Poisson-sampled index batches."""
    k = self._gradient_accumulation_steps
    num_macro_steps = math.ceil(self._steps_per_epoch / k)
    strategy = batch_selection.CyclicPoissonSampling(
        sampling_prob=self._sampling_prob,
        iterations=num_macro_steps,
    )
    epoch_batches = []
    for indices in strategy.batch_iterator(self._train_size, rng=self._rng):
      for slice_indices in np.array_split(np.asarray(indices), k):
        epoch_batches.append(
            _pad_batch_indices(slice_indices, self._padding_multiple)
        )
    self._epoch_batches = epoch_batches[: self._steps_per_epoch]


def _is_var_keyword_parameter(parameter: inspect.Parameter) -> bool:
  """Returns whether a signature entry corresponds to ``**kwargs``."""
  return parameter.kind is inspect.Parameter.VAR_KEYWORD


def _normalize_bound_fit_arguments(
    fit_signature: inspect.Signature,
    *args,
    **kwargs,
) -> dict[str, Any]:
  """Normalizes fit arguments into a kwargs-only call."""
  bound_arguments = fit_signature.bind_partial(*args, **kwargs)
  normalized_kwargs = {}
  for name, value in bound_arguments.arguments.items():
    parameter = fit_signature.parameters[name]
    if _is_var_keyword_parameter(parameter):
      # Flatten any bound **kwargs entry exposed by the wrapped Keras model.
      normalized_kwargs.update(value)
    else:
      normalized_kwargs[name] = value
  return normalized_kwargs


def _prepare_fit_kwargs_for_poisson_dataset(
    fit_kwargs: dict[str, Any],
    *,
    poisson_dataset: _PoissonSampledTrainingDataset,
) -> dict[str, Any]:
  """Swaps array inputs for a PyDataset and removes consumed fit arguments."""
  fit_kwargs = dict(fit_kwargs)
  fit_kwargs['x'] = poisson_dataset
  # Once x becomes a PyDataset, Keras expects targets and sample weights to be
  # yielded by the dataset rather than passed as separate fit arguments.
  for key in (
      'y',
      'sample_weight',
      'batch_size',
      'shuffle',
      'validation_split',
  ):
    fit_kwargs.pop(key, None)
  if fit_kwargs.get('steps_per_epoch') is None:
    fit_kwargs.pop('steps_per_epoch', None)
  return fit_kwargs


def _get_poisson_padding_multiple(dp_params: DPKerasConfig) -> int:
  if dp_params.microbatch_size is not None:
    return dp_params.microbatch_size
  return max(1, min(dp_params.batch_size, _DEFAULT_POISSON_PADDING_MULTIPLE))


def _pad_batch_indices(indices: np.ndarray, multiple: int) -> np.ndarray:
  """Pads indices with -1 so empty Poisson draws are still representable."""
  if indices.size == 0:
    return np.full(multiple, -1, dtype=np.int32)
  return batch_selection.pad_to_multiple_of(indices, multiple)


def _tree_batch_size(tree: optax.ArrayTree) -> int:
  """Returns and validates the batch size of a pytree of arrays."""
  leaves = jax.tree.leaves(tree)
  # Expected input: a non-empty pytree of random-access arrays whose leaves all
  # share the same leading batch dimension.
  if not leaves:
    raise ValueError('Expected at least one array leaf in the training data.')
  batch_size = None
  for leaf in leaves:
    if not hasattr(leaf, 'shape'):
      raise ValueError(
          'DP Keras training requires random-access array-like inputs.'
      )
    if not leaf.shape:
      raise ValueError(
          'DP Keras training requires each input leaf to have a batch'
          ' dimension.'
      )
    try:
      np.asarray(leaf[:1])
    except Exception as exc:  # pylint: disable=broad-exception-caught
      raise ValueError(
          'DP Keras training requires random-access array-like inputs.'
      ) from exc
    leaf_batch_size = leaf.shape[0]
    if batch_size is None:
      batch_size = leaf_batch_size
    elif leaf_batch_size != batch_size:
      raise ValueError(
          'All training data leaves must agree on their leading batch'
          ' dimension.'
      )
  return int(batch_size)


def _take_batch_from_leaf(
    leaf: jax.typing.ArrayLike, indices: np.ndarray
) -> np.ndarray:
  """Slices one batched leaf and zero-fills padded ``-1`` index positions."""
  leaf = np.asarray(leaf)
  valid_positions = indices >= 0
  batch = np.zeros((indices.shape[0],) + leaf.shape[1:], dtype=leaf.dtype)
  if np.any(valid_positions):
    batch[valid_positions] = leaf[indices[valid_positions]]
  return batch


def _take_batch_from_tree(
    tree: optax.ArrayTree | None, indices: np.ndarray
) -> optax.ArrayTree | None:
  if tree is None:
    return None
  return jax.tree.map(lambda leaf: _take_batch_from_leaf(leaf, indices), tree)


def _build_batch_sample_weight(
    sample_weight: optax.ArrayTree | None,
    indices: np.ndarray,
) -> optax.ArrayTree:
  """Builds sample weights that hide synthetic padding examples from Keras."""
  if sample_weight is None:
    return (indices >= 0).astype(np.float32)
  return _take_batch_from_tree(sample_weight, indices)


def _padding_mask_from_sample_weight(
    sample_weight: jax.Array,
) -> jax.Array:
  """Returns which batch entries are synthetic padding examples."""
  sample_weight = jnp.asarray(sample_weight)
  if sample_weight.ndim == 1:
    return sample_weight == 0
  if sample_weight.ndim == 2:
    return ~jnp.any(sample_weight, axis=1)
  raise ValueError('Expected sample_weight to be a 1D or 2D array.')


def _maybe_symbolically_build_private_model(
    model: keras.Model, dataset: _PoissonSampledTrainingDataset
) -> None:
  """Runs Keras' symbolic build before fit() sees the Poisson dataset."""
  if not hasattr(model, '_symbolic_build'):
    return
  # _symbolic_build is a private Keras helper, not a JAX-specific concept.
  # It lets Keras infer shapes and create state from a standard
  # (x, y, sample_weight) batch before the wrapped fit() path starts yielding
  # Poisson-sampled batches from the PyDataset. Without this eager build step,
  # lazily-built models can reach the wrapped fit() path before Keras has
  # created the model and optimizer state from a standard batch.
  x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(dataset[0])
  model._symbolic_build(data_batch=(x, y, sample_weight))  # pylint: disable=protected-access


def _masked_mean(
    values: jax.typing.ArrayLike, is_padding_example: jax.Array
) -> jax.Array:
  """Averages only the non-padding examples, returning 0 for empty batches."""
  values = jnp.asarray(values)
  if values.ndim == 0:
    return values
  where = jnp.asarray(~is_padding_example)
  for _ in range(values.ndim - 1):
    where = jnp.expand_dims(where, axis=-1)
  mean = jnp.mean(values, axis=0, where=where)
  return jnp.where(jnp.any(where, axis=0), mean, jnp.nan_to_num(mean))


def _create_fit_fn_with_validation(
    original_fit_fn: Callable[..., _FitFnReturnType],
    params: DPKerasConfig,
) -> Callable[..., _FitFnReturnType]:
  """Creates a fit function with validation for DP-SGD training.

   It validates that:
   1. DP parameters are aligned with the fit() arguments (e.g. batch sizes are
   equal).
   2. The number of optimizer steps that will be performed during fitting will
   not exceed the maximum number of optimizer steps.

  Args:
    original_fit_fn: The original fit function of the Keras model.
    params: The parameters for DP-SGD training.

  Returns:
    The fit function with same signature as original_fit_fn but with validation
    for DP-SGD training.
  """

  @functools.wraps(original_fit_fn)
  def fit_fn_with_validation(
      self,
      *args,
      **kwargs,
  ) -> _FitFnReturnType:
    _validate_optimizer(self, self._dp_params)  # pylint: disable=protected-access
    fit_signature = inspect.signature(original_fit_fn)
    fit_kwargs = _normalize_bound_fit_arguments(fit_signature, *args, **kwargs)
    use_poisson_sampling_in_fit = (
        self._dp_params.poisson_sampling_in_fit  # pylint: disable=protected-access
    )

    # batch_size is not set explicitely in the fit() call if the input dataset
    # is already batched. In this case, we assume that the batch sizes are
    # aligned and use the batch size from the DP parameters. We will check that
    # the batch sizes are aligned in the train_step function.
    batch_size = (
        _get_param(fit_signature, 'batch_size', *args, **kwargs)
        or params.batch_size
    )
    # Default values are set according to the Keras documentation.
    epochs = _get_param(fit_signature, 'epochs', *args, **kwargs) or 1
    initial_epoch = (
        _get_param(fit_signature, 'initial_epoch', *args, **kwargs) or 0
    )
    steps_per_epoch = _get_param(
        fit_signature, 'steps_per_epoch', *args, **kwargs
    )
    validation_split = (
        _get_param(fit_signature, 'validation_split', *args, **kwargs) or 0.0
    )
    x = _get_param(fit_signature, 'x', *args, **kwargs)
    y = _get_param(fit_signature, 'y', *args, **kwargs)
    sample_weight = _get_param(fit_signature, 'sample_weight', *args, **kwargs)
    validated_train_size = None
    if use_poisson_sampling_in_fit:
      if x is None:
        raise ValueError(
            'fit() must receive x when'
            ' DPKerasConfig.poisson_sampling_in_fit is enabled.'
        )
      if validation_split:
        raise ValueError(
            'validation_split is not supported for DP Keras training because'
            ' the privacy accountant needs the exact training-set size after'
            ' any split. Please create the train/validation split explicitly'
            ' and pass validation_data instead.'
        )
      validated_train_size = _tree_batch_size(x)
      if y is not None and _tree_batch_size(y) != validated_train_size:
        raise ValueError(
            'The target data must have the same leading batch dimension as'
            ' the training inputs.'
        )
      if (
          sample_weight is not None
          and _tree_batch_size(sample_weight) != validated_train_size
      ):
        raise ValueError(
            'The sample weights must have the same leading batch dimension as'
            ' the training inputs.'
        )

    # Note accessing self._dp_params is safe because it's added in
    # _add_dp_sgd_attributes, but requires disabling pylint because this
    # function is not a method within a class.
    _check_dp_params_aligned_with_fit_args(
        self._dp_params,  # pylint: disable=protected-access
        batch_size,
        train_size=validated_train_size,
    )

    performed_optimizer_steps = (
        _get_non_trainable_weight('_optimizer_steps', self).numpy().item()
    )
    optimizer_steps_to_perform = _calculate_optimizer_steps_to_perform_in_fit(
        self._dp_params.train_size,  # pylint: disable=protected-access
        batch_size,
        epochs,
        initial_epoch,
        steps_per_epoch,
        gradient_accumulation_steps=self._dp_params.gradient_accumulation_steps,  # pylint: disable=protected-access
    )
    if (
        performed_optimizer_steps + optimizer_steps_to_perform
        > self._dp_params.train_steps  # pylint: disable=protected-access
    ):
      raise RuntimeError(
          'fit() cannot be performed because you will run out of privacy'
          ' budget. Currently, you have already performed'
          f' {performed_optimizer_steps} optimizer training steps and you are'
          f' trying to perform {optimizer_steps_to_perform} more. However, you'
          f' can perform in total only {self._dp_params.train_steps} training'  # pylint: disable=protected-access
          ' steps (optimizer updates). If you fit() the model with current'
          ' parameters, training steps will exceed the maximum number of'
          f' training steps: {performed_optimizer_steps=} +'
          f' {optimizer_steps_to_perform=} ='
          f' {performed_optimizer_steps + optimizer_steps_to_perform} >'
          f' total_train_steps={self._dp_params.train_steps}.'  # pylint: disable=protected-access
      )
    if use_poisson_sampling_in_fit:
      default_steps_per_epoch = (
          _get_default_steps_per_epoch(
              validated_train_size,
              self._dp_params.effective_batch_size,  # pylint: disable=protected-access
          )
          * self._dp_params.gradient_accumulation_steps  # pylint: disable=protected-access
      )
      poisson_dataset = _PoissonSampledTrainingDataset(
          x,
          y,
          sample_weight,
          dp_params=self._dp_params,  # pylint: disable=protected-access
          steps_per_epoch=steps_per_epoch or default_steps_per_epoch,
      )
      _maybe_symbolically_build_private_model(self, poisson_dataset)
      fit_kwargs = _prepare_fit_kwargs_for_poisson_dataset(
          fit_kwargs,
          poisson_dataset=poisson_dataset,
      )
    return original_fit_fn(**fit_kwargs)

  return fit_fn_with_validation


def _check_dp_params_aligned_with_fit_args(
    dp_params: DPKerasConfig,
    batch_size: int,
    train_size: int | None = None,
) -> None:
  """Checks that the DP parameters are aligned with the fit() arguments."""
  if dp_params.batch_size != batch_size:
    raise ValueError(
        'The batch size in the DP parameters is not equal to the batch size'
        f' passed to fit(): {dp_params.batch_size=} != {batch_size=}. Please'
        ' make sure that the batch size in the DP parameters is equal to the'
        ' batch size passed to fit().'
    )
  if train_size is not None and dp_params.train_size != train_size:
    raise ValueError(
        'The train size in the DP parameters is not equal to the size of the'
        f' training data passed to fit(): {dp_params.train_size=} !='
        f' {train_size=}. Please make sure that DPKerasConfig.train_size'
        ' matches the number of training examples available for Poisson'
        ' sampling.'
    )


_XType = optax.ArrayTree
_YType = optax.ArrayTree
_SampleWeightType = optax.ArrayTree
_TrainableVariablesType = optax.ArrayTree
_NonTrainableVariablesType = list[jax.typing.ArrayLike]
_OptimizerVariablesType = list[jax.typing.ArrayLike]
_MetricsVariablesType = jax.typing.ArrayLike
_UnscaledLossType = jax.typing.ArrayLike
_YPredType = _YType
_KerasInputsDataType = tuple[_XType, _YType | None, _SampleWeightType | None]

_StateType = tuple[
    _TrainableVariablesType,
    _NonTrainableVariablesType,
    _OptimizerVariablesType,
    _MetricsVariablesType,
]

_AuxType = tuple[
    _UnscaledLossType,
    _YPredType,
    _NonTrainableVariablesType,
    _MetricsVariablesType,
]

_LogsType = dict[str, jax.typing.ArrayLike]


def _dp_train_step(
    self: keras.Model,
    state: _StateType,
    data: _KerasInputsDataType,
) -> tuple[_LogsType, _StateType]:
  """Performs a single training step.

  This function replaces Keras model train_step (that's why it has self arg).
  It differs from model.train_step only in the gradient computation (clipped
  and noised).

  Args:
    self: The Keras model.
    state: The state of the model (trainable, non-trainable, optimizer, metrics
      variables). As in model.train_step.
    data: The data for the model (x, y, sample_weight). As in model.train_step.
      Note that y and sample_weight can be None.

  Returns:
    logs: The logs for the training step, dict of metrics. As in
    model.train_step.
    state: The new state of the model. As in model.train_step.
  """
  (
      trainable_variables,
      _,
      optimizer_variables,
      _,
  ) = state
  x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
  is_padding_example = None
  if (
      self._dp_params.poisson_sampling_in_fit  # pylint: disable=protected-access
      and sample_weight is not None
  ):
    is_padding_example = _padding_mask_from_sample_weight(sample_weight)

  dp_batch_size = self._dp_params.batch_size  # pylint: disable=protected-access
  actual_batch_size = jax.tree_util.tree_leaves(x)[0].shape[0]
  if is_padding_example is None and dp_batch_size != actual_batch_size:
    # it is ok to throw an exception even though we are in a jit function
    # because the check is based on the static values, i.e. they won't
    # change between invocations, and if the condition is violated, it will
    # always fail during the tracing (first invocation) of this function.
    error_message = (
        'The batch size in the DP parameters is not equal to the batch size of'
        f' the actual data: {dp_batch_size=} !='
        f' actual_batch_size={actual_batch_size}. Please make sure that the'
        ' batch size in the DP parameters is equal to the batch size of the'
        ' data you supplied in the fit() call.'
    )
    raise ValueError(error_message)
  if is_padding_example is None:
    is_padding_example = jnp.zeros(actual_batch_size, dtype=jnp.bool_)

  (_, aux), grads = _noised_clipped_grads(
      self.compute_loss_and_updates,
      self._dp_params,  # pylint: disable=protected-access
      state,
      data,
      model=self,
      is_padding_example=is_padding_example,
  )
  (
      unscaled_loss,
      y_pred,
      non_trainable_variables,
      metrics_variables,
  ) = aux

  (
      trainable_variables,
      optimizer_variables,
  ) = self.optimizer.stateless_apply(
      optimizer_variables, grads, trainable_variables
  )
  # TODO: b/415360727 - access it and update it by name.
  non_trainable_variables[1] = non_trainable_variables[1] + 1

  logs, metrics_variables = self._update_metrics_variables(  # pylint: disable=protected-access
      metrics_variables, unscaled_loss, x, y, y_pred, sample_weight
  )

  if hasattr(self, '_enforce_jax_state_sharding'):
    # Sharding was moved out from train_step on
    # https://github.com/keras-team/keras/commit/0387d3057ac455d22f3e8d512f114115dfd7d12a
    # Which seems to be in Keras 3.12 (not released yet).
    # We can remove this if-clause once we stop supporting Keras <3.12.
    state = self._enforce_jax_state_sharding(  # pylint: disable=protected-access
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
        metrics_variables,
    )
  else:
    state = (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
        metrics_variables,
    )
  return logs, state


LossFn = Callable[..., tuple[jax.typing.ArrayLike, _AuxType]]


def _resolve_noise_multiplier(
    dp_params: DPKerasConfig, model: keras.Model | None = None
) -> float:
  """Returns a cached noise multiplier or calibrates it once.

  Args:
    dp_params: DP configuration to read or calibrate the noise multiplier from.
    model: Optional Keras model used to cache/reuse the calibrated value.

  Returns:
    The configured or calibrated noise multiplier.
  """
  if dp_params.noise_multiplier is not None:
    return dp_params.noise_multiplier
  if model is not None:
    cached = getattr(model, '_dp_noise_multiplier', None)
    if cached is not None:
      return cached
  calibrated = dp_params.update_with_calibrated_noise_multiplier()
  noise_multiplier = calibrated.noise_multiplier
  if model is not None:
    model._dp_noise_multiplier = noise_multiplier  # pylint: disable=protected-access
  return noise_multiplier


def _noised_clipped_grads(
    compute_loss_and_updates_fn: LossFn,
    dp_params: DPKerasConfig,
    state: _StateType,
    data: _KerasInputsDataType,
    model: keras.Model | None = None,
    *,
    is_padding_example: jax.Array | None = None,
) -> tuple[tuple[jax.typing.ArrayLike, _AuxType], optax.ArrayTree]:
  """Computes noised and clipped gradients.

  Args:
    compute_loss_and_updates_fn: The function that computes the loss and updates
      for the given state and data.
    dp_params: The parameters for DP-SGD training.
    state: The state of the model.
    data: The data for the model: triple of x, y (can be None), sample_weight
      (can be None).
    model: Optional Keras model used to cache the calibrated noise multiplier.
    is_padding_example: Optional mask marking padded examples introduced by the
      Poisson-sampled training wrapper.

  Returns:
    (loss, aux), grads
  """
  (
      trainable_variables,
      non_trainable_variables,
      optimizer_variables,
      metrics_variables,
  ) = state
  # TODO: b/415360727 - access it and update it by name.
  noise_state = non_trainable_variables[0], ()
  x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
  if is_padding_example is None:
    if dp_params.poisson_sampling_in_fit and sample_weight is not None:
      is_padding_example = _padding_mask_from_sample_weight(sample_weight)
    else:
      batch_size = jax.tree.leaves(x)[0].shape[0]
      is_padding_example = jnp.zeros(batch_size, dtype=jnp.bool_)

  clipped_grad_fn = clipping.clipped_grad(
      fun=compute_loss_and_updates_fn,
      has_aux=True,
      return_values=True,
      l2_clip_norm=dp_params.clipping_norm,
      rescale_to_unit_norm=dp_params.rescale_to_unit_norm,
      normalize_by=dp_params.batch_size,
      batch_argnums=(3, 4, 5),  # corresponding to (x, y, sample_weight)
      microbatch_size=dp_params.microbatch_size,
  )

  clipped_grad, per_example_aux = clipped_grad_fn(
      trainable_variables,
      non_trainable_variables,
      metrics_variables,
      x,
      y,
      sample_weight,
      True,  # training=True
      optimizer_variables,
      is_padding_example=is_padding_example,
  )

  noise_multiplier = _resolve_noise_multiplier(dp_params, model)
  l2_sensitivity = clipped_grad_fn.l2_norm_bound
  accumulation_factor = np.sqrt(dp_params.gradient_accumulation_steps)
  stddev = noise_multiplier * l2_sensitivity / accumulation_factor
  privatizer = noise_addition.gaussian_privatizer(stddev=stddev)

  noisy_grads, new_noise_state = privatizer.update(clipped_grad, noise_state)

  # Use masked means so padded Poisson examples do not affect the logs.
  loss = _masked_mean(per_example_aux.values, is_padding_example)
  unscaled_loss = _masked_mean(per_example_aux.aux[0], is_padding_example)
  y_pred = per_example_aux.aux[1]
  non_trainable_variables = [new_noise_state[0]] + non_trainable_variables[1:]
  # Metrics follow the same masked-mean reduction as the loss values above.
  new_metrics = jax.tree.map(
      lambda x: _masked_mean(x, is_padding_example),
      per_example_aux.aux[3],
  )

  aux = (unscaled_loss, y_pred, non_trainable_variables, new_metrics)

  return (loss, aux), noisy_grads


# This is copy-paste from
# https://github.com/keras-team/keras/blob/6b4a4dfaa26c14d3071a489e43453917f7b42e30/keras/src/backend/jax/trainer.py#L88
def _update_metrics_variables(  # pylint: disable=too-many-positional-arguments
    self: keras.Model,
    metrics_variables: _MetricsVariablesType,
    unscaled_loss: _UnscaledLossType,
    x: _XType,
    y: _YType,
    y_pred: _YType,
    sample_weight: _SampleWeightType,
) -> tuple[_LogsType, _MetricsVariablesType]:
  """Updates the metrics variables."""
  with keras.StatelessScope(
      state_mapping=list(zip(self.metrics_variables, metrics_variables))
  ) as scope:
    self._loss_tracker.update_state(  # pylint: disable=protected-access
        unscaled_loss, sample_weight=keras.tree.flatten(x)[0].shape[0]
    )
    logs = self.compute_metrics(x, y, y_pred, sample_weight)

  new_metrics_variables = []
  for ref_v in self.metrics_variables:
    new_v = scope.get_current_value(ref_v)
    if new_v is None:
      new_v = ref_v.value
    new_metrics_variables.append(new_v)
  return logs, new_metrics_variables


def _get_param(
    method_signature: inspect.Signature,
    param_name: str,
    *args,
    **kwargs,
) -> Any:
  """Returns the value of the parameter in the method call.

  This function is used to get the value of the parameter in the method
  call. It checks if the parameter is passed in *args or **kwargs and
  returns the value of the parameter if it is. Otherwise, it returns the default
  value of the parameter if it is explicitly present in the signature or throws
  exception otherwise.

  Args:
    method_signature: The signature of the method.
    param_name: The name of the parameter to get the value of.
    *args: The positional arguments passed to the method.
    **kwargs: The keyword arguments passed to the method.
  """
  parameters = method_signature.parameters
  param_index = None
  try:
    param_index = list(parameters.keys()).index(param_name)
  except ValueError:
    # The parameter name is not present in the signature, but it might be passed
    # in **kwargs.
    pass
  # Check if the parameter is passed in *args
  if param_index is not None and param_index < len(args):
    return args[param_index]
  # Check if the parameter is passed in **kwargs
  if param_name in kwargs:
    return kwargs[param_name]
  return parameters[param_name].default if param_name in parameters else None


def _get_non_trainable_weight(
    weight_name: str, model: keras.Model
) -> keras.Variable:
  """Returns the non-trainable weight with the given name."""
  return next(w for w in model.non_trainable_weights if w.name == weight_name)


def _calculate_optimizer_steps_to_perform_in_fit(
    train_size: int,
    batch_size: int,
    epochs: int,
    initial_epoch: int,
    steps_per_epoch: int | None,
    gradient_accumulation_steps: int = 1,
) -> int:
  """Returns the number of optimizer steps that will be performed by fit."""
  epochs_to_perform = epochs - initial_epoch
  if steps_per_epoch is None:
    effective_batch_size = batch_size * gradient_accumulation_steps
    optimizer_steps_per_epoch = _get_default_steps_per_epoch(
        train_size, effective_batch_size
    )
  else:
    optimizer_steps_per_epoch = math.ceil(
        steps_per_epoch / gradient_accumulation_steps
    )
  return optimizer_steps_per_epoch * epochs_to_perform


def _get_default_steps_per_epoch(train_size: int, batch_size: int) -> int:
  return max(1, math.floor(train_size / batch_size))


def _get_random_int64() -> np.int64:
  int64_info = np.iinfo(np.int64)
  return np.random.randint(
      low=int64_info.min, high=int64_info.max, dtype=np.int64
  )

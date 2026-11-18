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

r"""Standalone binary for running distributed noise generation.

This file is intended to provide a standalone example of how to generate
correlated noise efficiently with DP-MF.  It also provides sharding information
and visualizations about the intermediate arrays and their sharding to help
understand how the sharding strategy works.  It can also be used to determine if
a given model will run on a given TPU topology and give intuition about what is
feasible and what is not.  It is also a useful tool for testing and benchmarking
alternate implementations of noise generation, which can be helpful to quickly
test out how changes to the implementation will impact the performance
characteristics.

TODO: b/384095272 - Add example command for running in OSS environments.
"""

import os
import time
from typing import Any

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
from jax_privacy import noise_addition
from jax_privacy.matrix_factorization import toeplitz


jax.config.update('jax_threefry_partitionable', True)

_BANDS = flags.DEFINE_integer('bands', 8, 'Number of bands.')
_HIDDEN_SIZE = flags.DEFINE_integer('hidden_size', 1024, 'Hidden size.')
_XLA_DUMP = flags.DEFINE_string('xla_dump', None, 'XLA dump directory.')
_STEPS = flags.DEFINE_integer(
    'steps',
    1,
    'Number of steps of noise generation to run. Larger values can give more'
    ' reliable estimates of runtime.',
)
_MODEL = flags.DEFINE_enum('model', 'bert', ['bert', 'toy'], 'Model to use.')


def bert_shapes(hidden_size: int) -> Any:
  """Returns a PyTree of shapes for a Bert model."""
  # Meant to demonstrate that our code works with complex model structures.
  return {
      'params': {
          'lm': {
              'final_ln': {'bias': (hidden_size,), 'scale': (hidden_size,)},
              'softmax': {
                  'logits_ffn': {
                      'bias': {'b': (32000,)},
                      'linear': {'w': (hidden_size, 32000)},
                  }
              },
          },
          'transformer': {
              'x_layers_0': {
                  'ff_layer': {
                      'ffn_layer1': {
                          'bias': {'b': (4 * hidden_size,)},
                          'linear': {'w': (hidden_size, 4 * hidden_size)},
                      },
                      'ffn_layer2': {
                          'bias': {'b': (hidden_size,)},
                          'linear': {'w': (4 * hidden_size, hidden_size)},
                      },
                      'layer_norm': {
                          'bias': (hidden_size,),
                          'scale': (hidden_size,),
                      },
                  },
                  'layer_norm': {
                      'bias': (hidden_size,),
                      'scale': (hidden_size,),
                  },
                  'self_attention': {
                      'combined_qkv': {'w': (3, hidden_size, 16, 32)},
                      'per_dim_scale': {'per_dim_scale': (32,)},
                      'post': {'w': (hidden_size, 16, 32)},
                  },
              }
          },
      }
  }


def bert_model_params(hidden_size: int) -> Any:
  """Returns a PyTree of model parameters for a Bert model."""

  # Model fits on one device, so we configure it for pure data parallelism.

  model_params = jax.tree.map(
      jnp.zeros,
      bert_shapes(hidden_size),
      is_leaf=lambda x: isinstance(x, tuple),
  )

  return jax.sharding.reshard(model_params, jax.sharding.PartitionSpec())


def toy_model_params(hidden_size: int) -> jax.Array:
  """Returns model parameters for a toy model."""
  # This is a toy example where the model is just a 2D array of size (H, H^2).
  leaf_shape = (hidden_size, hidden_size**2)

  return jax.sharding.reshard(
      jnp.zeros(leaf_shape), jax.sharding.PartitionSpec('x', 'y')
  )


def generate_noise(
    model_params: Any, bands: int = 4, steps: int = 1
) -> tuple[Any, Any]:
  """Generates noise for DP-BandMF."""

  param_size = sum(jax.tree.flatten(jax.tree.map(jnp.size, model_params))[0])
  state_size = param_size * bands
  total_memory = state_size * 4 // 2**30
  per_device_memory = total_memory / jax.device_count()

  print(f'[BandMF] Total Memory: {total_memory} GiB')
  print(f'[BandMF] Per Chip Memory: {per_device_memory} GiB')

  iterations = 1_000
  strategy_coefs = toeplitz.optimal_max_error_strategy_coefs(bands)
  privatizer = noise_addition.matrix_factorization_privatizer(
      noising_matrix=toeplitz.inverse_as_streaming_matrix(
          strategy_coefs, column_normalize_for_n=iterations
      ),
      stddev=1.0,
      rng=0,
      intermediate_strategy=noise_addition.SupportedStrategies.ZERO,
  )

  @jax.jit
  def run(pytree_like_model_params):
    state = privatizer.init(pytree_like_model_params)
    noisy_grad = None
    for _ in range(steps):
      # In real applications, pass in the actual clipped gradient here.
      # For benchmarking, we just pass in something that has the same structure.
      noisy_grad, state = privatizer.update(
          pytree_like_model_params,
          state,
      )
    return state, noisy_grad

  t0 = time.time()
  compiled_run = run.lower(model_params).compile()
  t1 = time.time()
  print(f'[BandMF] Compilation time: {t1-t0:.3f} seconds')
  state, noisy_grad = jax.block_until_ready(compiled_run(model_params))
  t2 = time.time()
  print(f'[BandMF] Per-step run time: {(t2-t1)/steps:.3f} seconds')

  return state, noisy_grad


def main(_):

  if _XLA_DUMP.value:
    os.environ['XLA_FLAGS'] = (
        os.environ.get('XLA_FLAGS', '') + f' --xla_dump_to={_XLA_DUMP.value}'
    )

  assert jax.device_count() >= 8, 'Toy model requires at least 8 devices.'
  axis_types = (jax.sharding.AxisType.Explicit,) * 3
  mesh = jax.make_mesh(
      (2, 4, jax.device_count() // 8), ('x', 'y', 'z'), axis_types=axis_types
  )

  if _MODEL.value == 'bert':

    with jax.set_mesh(mesh):
      params = bert_model_params(_HIDDEN_SIZE.value)
      state, noisy_grad = generate_noise(params, _BANDS.value, _STEPS.value)

    def qkv(tree):
      return tree['params']['transformer']['x_layers_0']['self_attention']['combined_qkv']['w']  # pylint: disable=line-too-long

    print('[BandMF] Model Shape + Sharding [combined_qkv]')
    print(qkv(params).shape)
    print(qkv(params).sharding)

    print('[BandMF] State Shape + Sharding [combined_qkv]')
    print(qkv(state[1])[1].shape)
    print(qkv(state[1])[1].sharding)

    print('[BandMF] Correlated Noise Shape + Sharding [combined_qkv]')
    print(qkv(noisy_grad).shape)
    print(qkv(noisy_grad).sharding)

  else:
    with jax.set_mesh(mesh):
      params = toy_model_params(_HIDDEN_SIZE.value)
      state, noisy_grad = generate_noise(params, _BANDS.value, _STEPS.value)

    print('[BandMF] Model Shape + Sharding')
    print(params.shape)
    print(params.sharding)
    jax.debug.visualize_array_sharding(params)

    print('[BandMF] State Shape + Sharding')
    print(state[1][1].shape)
    print(state[1][1].sharding)
    jax.debug.visualize_array_sharding(state[1][1])

    print('[BandMF] Correlated Noise Shape + Sharding')
    print(noisy_grad.shape)
    print(noisy_grad.sharding)
    jax.debug.visualize_array_sharding(noisy_grad)


if __name__ == '__main__':
  app.run(main)

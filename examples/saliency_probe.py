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

"""Probes a tiny MLP and accounts for a planned DP-SGD fine-tune.

Run on CPU: JAX_PLATFORMS=cpu python examples/saliency_probe.py
Only the probe is executed; the downstream training event is illustrative.
"""

import secrets

import dp_accounting
import jax
import jax.numpy as jnp
from jax_privacy import accounting
from jax_privacy.experimental import saliency


def main() -> None:
  # Public initialization and synthetic data; neither needs a private seed.
  params = {'hidden': jnp.eye(2, 4), 'output': jnp.ones((4, 1))}
  x = jax.random.normal(jax.random.key(0), (32, 2))
  data = {'x': x, 'y': x[:, :1]}

  def loss_fn(params, batch):
    hidden = jax.nn.relu(batch['x'] @ params['hidden'])
    predictions = hidden @ params['output']
    return jnp.mean((predictions - batch['y']) ** 2)

  # Keep noise randomness private and independent of the public data seed.
  noise_key = jax.random.wrap_key_data(
      jnp.array([secrets.randbits(32), secrets.randbits(32)], dtype=jnp.uint32),
      impl='threefry2x32',
  )
  noise_multiplier = 3.0
  result = saliency.topk_vote_probe(
      loss_fn,
      data,
      params,
      vote_top_k=1,
      select_top_k=1,
      noise_multiplier=noise_multiplier,
      candidate_mask={'hidden': True, 'output': True},
      prng_key=noise_key,
  )
  print('Selected parameter leaves:', result.selected_mask)
  for index, score in result.ranked_scores:
    print(f'  {sorted(params)[index]}: noisy votes = {score:.2f}')

  # Full-batch probe: no sampling amplification; add/remove-one adjacency.
  probe_event = dp_accounting.GaussianDpEvent(noise_multiplier)
  print('Probe event:', probe_event)
  # A hypothetical fine-tune on the same population, with independent Poisson
  # sampling at each step. Its noise must also be independent of the probe's.
  training_event = accounting.dpsgd_event(
      noise_multiplier=2.0,
      iterations=20,
      sampling_prob=0.125,
  )
  composed_event = dp_accounting.ComposedDpEvent([probe_event, training_event])
  delta = 1e-5
  epsilon = (
      dp_accounting.rdp.RdpAccountant()
      .compose(composed_event)
      .get_epsilon(delta)
  )
  print(
      f'Planned probe + training budget: epsilon={epsilon:.3f}, delta={delta}'
  )
  print('Training was not run; the budget assumes the stated DP-SGD plan.')


if __name__ == '__main__':
  main()

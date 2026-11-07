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

"""Monte Carlo discretized privacy-loss sample accumulation."""

import math

import jax
import jax.numpy as jnp
from jax_privacy import _validate
from jax_privacy import batch_selection
from jax_privacy.experimental.monte_carlo import delta_calculation
from jax_privacy.experimental.monte_carlo import jax_sample_generation
from jax_privacy.experimental.monte_carlo import jax_scoring
import numpy as np

__all__ = [
    'privacy_loss_samples',
]

# Strategies whose scoring is supported by jax_scoring.compute_privacy_loss.
_Strategy = (
    batch_selection.MinSepRandomAllocation
    | batch_selection.RandomAllocationSampling
)


def _make_pattern_sampler(strategy: _Strategy):
  """Returns ``key -> (positions, poisson_mask)`` for ``strategy``."""
  if isinstance(strategy, batch_selection.RandomAllocationSampling):
    return lambda key: jax_sample_generation.sample_random_allocation(
        key,
        strategy.cycle_length,
        strategy.total_participations,
        strategy.iterations,
        sampling_prob=strategy.sampling_prob,
    )
  if isinstance(strategy, batch_selection.MinSepRandomAllocation):
    return lambda key: jax_sample_generation.sample_min_sep_random_allocation(
        key,
        strategy.min_sep,
        strategy.total_participations,
        strategy.iterations,
        random_rotation=strategy.random_rotation,
        sampling_prob=strategy.sampling_prob,
    )
  raise ValueError(f'Unsupported strategy: {type(strategy).__name__}')


def _to_samples(
    counts: jax.Array, grid: np.ndarray
) -> delta_calculation.PrivacyLossSamples:
  """Packages nonzero bin counts into a ``PrivacyLossSamples``."""
  counts = np.asarray(counts)
  nonzero = counts > 0
  return delta_calculation.PrivacyLossSamples(
      privacy_loss=grid[nonzero], counts=counts[nonzero]
  )


def _accumulate_bincounts(
    per_sample, num_samples, num_bins, *, microbatch_size
):
  """Sums per-sample bin indices into histograms of length ``num_bins``."""
  # A fori_loop over microbatches derives sample indices from the loop counter,
  # so no length-num_samples buffer is materialized and peak memory is
  # O(num_bins + microbatch_size).

  def block(start):
    indices = start + jnp.arange(microbatch_size, dtype=jnp.uint32)
    bins = jax.vmap(per_sample)(indices)
    return jax.tree.map(lambda b: jnp.bincount(b, length=num_bins), bins)

  def body(i, carry):
    return jax.tree.map(jnp.add, carry, block(i * microbatch_size))

  num_microbatches = num_samples // microbatch_size
  return jax.lax.fori_loop(1, num_microbatches, body, block(0))


# Compile the entire pipeline (sampling, scoring, and histogram accumulation)
# into a single jitted function for efficiency.
@jax.jit(
    static_argnums=(0,),
    static_argnames=('microbatch_size',),
)
def _sample_losses_and_bin(
    strategy: _Strategy,
    noise_multiplier: float,
    c_col: jax.typing.ArrayLike,
    grid: jax.typing.ArrayLike,
    key: jax.Array,
    *,
    num_samples: int,
    microbatch_size: int,
) -> tuple[jax.Array, jax.Array]:
  """Jitted sampling + scoring + histogram accumulation (see public wrapper)."""
  iterations = strategy.iterations
  sample_pattern = _make_pattern_sampler(strategy)
  grid = jnp.asarray(grid)

  def per_sample(index: jax.Array) -> tuple[jax.Array, jax.Array]:
    # Algorithm 3 of https://arxiv.org/abs/2610.12712: draw a participation
    # pattern, form positive/negative observations, and score both directions.
    sample_key = jax.random.fold_in(key, index)
    pattern_key, pos_key, neg_key = jax.random.split(sample_key, 3)
    pattern, keep = sample_pattern(pattern_key)
    mode = jax_sample_generation.banded_c_times_pattern(
        c_col, pattern, iterations, keep
    )
    noise_pos = jax.random.normal(pos_key, (iterations,), dtype=c_col.dtype)
    noise_neg = jax.random.normal(neg_key, (iterations,), dtype=c_col.dtype)
    # Positive: y ~ P (mode + noise); negative: y ~ Q (noise), loss negated.
    loss_pos = jax_scoring.compute_privacy_loss(
        strategy, mode + noise_multiplier * noise_pos, noise_multiplier, c_col
    )
    loss_neg = -jax_scoring.compute_privacy_loss(
        strategy, noise_multiplier * noise_neg, noise_multiplier, c_col
    )
    # side='left' returns the first index i with grid[i] >= loss, rounding each
    # loss up to the next grid point (or the trailing +inf bin on overflow).
    return (
        jnp.searchsorted(grid, loss_pos, side='left'),
        jnp.searchsorted(grid, loss_neg, side='left'),
    )

  return _accumulate_bincounts(
      per_sample, num_samples, grid.size, microbatch_size=microbatch_size
  )


def privacy_loss_samples(
    strategy: _Strategy,
    noise_multiplier: float,
    c_col: jax.typing.ArrayLike,
    key: jax.Array,
    *,
    num_samples: int,
    microbatch_size: int | None = None,
    grid_lo: float = -20.0,
    grid_hi: float = 30.0,
    grid_size: int = 50_000,
) -> tuple[
    delta_calculation.PrivacyLossSamples, delta_calculation.PrivacyLossSamples
]:
  """Accumulates discretized privacy-loss samples for a strategy and noise.

  Draws ``num_samples`` dominating-pair observations in each direction (Section
  3.4, Algorithm 3 of https://arxiv.org/abs/2610.12712) and rounds the privacy
  losses up onto a uniform grid of ``grid_size`` points over ``[grid_lo,
  grid_hi]`` (losses above ``grid_hi`` map to ``+inf``). Following
  ``dp_accounting``'s PLD convention, rounding up makes each stored loss an
  over-estimate of the true loss.

  Args:
    strategy: The batch selection strategy. Supported:
      ``MinSepRandomAllocation`` and ``RandomAllocationSampling``.
    noise_multiplier: Standard deviation ``sigma`` of the Gaussian noise.
    c_col: The nonzero first-column band of the Toeplitz matrix ``C``, with
      ``len(c_col) <= b`` so participation bands do not overlap.
    key: JAX PRNG key.
    num_samples: Minimum number of samples per direction; rounded up to a whole
      multiple of ``microbatch_size``.
    microbatch_size: Samples processed per batch (larger values run faster but
      use more device memory); defaults to ``num_samples``.
    grid_lo: Lowest grid point.
    grid_hi: Highest finite grid point.
    grid_size: Number of finite grid points spanning ``[grid_lo, grid_hi]``.

  Returns:
    A pair ``(positive_samples, negative_samples)`` of
    :class:`~.delta_calculation.PrivacyLossSamples` holding the nonzero bin
    grid points and counts for ``L(y)`` (``y ~ P``) and ``-L(y)`` (``y ~ Q``).
  """
  if microbatch_size is None:
    microbatch_size = num_samples
  _validate.positive(
      num_samples=num_samples,
      microbatch_size=microbatch_size,
      grid_size=grid_size,
  )
  microbatch_size = min(microbatch_size, num_samples)
  # Round up so every microbatch is full; treats num_samples as a minimum.
  num_samples = math.ceil(num_samples / microbatch_size) * microbatch_size

  grid = np.r_[np.linspace(grid_lo, grid_hi, grid_size), np.inf]
  counts_pos, counts_neg = _sample_losses_and_bin(
      strategy,
      noise_multiplier,
      c_col,
      grid,
      key,
      num_samples=num_samples,
      microbatch_size=microbatch_size,
  )
  return _to_samples(counts_pos, grid), _to_samples(counts_neg, grid)

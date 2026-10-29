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

"""Pure-JAX sampling of participation patterns for Monte Carlo accounting.

Each function draws the participation pattern of a *single* example under one of
the fixed-participation strategies in ``batch_selection``: an example
participates in exactly ``num_participations`` (K) of the ``iterations`` (T)
steps. Each sampler returns a pair ``(positions, poisson_mask)``: ``positions``
is a sorted length-K int array of the sampled iteration indices, and
``poisson_mask`` is a length-K boolean array marking which of those
participations survive Poisson subsampling (all True when ``sampling_prob ==
1``). Feed both to ``banded_c_times_pattern`` to form ``C @ x`` for the kept
participations.

These helpers deliberately handle a single example (no batch dimension); draw a
batch by mapping over PRNG keys, e.g.
``jax.vmap(sample_random_allocation, in_axes=(0, None, None, None))``. The
size-determining arguments (``cycle_length``, ``num_participations``,
``iterations``, ``min_sep``, ``random_rotation``) are static and must be Python
scalars when the functions are jitted.
"""

import jax
import jax.numpy as jnp

__all__ = [
    'banded_c_times_pattern',
    'sample_min_sep_random_allocation',
    'sample_random_allocation',
]


def _sample_k_distinct(key: jax.Array, n: int, k: int) -> jax.Array:
  """Returns a uniform ``k``-subset of ``[0, n)`` as a sorted int array.

  Uses Floyd's algorithm: ``k`` iterations, each drawing a single integer and
  resolving collisions against the (at most ``k``) already-chosen values. This
  costs ``O(k^2)`` and touches no length-``n`` buffer, unlike
  ``jax.random.choice(..., replace=False)``, which materialises and sorts a
  length-``n`` permutation. For the Monte Carlo generation regime (``k = K`` a
  handful, ``n ~ T`` in the thousands) that permutation dominates end-to-end
  runtime, so this is the fast path. ``n`` and ``k`` are static Python ints with
  ``0 <= k <= n``.

  Args:
    key: JAX PRNG key.
    n: Size of the universe ``[0, n)``.
    k: Number of distinct elements to draw.

  Returns:
    Sorted int array of shape ``(k,)`` with distinct entries in ``[0, n)``.
  """
  if k == 0:
    return jnp.zeros((0,), dtype=jnp.int32)

  def body(i: jax.Array, chosen: jax.Array) -> jax.Array:
    # Floyd: at step i draw t in [0, j] where j = n - k + i. If t was already
    # chosen, take j instead -- j exceeds every earlier pick, so it is fresh.
    j = n - k + i
    t = jax.random.randint(jax.random.fold_in(key, i), (), 0, j + 1)
    already = jnp.any(chosen == t)  # Unfilled slots are -1, never equal to t.
    return chosen.at[i].set(jnp.where(already, j, t))

  chosen = jax.lax.fori_loop(0, k, body, jnp.full((k,), -1, dtype=jnp.int32))
  return jnp.sort(chosen)


def _random_composition(
    key: jax.Array, total: int, num_parts: int
) -> jax.Array:
  """Returns a uniform composition of ``total`` into ``num_parts`` parts.

  Stars and bars: choose ``num_parts - 1`` distinct bars in
  ``[0, total + num_parts - 1)``; the gaps between the sorted bars (minus one)
  are the nonnegative parts, which sum to ``total``. ``total`` and ``num_parts``
  are static Python ints.

  Args:
    key: JAX PRNG key.
    total: The value the parts sum to.
    num_parts: The number of parts.

  Returns:
    Int array of shape ``(num_parts,)`` summing to ``total``.
  """
  n = total + num_parts - 1
  bars = _sample_k_distinct(key, n, num_parts - 1)
  return jnp.diff(jnp.r_[-1, bars, n]) - 1


def sample_random_allocation(
    key: jax.Array,
    cycle_length: int,
    num_participations: int,
    iterations: int,
    sampling_prob: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
  """Samples one participation pattern for ``RandomAllocationSampling``.

  The example is assigned a bin ``i`` uniformly from ``{0, ..., b - 1}`` and
  then
  random-allocates ``num_participations`` of that bin's candidate iterations
  ``{i, i + b, i + 2b, ...} n [0, T)``. Because a bin's iterations are exactly
  ``cycle_length`` apart, every gap is a multiple of ``cycle_length`` and the
  minimum separation is automatic. Assumes ``iterations`` is a multiple of
  ``cycle_length``, so every bin has exactly ``iterations // cycle_length``
  candidates.

  When ``sampling_prob < 1`` each allocated participation is additionally kept
  independently with that probability, recorded in the returned
  ``poisson_mask``;
  the ``positions`` themselves are unchanged.

  Args:
    key: JAX PRNG key.
    cycle_length: The number of bins ``b`` (the minimum separation).
    num_participations: The number ``K`` of iterations to participate in.
    iterations: The total number of iterations ``T``.
    sampling_prob: Probability of keeping each allocated participation. The
      default of 1.0 keeps all of them (no subsampling).

  Returns:
    A pair ``(positions, poisson_mask)``. ``positions`` is a sorted int array of
    shape ``(num_participations,)`` of iteration indices; ``poisson_mask`` is a
    boolean array of the same shape marking the kept participations (all True
    when ``sampling_prob == 1``).
  """
  assert iterations % cycle_length == 0, 'iterations must be a multiple of b.'
  candidates_per_bin = iterations // cycle_length
  assert num_participations <= candidates_per_bin, 'K exceeds bin size T / b.'
  bin_key, pos_key = jax.random.split(key)
  chosen_bin = jax.random.randint(bin_key, (), 0, cycle_length)
  # Every bin has T / b candidates {i, i + b, ...}; random-allocate K of them.
  chosen = _sample_k_distinct(pos_key, candidates_per_bin, num_participations)
  positions = jnp.sort(chosen_bin + cycle_length * chosen)
  if sampling_prob < 1.0:
    # Keep each participation independently with probability sampling_prob.
    mask = jax.random.bernoulli(
        jax.random.fold_in(key, 2), sampling_prob, (num_participations,)
    )
  else:
    mask = jnp.ones(num_participations, dtype=bool)
  return positions, mask


def sample_min_sep_random_allocation(
    key: jax.Array,
    min_sep: int,
    num_participations: int,
    iterations: int,
    random_rotation: bool = False,
    sampling_prob: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
  """Samples one participation pattern for ``MinSepRandomAllocation``.

  Both branches lay out ``K`` positions from a uniform composition (see
  ``_random_composition``): the linear branch leaves the two ends free, while
  the
  cyclic branch also constrains the wrap-around gap and then rotates.

  When ``sampling_prob < 1`` each participation is additionally kept
  independently with that probability, recorded in the returned
  ``poisson_mask``;
  the ``positions`` themselves are unchanged.

  Args:
    key: JAX PRNG key.
    min_sep: The minimum separation ``b`` between consecutive participations.
    num_participations: The number ``K`` of iterations to participate in.
    iterations: The total number of iterations ``T``.
    random_rotation: If True, place the ``K`` participations on a length-``T``
      cycle (all gaps ``>= min_sep``, including the wrap-around) and apply a
      uniform random rotation, making the per-iteration marginal exactly ``K /
      T``. If False, leave the two ends free, which over-represents the start
      and end of training.
    sampling_prob: Probability of keeping each participation. The default of 1.0
      keeps all of them (no subsampling).

  Returns:
    A pair ``(positions, poisson_mask)``. ``positions`` is a sorted int array of
    shape ``(num_participations,)`` of iteration indices; ``poisson_mask`` is a
    boolean array of the same shape marking the kept participations (all True
    when ``sampling_prob == 1``).
  """
  k, b, t = num_participations, min_sep, iterations
  assert t >= k * b, 'iterations must be at least num_participations * min_sep.'
  if not random_rotation:
    # K - 1 interior gaps of size >= b plus free space at both ends: distribute
    # the leftover T - 1 - (K - 1) b over K + 1 parts, keep the first K, and
    # offset by b per step. Free ends over-represent the start/end of training.
    parts = _random_composition(key, t - 1 - (k - 1) * b, k + 1)
    positions = jnp.cumsum(parts[:k]) + jnp.arange(k) * b
  else:
    # K cyclic gaps each >= b summing to T: distribute the slack T - K*b over K
    # parts and add b to each, then apply a uniform rotation, which is what
    # flattens the marginal to K / T.
    comp_key, rot_key = jax.random.split(key)
    parts = _random_composition(comp_key, t - k * b, k)
    starts = jnp.cumsum(parts) + jnp.arange(k) * b
    rotation = jax.random.randint(rot_key, (), 0, t)
    positions = jnp.sort((starts + rotation) % t)
  if sampling_prob < 1.0:
    # Keep each participation independently with probability sampling_prob.
    mask = jax.random.bernoulli(
        jax.random.fold_in(key, 2), sampling_prob, (num_participations,)
    )
  else:
    mask = jnp.ones(num_participations, dtype=bool)
  return positions, mask


def banded_c_times_pattern(
    c_col: jax.Array,
    part_pattern: jax.Array,
    iterations: int,
    poisson_mask: jax.Array,
) -> jax.Array:
  """Multiplies banded Toeplitz ``C`` by a 0/1 participation vector.

  ``C`` is the ``iterations x iterations`` lower-triangular banded Toeplitz
  matrix with ``C[i, j] = c_col[i - j]`` for ``0 <= i - j < len(c_col)`` (and 0
  otherwise); that is, ``c_col`` is the band of nonzero first-column entries.
  The
  participation vector ``x`` is the 0/1 indicator of ``part_pattern``
  (``x[p] = 1`` iff ``p in part_pattern`` and ``poisson_mask`` keeps ``p``), so
  this returns ``C @ x``:

    out[i] = sum_{p in part_pattern kept, 0 <= i - p < b} c_col[i - p].

  We assume the participations are at least ``b`` apart (``part_pattern`` sorted
  with gaps ``>= len(c_col)``), which holds by construction whenever the min
  separation is at least the number of bands. The bands then never overlap, so
  we build the ``(K, b)`` block of band rows directly and scatter-add ``c_col``
  into the length-``T`` output, touching only the ``K * len(c_col)`` nonzero
  entries. Because the bands are disjoint the scattered adds never collide.
  Participations whose ``poisson_mask`` entry is False (dropped by Poisson
  subsampling) contribute nothing.

  The size-determining arguments (``len(c_col)``, ``len(part_pattern)``,
  ``iterations``) are static; the function is jittable and can be mapped over a
  batch of patterns and masks with
  ``jax.vmap(..., in_axes=(None, 0, None, 0))``.

  Args:
    c_col: 1D array of the ``b`` nonzero first-column entries of banded ``C``.
    part_pattern: 1D sorted int array of the ``K`` iteration indices where the
      example participates (e.g. the output of the samplers above), with gaps
      ``>= len(c_col)``.
    iterations: The total number of iterations ``T`` (the output length).
    poisson_mask: 1D boolean array of shape ``(K,)`` aligned with
      ``part_pattern``; a False entry drops that participation (its band
      contributes 0). Pass an all-True mask for no subsampling.

  Returns:
    Array of shape ``(iterations,)`` equal to ``C @ x``.
  """
  b = c_col.shape[0]
  # Each participation p contributes c_col to rows p, p+1, ..., p+b-1. Build the
  # (K, b) block of those row indices and scatter-add the band values into the
  # length-T output. Rows past the last iteration are clipped and masked to 0;
  # disjoint bands mean the adds never collide. Participations dropped by
  # Poisson subsampling (poisson_mask False) contribute nothing.
  rows = part_pattern[:, None] + jnp.arange(b)[None, :]
  in_band = (rows < iterations) & poisson_mask[:, None]
  vals = jnp.where(in_band, c_col[None, :], 0.0)
  out = jnp.zeros((iterations,), dtype=c_col.dtype)
  return out.at[jnp.clip(rows, 0, iterations - 1)].add(vals)

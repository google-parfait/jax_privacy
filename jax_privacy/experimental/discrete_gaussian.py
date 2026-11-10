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

"""Utilities for sampling from the discrete Gaussian distribution.

This implements the sampling algorithm for the discrete Gaussian distribution
from the paper:
  Clement Canonne, Gautam Kamath, & Thomas Steinke
  "The Discrete Gaussian for Differential Privacy"
  NeurIPS 2020 https://arxiv.org/abs/2004.00010
Note that the implementation here uses floating point arithmetic for
computing probabilities, which can result in small inaccuracies, but
not catastrophic failures. Roughly speaking, the floating point errors in
computing the probabilities are added to the delta term in differential privacy.
"""

import jax
import numpy as np
import optax

from .. import _validate


def _get_sampling_parameters(sigma: float, size: int) -> tuple[float, int]:
  """Computes the sampling parameters for the discrete Gaussian sampler.

  The discrete Gaussian sampling algorithm is correct for *any* choice of
  laplace_scale > 0 and oversample >= 1. However, the choice of parameters
  affects the efficiency of the algorithm.

  These parameter settings are tuned numerically to ensure good performance.
  First, laplace_scale is chosen so that the probability of accepting each
  sample in the rejection sampling step is >60%.
  Second, oversample is chosen to be large enough to ensure that if we generate
  oversample samples, then after rejection sampling we have at least size
  samples remaining with >95% probability. Thus the sampling algorithm will
  stop after one iteration with >95% probability.

  Args:
    sigma: Standard deviation proxy of the discrete Gaussian distribution.
    size: Size of the output array.

  Returns:
    A tuple of (laplace_scale, oversample) for the discrete Gaussian sampler.
    laplace_scale: The scale parameter of the Laplace distribution.
    oversample: The number of samples to draw before rejection sampling.
  """
  laplace_scale = sigma * 1.3
  oversample = (size * 5) // 3 + int(np.sqrt(size) * 2) + 4
  return laplace_scale, oversample


def sample_discrete_gaussian(
    rng: np.random.Generator,
    sigma: float,
    size: int,
    oversample: int | None = None,
    dtype: np.typing.DTypeLike = np.int64,
) -> np.ndarray:
  r"""Generates samples from a discrete Gaussian distribution.

  Specifically
  :math:`\mathbb{P}[\text{output}=x] \propto \exp(-\|x\|_2^2 / (2 \sigma^2))`
  for integer vectors :math:`x`. The output is a vector of integers of the given
  size. Samples are i.i.d. with mean 0 and variance :math:`< \sigma^2`.
  See "The Discrete Gaussian for Differential Privacy"
  `Canonne et al. (2020) <https://arxiv.org/abs/2004.00010>`_ for more details.

  Args:
    rng: PRNG to use for sampling.
    sigma: Standard deviation proxy of the discrete Gaussian distribution.
    size: Size of the output array.
    oversample: Number of samples to draw before rejection sampling. (If
      ``None``, a default value is used. This parameter is only useful for
      tuning the algorithm for efficiency purposes.)
    dtype: Output dtype. Must be an integer type. Defaults to ``np.int64``.

  Returns:
    An array of the given size sampled from the discrete Gaussian distribution.

  Raises:
    ValueError: If ``size <= 0``, ``sigma < 0``, or ``oversample <= 0``.
    ValueError: If dtype is not an integer type.
  """
  if not np.issubdtype(dtype, np.integer):
    raise ValueError(f"dtype must be an integer type, got {dtype}.")
  _validate.positive(size=size)
  _validate.non_negative(sigma=sigma)
  if sigma == 0:  # degenerate case, return 0
    return np.zeros(size, dtype=dtype)
  if oversample is None:
    laplace_scale, oversample = _get_sampling_parameters(sigma, size)
  else:
    _validate.positive(oversample=oversample)
    laplace_scale, _ = _get_sampling_parameters(sigma, size)
  p = -np.expm1(-1 / laplace_scale)
  ans = np.zeros(size, dtype=np.int64)
  num_accepted = 0  # Number of samples generated so far.
  while num_accepted < size:
    # Step 1: Sample Laplace noise as difference of two geometric variables.
    laplace_samples = rng.geometric(p, size=oversample)
    laplace_samples -= rng.geometric(p, size=oversample)
    # Step 2: Compute acceptance probability and perform rejection sampling.
    accept_prob = np.exp(
        -((np.abs(laplace_samples) - sigma**2 / laplace_scale) ** 2)
        / sigma**2
        / 2
    )  # accept = bernoulli(accept_prob):
    accept = rng.uniform(size=oversample) < accept_prob
    new_samples = laplace_samples[accept]  # These are the samples we generated.
    # Step 3: Append the new samples to ans.
    num_to_take = min(size - num_accepted, new_samples.shape[0])
    ans[num_accepted : num_accepted + num_to_take] = new_samples[:num_to_take]
    num_accepted += num_to_take
    # Step 4: Generate more samples if needed.
    # Reduce the oversample size to avoid wasting time:
    _, new_oversample = _get_sampling_parameters(sigma, size - num_accepted)
    oversample = min(new_oversample, oversample)
  return ans.astype(dtype)


def sample_discrete_gaussian_pytree(
    rng: np.random.Generator,
    sigma: float,
    pytree: optax.ArrayTree,
    oversample: int | None = None,
    dtype: np.typing.DTypeLike = np.int64,
) -> optax.ArrayTree:
  """Generates discrete Gaussian samples matching the structure of a PyTree.

  Specifically, samples a flat array of discrete Gaussian noise for the total
  number of parameters in ``pytree`` and reshapes it to match the PyTree
  structure.

  Args:
    rng: PRNG to use for sampling.
    sigma: Standard deviation proxy of the discrete Gaussian distribution.
    pytree: PyTree defining the desired output structure and shapes.
    oversample: Number of samples to draw before rejection sampling. (If
      ``None``, a default value is used. This parameter is only useful for
      tuning the algorithm for efficiency purposes.)
    dtype: Output dtype. Must be an integer type. Defaults to ``np.int64``.

  Returns:
    A PyTree of discrete Gaussian samples matching the structure of ``pytree``.
  """
  leaves, treedef = jax.tree_util.tree_flatten(pytree)
  num_params = sum(p.size for p in leaves)
  if num_params == 0:
    flat_noise = np.zeros(0, dtype=dtype)
  else:
    flat_noise = sample_discrete_gaussian(
        rng,
        sigma=sigma,
        size=num_params,
        oversample=oversample,
        dtype=dtype,
    )
  noise_leaves = []
  offset = 0
  for p in leaves:
    noise_leaves.append(flat_noise[offset : offset + p.size].reshape(p.shape))
    offset += p.size
  return jax.tree_util.tree_unflatten(treedef, noise_leaves)

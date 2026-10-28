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

r"""Importance sampling for Monte Carlo accounting of DP-BandMF.

Standard Monte Carlo accounting estimates the hockey-stick divergence between
the dominating pair :math:`(P, Q)` as the mean of samples bounded by 1, where
only a :math:`\delta` fraction of samples is non-zero, so on the order of
:math:`1 / \delta` samples are needed. Importance sampling instead draws from a
proposal distribution :math:`P_w`, whose modes are those of the mixture
:math:`P` stretched by :math:`w`, and reweights each sample by the ratio of the
evaluation density to the proposal density. Writing :math:`\{u, v\} = \{0, 1\}`
for the baseline and evaluation distributions (:math:`P_1 = P`,
:math:`P_0 = Q`) and :math:`w = u + \alpha (v - u)` for a stretch factor
:math:`\alpha \geq 1`, every weighted sample lies in :math:`[0, B(\alpha)]` with

.. math::

  B(\alpha) = \frac{(\alpha - 1)^{\alpha - 1}}{\alpha^\alpha}
  \exp\left(\frac{\kappa (\alpha^2 - \alpha)}{2} - (\alpha - 1) \varepsilon
  \right), \qquad \kappa = \max_i \|c_i\|_2^2 / \sigma^2,

where the :math:`c_i` are the modes of :math:`P` and :math:`\sigma` is the
noise multiplier. The number of samples needed then scales with
:math:`B(\alpha) / \delta` instead of :math:`1 / \delta`; see
:func:`~.delta_calculation.get_overall_delta`.

Currently only :class:`~jax_privacy.batch_selection.BallsInBinsSampling` is
supported, since the mixture modes must be enumerable.
"""

import dp_accounting
import numpy as np
import scipy as sp

from ... import _validate
from ... import batch_selection
from . import delta_calculation
from . import sample_generation


def _validate_balls_in_bins(strategy) -> None:
  if not isinstance(strategy, batch_selection.BallsInBinsSampling):
    raise ValueError(
        'Importance sampling is only supported for balls-in-bins sampling, got'
        f' {type(strategy)}.'
    )


def log_support_bound(alpha: float, epsilon: float, kappa: float) -> float:
  r"""Log of the bound :math:`B(\alpha)` on each importance-weighted sample.

  The bound is decreasing in ``epsilon``, so a bound computed for ``epsilon``
  remains valid when the same samples are used to estimate the hockey-stick
  divergence at any larger epsilon.

  Args:
    alpha: The stretch factor, at least 1. A value of 1 corresponds to plain
      Monte Carlo, with bound 1.
    epsilon: The epsilon at which the hockey-stick divergence is estimated.
    kappa: The largest squared mode norm divided by the noise variance, see
      :func:`compute_kappa`.

  Returns:
    :math:`\ln B(\alpha)`.
  """
  _validate.in_range(1, np.inf, alpha=alpha)
  _validate.non_negative(epsilon=epsilon, kappa=kappa)
  return (
      sp.special.xlogy(alpha - 1, alpha - 1)
      - alpha * np.log(alpha)
      + kappa * (alpha**2 - alpha) / 2
      - (alpha - 1) * epsilon
  )


def optimal_stretch(epsilon: float, kappa: float) -> float:
  """The stretch factor minimizing :func:`log_support_bound`.

  Args:
    epsilon: The epsilon at which the hockey-stick divergence is estimated.
    kappa: The largest squared mode norm divided by the noise variance, see
      :func:`compute_kappa`.

  Returns:
    The minimizing stretch factor, at least 1.
  """
  _validate.non_negative(epsilon=epsilon)
  _validate.positive(kappa=kappa)

  # The derivative of log_support_bound in alpha. It increases from -inf at
  # alpha = 1 to +inf, so it has a unique root which is the minimizer.
  def _derivative(alpha):
    return (
        np.log(alpha - 1)
        - np.log(alpha)
        + kappa * (2 * alpha - 1) / 2
        - epsilon
    )

  lower = 1 + 1e-15
  if _derivative(lower) >= 0:
    # The minimizer is indistinguishable from 1 in floating point.
    return 1.0
  upper = 2.0
  while _derivative(upper) < 0:
    upper *= 2
  return sp.optimize.brentq(_derivative, lower, upper)


def compute_kappa(
    strategy: batch_selection.BallsInBinsSampling,
    noise_multiplier: float,
    c_col: np.ndarray,
) -> float:
  r"""The largest squared mode norm divided by the noise variance.

  Args:
    strategy: The balls-in-bins sampling strategy.
    noise_multiplier: The noise multiplier of DP-MF. This is multiplied by the
      clip norm, not accounting for the norm of ``c_col``.
    c_col: The non-zero entries in the first column of C. Should be non-negative
      and 1D.

  Returns:
    :math:`\kappa = \max_i \|c_i\|_2^2 / \sigma^2` over the modes :math:`c_i`
    of the positive distribution in the dominating pair.
  """
  _validate_balls_in_bins(strategy)
  _validate.positive(noise_multiplier=noise_multiplier)
  modes = sample_generation.balls_in_bins_modes(strategy, c_col)
  return np.max(np.sum(modes**2, axis=0)) / noise_multiplier**2


def reverse_divergence_bound(
    strategy: batch_selection.BallsInBinsSampling,
    noise_multiplier: float,
    c_col: np.ndarray,
    epsilon: float,
) -> float:
  """Upper bound on the reverse hockey-stick divergence, without sampling.

  Monte Carlo accounting must estimate the hockey-stick divergence in both
  directions of the dominating pair. The reverse direction is typically far
  smaller than the forward one, and this bound is often enough to show it meets
  the target delta, in which case only the forward direction needs sampling.
  The bound is exact for a single mode and informative when
  :func:`compute_kappa` is below roughly ``2 * epsilon``, the same regime in
  which importance sampling is worthwhile.

  Args:
    strategy: The balls-in-bins sampling strategy.
    noise_multiplier: The noise multiplier of DP-MF. This is multiplied by the
      clip norm, not accounting for the norm of ``c_col``.
    c_col: The non-zero entries in the first column of C. Should be non-negative
      and 1D.
    epsilon: The epsilon of the hockey-stick divergence.

  Returns:
    An upper bound on the reverse hockey-stick divergence, or the trivial bound
    1 when it is uninformative.
  """
  _validate_balls_in_bins(strategy)
  _validate.positive(noise_multiplier=noise_multiplier)
  modes = sample_generation.balls_in_bins_modes(strategy, c_col)
  sensitivity = np.linalg.norm(modes.mean(axis=1))
  # Half the mean squared distance of the modes from their mean, in noise units.
  shift = (np.mean(np.sum(modes**2, axis=0)) - sensitivity**2) / (
      2 * noise_multiplier**2
  )
  if epsilon <= shift:
    return 1.0
  sigma = noise_multiplier / sensitivity if sensitivity > 0 else np.inf
  return float(dp_accounting.get_delta_gaussian(epsilon - shift, sigma))


def get_importance_sampled_privacy_loss(
    strategy: batch_selection.BallsInBinsSampling,
    noise_multiplier: float,
    c_col: np.ndarray,
    epsilon: float,
    alpha: float | None = None,
    seed: sample_generation.Seed = None,
    positive_sample: bool = True,
    num_samples: int = 1,
) -> delta_calculation.PrivacyLossSamples:
  r"""Returns importance-weighted samples from the dominating PLD of DP-BandMF.

  This is the importance sampling analogue of
  :func:`~.sample_generation.get_privacy_loss_sample`. Samples are drawn from
  the stretched mixture :math:`P_w` with :math:`w = \alpha` for positive
  samples and :math:`w = 1 - \alpha` for negative samples, and the returned log
  weights make the weighted hockey-stick estimate unbiased.

  Example usage::

    >>> from jax_privacy.experimental.monte_carlo import delta_calculation
    >>> strategy = batch_selection.BallsInBinsSampling(
    ...     cycle_length=2, iterations=4)
    >>> positive = get_importance_sampled_privacy_loss(
    ...     strategy, noise_multiplier=1.0, c_col=np.array([1.0, 0.5]),
    ...     epsilon=1.0, seed=0, num_samples=1000)
    >>> delta = delta_calculation.delta_from_epsilon_and_samples(1.0, positive)
    >>> overall_delta = delta_calculation.get_overall_delta(
    ...     1000, delta, support_bound=positive.support_bound)
    >>> bool(positive.support_bound < 1 and delta <= overall_delta <= 1)
    True

  Args:
    strategy: The balls-in-bins sampling strategy to use.
    noise_multiplier: The noise multiplier of DP-MF. This is multiplied by the
      clip norm, not accounting for the norm of ``c_col``.
    c_col: The non-zero entries in the first column of C. Should be non-negative
      and 1D.
    epsilon: The epsilon at which the hockey-stick divergence will be estimated.
      The returned ``support_bound`` also holds for any larger epsilon.
    alpha: The stretch factor, at least 1. If ``None``, the factor minimizing
      the support bound, :func:`optimal_stretch`, is used. Raises if the
      resulting support bound exceeds 1, i.e. if ``alpha`` is a worse choice
      than plain Monte Carlo.
    seed: The rng or seed to use for sampling.
    positive_sample: If ``True``, the evaluation distribution is the one in the
      dominating pair where the sensitive example is included, otherwise the one
      where it is excluded.
    num_samples: The number of samples to generate. It is typically much more
      efficient to generate multiple samples in a single call.

  Returns:
    ``num_samples`` importance-weighted
    :class:`~.delta_calculation.PrivacyLossSamples`, with ``log_weights`` the
    log of the evaluation density over the proposal density for each sample.
  """
  kappa = compute_kappa(strategy, noise_multiplier, c_col)
  if alpha is None:
    alpha = optimal_stretch(epsilon, kappa)
  log_bound = log_support_bound(alpha, epsilon, kappa)
  if log_bound > 0:
    raise ValueError(
        f'alpha={alpha} gives a support bound of {np.exp(log_bound)} > 1, which'
        ' is worse than plain Monte Carlo. Use optimal_stretch, or alpha=1.'
    )

  mode_scale = alpha if positive_sample else 1.0 - alpha
  samples = sample_generation.generate_sample(
      strategy,
      noise_multiplier,
      c_col,
      seed=seed,
      num_samples=num_samples,
      mode_scale=mode_scale,
  )

  def _log_density_ratio(scale: float) -> np.ndarray:
    # log P_scale(y) / Q(y).
    return sample_generation.compute_privacy_loss(
        strategy, samples, noise_multiplier, c_col, mode_scale=scale
    )

  positive_privacy_loss = _log_density_ratio(1.0)
  proposal_log_density_ratio = _log_density_ratio(mode_scale)
  if positive_sample:
    privacy_loss = positive_privacy_loss
    log_weights = positive_privacy_loss - proposal_log_density_ratio
  else:
    privacy_loss = -positive_privacy_loss
    log_weights = -proposal_log_density_ratio
  return delta_calculation.PrivacyLossSamples(
      privacy_loss, log_weights=log_weights, support_bound=np.exp(log_bound)
  )

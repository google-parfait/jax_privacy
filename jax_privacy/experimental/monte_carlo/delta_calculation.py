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

"""Utilities for calculating deltas in Monte Carlo accounting."""

import dataclasses
from typing import Sequence

import dp_accounting
import numpy as np
import numpy.typing as npt
import scipy

from ... import _validate

DpEventOrAccountant = (
    dp_accounting.dp_event.DpEvent | dp_accounting.pld.PLDAccountant
)


@dataclasses.dataclass(frozen=True)
class PrivacyLossSamples:
  r"""Samples from a privacy loss distribution.

  Attributes:
    privacy_loss: 1D array of privacy loss samples. That is, to estimate the
      :math:`e^\varepsilon` hockey-stick divergence between :math:`P` and
      :math:`Q`, we sample :math:`x` from :math:`P` and record
      :math:`\ln(P(x)/Q(x))`.
    counts: Optional counts of each sample, of the same shape as
      ``privacy_loss``. This can be helpful if the number of samples is very
      large, in which case one can discretize the samples and pass the counts of
      each discretization. If ``None``, each sample has count 1.
    log_weights: Optional importance weights in log space, of the same shape as
      ``privacy_loss``, for samples :math:`x` drawn from a proposal distribution
      :math:`R` instead of :math:`P`. Passing :math:`\ln(P(x)/R(x))` for each
      sample keeps the estimate unbiased. Cannot be combined with ``counts``:
      the weight depends on :math:`x` and not only on its privacy loss, so
      discretizing by privacy loss would bias the estimate.
    support_bound: An upper bound on each (weighted) sample of the hockey-stick
      divergence, see :func:`get_overall_delta`. Plain Monte Carlo samples are
      bounded by 1.
  """

  privacy_loss: npt.ArrayLike
  counts: npt.ArrayLike | None = None
  log_weights: npt.ArrayLike | None = None
  support_bound: float = 1.0

  def __post_init__(self):
    shape = np.shape(self.privacy_loss)
    if len(shape) != 1:
      raise ValueError('privacy_loss must be a 1D array.')
    for name in ['counts', 'log_weights']:
      value = getattr(self, name)
      if value is not None and np.shape(value) != shape:
        raise ValueError(f'{name} must have the same shape as privacy_loss.')
    if self.counts is not None and self.log_weights is not None:
      raise ValueError(
          'counts cannot be combined with log_weights: discretizing'
          ' importance-weighted samples is not supported.'
      )

  @property
  def num_samples(self) -> float:
    """The total count of samples."""
    if self.counts is None:
      return np.size(self.privacy_loss)
    return np.sum(self.counts)


def _convert_to_accountant(
    dp_events: DpEventOrAccountant,
) -> dp_accounting.pld.PLDAccountant:
  """Converts DpEventOrAccountant to a PLDAccountant."""
  if isinstance(dp_events, dp_accounting.dp_event.DpEvent):
    return dp_accounting.pld.PLDAccountant().compose(dp_events)
  elif isinstance(dp_events, dp_accounting.pld.PLDAccountant):
    return dp_events
  else:
    raise ValueError('Unsupported DpEventOrAccountant type.')


def _kl(q: npt.ArrayLike, p: npt.ArrayLike) -> np.ndarray:
  """KL divergence between Bernoulli(q) and Bernoulli(p), elementwise."""
  q, p = np.asarray(q, dtype=float), np.asarray(p, dtype=float)
  # The two terms nearly cancel when q and p are both small, so the second is
  # computed with log1p to retain full relative precision.
  with np.errstate(divide='ignore', invalid='ignore'):
    kl = q * np.log(q / p) + (1 - q) * (np.log1p(-q) - np.log1p(-p))
  return np.where(q == p, 0.0, kl)


def _hoeffding_bound(
    num_samples: npt.ArrayLike,
    tau: float,
    delta: float,
    support_bound: npt.ArrayLike = 1.0,
) -> np.ndarray:
  """Bound on prob that sample mean <= delta if true mean > tau * delta.

  Fact 4.2 of https://arxiv.org/pdf/2412.16802, applied to samples in
  ``[0, support_bound]`` by rescaling them to ``[0, 1]``.

  Args:
    num_samples: The number of i.i.d. samples.
    tau: The ratio of the true mean to ``delta``, at least 1.
    delta: The threshold on the sample mean.
    support_bound: An upper bound on the samples.

  Returns:
    The KL-Hoeffding bound on the probability, elementwise in ``num_samples``
    and ``support_bound``.
  """
  _validate.in_range(1, np.inf, tau=tau)
  rescaled_delta = delta / np.asarray(support_bound)
  # A true mean above support_bound is impossible, so where tau * delta exceeds
  # it the bound is 0 (the KL is infinite).
  kl = _kl(rescaled_delta, np.minimum(tau * rescaled_delta, 1.0))
  return np.exp(-np.asarray(num_samples) * kl)


def _validate_support_bound(support_bound: npt.ArrayLike, delta: float) -> None:
  """Checks that every support_bound is in (delta, 1]."""
  if np.max(support_bound) > 1:
    raise ValueError(
        'support_bound must be at most 1. If this came from importance'
        ' sampling, the proposal is worse than plain Monte Carlo; use'
        ' support_bound=1 and unweighted samples instead.'
    )
  if np.min(support_bound) <= delta:
    raise ValueError('support_bound must be greater than delta.')


def _validate_num_samples(num_samples: npt.ArrayLike) -> None:
  if np.min(num_samples) <= 0:
    raise ValueError(f'Expected num_samples={num_samples} > 0.')


def get_overall_delta(
    num_samples: npt.ArrayLike,
    base_delta: float,
    support_bound: npt.ArrayLike = 1.0,
) -> float:
  """The delta we can formally report using Monte Carlo verification.

  In more detail, if we use num_samples samples to verify that a mechanism
  satisfies (epsilon, base_delta)-DP, then we can formally report that the
  mechanism satisfies (epsilon, overall_delta)-DP.

  This method assumes we are doing many Monte Carlo verifications of different
  ordered mechanisms, and then picking the highest-utility mechanism that meets
  our target privacy guarantee, as in Appendix A of
  https://arxiv.org/abs/2602.09338. If we are only verifying one mechanism
  (which is undesirable in practice), a tighter bound is possible.

  The verifications may use different numbers of samples and support bounds,
  passed as arrays with one entry per verification. Each verification is then
  wrong with a probability computed from its own number of samples and support
  bound, while the privacy loss after a wrong verification is bounded by the
  largest support bound.

  Args:
    num_samples: The number of samples used in each Monte Carlo verification.
    base_delta: The base_delta such that we use the samples to verify that each
      mechanism satisfies (epsilon, base_delta)-DP.
    support_bound: An upper bound on each (weighted) sample of the hockey-stick
      divergence, in (base_delta, 1], for each verification. This is 1 for
      standard Monte Carlo estimation, and can be much smaller with importance
      sampling, in which case fewer samples are needed. Since the true delta is
      the mean of the samples, it is also at most support_bound, so for a single
      support bound the result equals ``support_bound *
      get_overall_delta(num_samples, base_delta / support_bound)``.

  Returns:
    overall_delta such that we can report the end-to-end pipeline of Monte Carlo
    verification and then running the best verified mechanism is (epsilon,
    overall_delta)-DP.
  """
  _validate.in_range(0, 1, closed='right', base_delta=base_delta)
  _validate_num_samples(num_samples)
  _validate_support_bound(support_bound, base_delta)
  max_support_bound = float(np.max(support_bound))

  def overall_delta_from_tau(tau):
    # With probability at most q the first verification that should fail
    # passes instead (so q is a maximum, not a union bound), in which case we
    # only know the true delta is at most the largest support bound.
    q = np.max(_hoeffding_bound(num_samples, tau, base_delta, support_bound))
    return tau * base_delta + q * (max_support_bound - tau * base_delta)

  best_tau = scipy.optimize.minimize_scalar(
      overall_delta_from_tau,
      bounds=(1, max_support_bound / base_delta),
      method='bounded',
  ).x
  return min(overall_delta_from_tau(best_tau), max_support_bound)


def minimum_samples_to_calibrate(
    base_delta: float, target_delta: float, support_bound: float = 1.0
) -> int:
  """The minimum number of samples needed to calibrate to target_delta.

  Args:
    base_delta: The base_delta such that we use Monte Carlo verification to
      check if each mechanism satisfies (epsilon, base_delta)-DP.
    target_delta: The value such that we want to report the overall mechanism of
      Monte Carlo verification and then running the best verified mechanism is
      (epsilon, target_delta)-DP.
    support_bound: An upper bound on each (weighted) sample of the hockey-stick
      divergence. See :func:`get_overall_delta`.

  Returns:
    The minimum number of samples needed to calibrate to target_delta.
  """
  num_samples = minimum_samples_per_verification(
      base_delta, target_delta, [support_bound]
  )
  return int(num_samples[0])


def minimum_samples_per_verification(
    base_delta: float, target_delta: float, support_bounds: Sequence[float]
) -> np.ndarray:
  """The numbers of samples per verification needed to calibrate.

  This is :func:`minimum_samples_to_calibrate` for verifications with different
  support bounds: the returned numbers of samples minimize their total subject
  to :func:`get_overall_delta` being at most ``target_delta``.

  Args:
    base_delta: The base_delta such that we use Monte Carlo verification to
      check if each mechanism satisfies (epsilon, base_delta)-DP.
    target_delta: The value such that we want to report the overall mechanism of
      Monte Carlo verification and then running the best verified mechanism is
      (epsilon, target_delta)-DP.
    support_bounds: An upper bound on each (weighted) sample of the hockey-stick
      divergence, for each verification. See :func:`get_overall_delta`.

  Returns:
    The number of samples for each verification.
  """
  _validate.in_range(
      0, 1, closed='right', base_delta=base_delta, target_delta=target_delta
  )
  if target_delta <= base_delta:
    raise ValueError('target_delta must be > base_delta.')
  _validate_support_bound(support_bounds, target_delta)
  support_bounds = np.asarray(support_bounds, dtype=float)
  max_support_bound = np.max(support_bounds)

  def samples_from_tau(tau):
    # Solving the bound of get_overall_delta for the largest acceptable
    # probability q of a wrong verification, each verification needs
    # ln(1 / q) / KL samples for the KL-Hoeffding bound to reach q.
    threshold = tau * base_delta
    q = (target_delta - threshold) / (max_support_bound - threshold)
    rescaled_delta = base_delta / support_bounds
    kl = _kl(rescaled_delta, np.minimum(tau * rescaled_delta, 1.0))
    with np.errstate(divide='ignore'):
      return -np.log(q) / kl

  # The sample counts blow up at both ends: at tau = 1 the KL vanishes and at
  # tau = target_delta / base_delta no wrong verification can be tolerated.
  best_tau = scipy.optimize.minimize_scalar(
      lambda tau: np.sum(samples_from_tau(tau)),
      bounds=(1, target_delta / base_delta),
      method='bounded',
  ).x
  # A small margin absorbs the numerical optimization here and in
  # get_overall_delta; the loop is a safeguard that should not trigger.
  margin = 1 + 1e-6
  num_samples = np.ceil(samples_from_tau(best_tau) * margin).astype(int)
  while (
      get_overall_delta(num_samples, base_delta, support_bounds) > target_delta
  ):
    num_samples = np.ceil(num_samples * margin).astype(int)
  return num_samples


def get_base_delta(
    num_samples: npt.ArrayLike,
    target_delta: float,
    support_bound: npt.ArrayLike = 1.0,
) -> float:
  """The base_delta for Monte Carlo verification to achieve target_delta.

  In more detail, if we use num_samples samples to verify that a mechanism
  satisfies (epsilon, base_delta)-DP, then we can formally report that the
  mechanism satisfies (epsilon, target_delta)-DP.

  This method assumes we are doing many Monte Carlo verifications of different
  ordered mechanisms, and then picking the highest-utility mechanism that meets
  our target privacy guarantee. If we are only verifying one mechanism (which
  requires some chance of an empty output which is usually undesirable in
  practice), a tighter bound is possible.

  Args:
    num_samples: The number of samples used in each Monte Carlo verification.
    target_delta: The value such that we want to report the overall mechanism of
      Monte Carlo verification and then running the best verified mechanism is
      (epsilon, target_delta)-DP.
    support_bound: An upper bound on each (weighted) sample of the hockey-stick
      divergence, for each verification. See :func:`get_overall_delta`.

  Returns:
    The base_delta such that we use Monte Carlo verification to check if each
    mechanism satisfies (epsilon, base_delta)-DP.
  """
  _validate_num_samples(num_samples)
  _validate.in_range(0, 1, closed='right', target_delta=target_delta)
  _validate_support_bound(support_bound, target_delta)

  def _excess_delta(base_delta):
    return (
        get_overall_delta(num_samples, base_delta, support_bound) - target_delta
    )

  # overall_delta is increasing in base_delta and exceeds target_delta at
  # base_delta = target_delta, so a valid base_delta exists iff the excess is
  # negative for some smaller base_delta.
  lower = target_delta
  while _excess_delta(lower) >= 0:
    lower *= 1e-3
    if lower < 1e-30 * target_delta:
      raise ValueError(
          'Failed to find a valid base_delta. num_samples may be too small.'
      )
  xtol = 1e-6 * target_delta
  base_delta = scipy.optimize.brentq(
      _excess_delta, lower, target_delta, xtol=xtol
  )
  # The root is bracketed to within xtol; step down until conservatively valid.
  while _excess_delta(base_delta) > 0:
    base_delta -= xtol
  return base_delta


def delta_from_epsilon_and_samples(
    epsilon: float,
    samples: PrivacyLossSamples,
    other_event: DpEventOrAccountant | None = None,
) -> float:
  """Calculate the delta parameter for a given epsilon and samples.

  Args:
    epsilon: The epsilon parameter of the DP guarantee.
    samples: Samples from the privacy loss distribution.
    other_event: An optional DpEvent or PrivacyAccountant object. If ``None``,
      we are just computing the delta parameter for the event whose samples are
      given by samples. If an accountant is given, the method returns the
      estimate for Monte Carlo accounting of the composition of the event whose
      samples are given by samples, and other_event. Cannot be combined with
      importance-weighted samples.

  Returns:
    The delta parameter given by Monte Carlo estimation of the hockey-stick
    divergence.
  """
  _validate.non_negative(epsilon=epsilon)
  if other_event is not None and samples.log_weights is not None:
    raise ValueError('other_event cannot be combined with log_weights.')
  privacy_loss = np.asarray(samples.privacy_loss)
  if other_event is None:
    estimands = -np.expm1(np.minimum(epsilon - privacy_loss, 0.0))
    if samples.log_weights is not None:
      # Combine in log space: a zero estimand with a huge weight is 0, not nan.
      with np.errstate(divide='ignore'):
        estimands = np.exp(np.log(estimands) + samples.log_weights)
  else:
    other_event = _convert_to_accountant(other_event)
    # TODO: Cite an external derivation for this formula.
    estimands = [other_event.get_delta(epsilon - s) for s in privacy_loss]
  return float(np.average(estimands, weights=samples.counts))


def perform_calibration_from_samples(
    epsilon: float,
    delta: float,
    *,
    positive_samples: Sequence[PrivacyLossSamples],
    negative_samples: Sequence[PrivacyLossSamples | None] | None = None,
    other_event: (
        DpEventOrAccountant | Sequence[DpEventOrAccountant] | None
    ) = None,
) -> tuple[bool, int | float]:
  r"""Perform calibration to find highest-utility parameter for DP target.

  This is Algorithm 5 in https://arxiv.org/pdf/2602.09338.

  In particular, this method assumes we have chosen a sweep of a hyperparameter
  (such as noise multiplier), for each value in the sweep independently
  generated some number of samples, and now want to find the highest-utility
  hyperparameter (e.g. the lowest noise multiplier or highest number of training
  iterations in DP-SGD) we can use and still report end-to-end
  (epsilon, delta)-DP using Monte Carlo verification.

  We assume that the samples are ordered from highest to lowest privacy of the
  hyperparameters, i.e. lowest to highest delta at a fixed epsilon
  (or vice-versa), even though the privacy parameters may not be explicitly
  known. e.g., in the example of calibrating the noise multiplier for
  DP-BandMF, we would order the samples from highest to lowest noise multiplier,
  even if we may not know the exact (epsilon, delta)-DP guarantee for each noise
  multiplier.

  If different numbers of samples were used for each hyperparameter, this method
  conservatively uses the minimum number of samples across all values of the
  hyperparameter for computing tail bounds. Likewise, if different support
  bounds apply to different hyperparameters, the maximum is used.

  Example Usage (calibrating a Gaussian mechanism):
    >>> epsilon, delta = 4.0, 1e-3
    >>> nm_sweep = [2.0, 1.0, 0.5]
    >>> positive_samples = []
    >>> for nm in nm_sweep:
    ...   positive_samples.append(PrivacyLossSamples(
    ...     np.random.normal(loc=0.5 / nm ** 2, size=1_000_000)
    ...   ))
    >>> passes_verification, best_nm_index = perform_calibration_from_samples(
    ...   epsilon, delta, positive_samples=positive_samples
    ... )
    >>> if passes_verification:
    ...   print(nm_sweep[best_nm_index])
    1.0

  Args:
    epsilon: The epsilon parameter of the target DP guarantee.
    delta: The delta parameter of the target DP guarantee.
    positive_samples: Privacy loss samples from the positive case, one
      :class:`PrivacyLossSamples` for each hyperparameter value.
    negative_samples: Optional privacy loss samples from the negative case, one
      for each hyperparameter value. If None, this means we assume the positive
      case has a worse DP guarantee than the negative case, so we only need to
      do accounting for the positive case. Individual entries may also be None,
      meaning the negative case is known (e.g. from an analytic bound) to have
      delta at most the base delta that the positive case is verified against,
      see :func:`get_base_delta`, so only the positive case is checked for it.
    other_event: Optional DpEvent or PLDAccountant or sequence thereof. For
      simplicity we don't accept Sequence[DpEvent], the ComposedDpEvent can be
      used instead. For brevity, we below refer to other_event as if it is a
      DpEvent, but if an accountant is passed the below logic is equivalent by
      substituting in the event represented by the accountant.  - If other_event
      is None, we are just calibrating the single event corresponding to the
      sequence of privacy loss samples. - If other_event is a single DpEvent, we
      are calibrating the composition of the mechanism whose privacy loss
      samples we computed and other_event, and are assuming other_event is fixed
      with respect to the parameter we are calibrating. - If other_event is a
      list of other_event objects, it must have the same length as
      positive_samples, i.e. one event for each hyperparameter value. In this
      case, we are again calibrating the composition of the mechanism whose
      privacy loss samples we computed, and other_event, but now we are assuming
      other_event is defined as a function of the parameter we are calibrating
      (and other_event[i] corresponds to the i-th hyperparameter value). For
      example, suppose we are calibrating sigma in {sigma_1, sigma_2} for the
      composition of a mechanism M(sigma) with a Gaussian mechanism with noise
      multiplier sigma. Then we should have ``positive_samples = [samples_1,
      samples_2]``, where ``samples_i`` is the privacy loss samples for
      M(sigma_i) (and ``samples_1`` is independent of ``samples_2``), and
      other_event is ``[GaussianDpEvent(sigma_1), GaussianDpEvent(sigma_2)]``.
      Cannot be combined with importance-weighted samples.

  Returns:
    Either ``(True, i)``, or ``(False, base_delta)``. If there is an associated
    set of hyperparameters and ``(True, i)`` is returned, ``i`` is the index of
    the highest-utility hyperparameter we can use while satisfying the target
    :math:`(\varepsilon, \delta)`-DP guarantee. If ``(False, base_delta)`` is
    returned, we should fall back to a mechanism that is known to be
    :math:`(\varepsilon, \text{base\_delta})`-DP (i.e., one that can be
    calibrated without Monte Carlo verification).
  """  # fmt: skip
  if not positive_samples:
    raise ValueError('positive_samples must be non-empty.')
  num_values = len(positive_samples)
  if negative_samples is None:
    negative_samples = [None] * num_values
  elif len(negative_samples) != num_values:
    raise ValueError(
        'positive_samples and negative_samples must have the same length.'
    )
  if other_event is None:
    other_events = [None] * num_values
  elif isinstance(other_event, Sequence):
    other_events = [_convert_to_accountant(e) for e in other_event]
  else:
    other_events = [_convert_to_accountant(other_event)] * num_values

  all_samples = [
      s for s in [*positive_samples, *negative_samples] if s is not None
  ]
  # TODO: Support a per-hyperparameter base_delta.
  base_delta = get_base_delta(
      [s.num_samples for s in all_samples],
      delta,
      [s.support_bound for s in all_samples],
  )

  def _passes(i: int) -> bool:
    positive_delta = delta_from_epsilon_and_samples(
        epsilon, positive_samples[i], other_events[i]
    )
    if positive_delta > base_delta:
      return False
    if negative_samples[i] is None:
      return True
    negative_delta = delta_from_epsilon_and_samples(
        epsilon, negative_samples[i], other_events[i]
    )
    return negative_delta <= base_delta

  for i in range(num_values):
    if not _passes(i):
      # This hyperparameter does not pass verification, return the previous one
      # (or fall back to base_delta if even the first one fails).
      return (True, i - 1) if i > 0 else (False, base_delta)
  return True, num_values - 1

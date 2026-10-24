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

"""Uses Monte Carlo accounting to calibrate noise multiplier for DP-SGD.

We use a small training setup for the purpose of making this example easy to
run. With larger setups, sample generation would be slower and we would need
more samples. In such cases, we recommend parallelizing sample generation. It
may also be good to discretize the samples by rounding up to the nearest
multiplier of some float and store the discretized samples as a histogram, since
this reduces the memory overhead.

Samples are drawn with importance sampling, which bounds each weighted sample by
a constant B < 1 and so reduces the number of samples needed by roughly a factor
of B. B depends on the noise multiplier (through the ratio of the largest mode
norm to the noise scale), and for calibration, we conservatively use the largest
B across the sweep.
"""

import math

from absl import app
import dp_accounting
from jax_privacy import accounting
from jax_privacy import batch_selection
from jax_privacy.experimental.monte_carlo import delta_calculation
from jax_privacy.experimental.monte_carlo import importance_sampling
import numpy as np


ITERATIONS = 100
EPOCH_LENGTH = 10
EPSILON = 1.0
DELTA = 1e-2
BASE_DELTA = DELTA / 2  # Make this closer to DELTA for more accuracy, further
# from DELTA for more speed.

# The non-zero entries of the first column of the banded Toeplitz matrix C
# used in DP-MF. If this vector is longer than EPOCH_LENGTH, the calculation of
# nm_upper_bound needs to be updated. Use [1.0] for DP-SGD.
C_COL = np.array([1.0, 1 / 2, 1 / 4, 1 / 8])
C_COL = C_COL / np.linalg.norm(C_COL)


def main(_) -> None:
  # We figure out a lower and upper bound on the noise multiplier necessary to
  # achieve (EPSILON, BASE_DELTA)-DP, and then define a sweep between these
  # bounds. Note that this is calibrating to BASE_DELTA, not DELTA.

  # For the lower bound, we use the noise multiplier required for DP-SGD with
  # Poisson sampling. This is a reasonable lower bound for DP-MF since DP-MF
  # generally requires more noise than DP-SGD.
  nm_lower_bound = dp_accounting.calibrate_dp_mechanism(
      dp_accounting.pld.PLDAccountant,
      lambda nm: accounting.dpsgd_event(
          noise_multiplier=nm,
          iterations=ITERATIONS,
          sampling_prob=1 / EPOCH_LENGTH,
      ),
      EPSILON,
      BASE_DELTA,
  )

  # For the upper bound, we calculate the needed noise multiplier if we assume
  # no amplification, in which case DP-SGD is just a Gaussian mechanism with
  # sensitivity math.ceil(ITERATIONS / EPOCH_LENGTH) ** 0.5.
  nm_upper_bound = (
      dp_accounting.calibrate_dp_mechanism(
          dp_accounting.pld.PLDAccountant,
          dp_accounting.GaussianDpEvent,
          EPSILON,
          BASE_DELTA,
      )
      * math.ceil(ITERATIONS / EPOCH_LENGTH) ** 0.5
  )

  print(f'Sweeping noise multiplier from {nm_lower_bound} to {nm_upper_bound}')

  sweep_size = math.ceil(np.log(nm_upper_bound / nm_lower_bound) / np.log(1.1))
  # delta_calculation's calibration function assumes that the parameters are
  # ordered from highest to lowest privacy, so we create a sweep in that order.
  # We know nm_upper_bound is a valid noise multiplier, so we don't need to
  # include it in the sweep.
  nm_sweep = [nm_upper_bound / 1.1**i for i in range(1, sweep_size + 1)]

  strategy = batch_selection.BallsInBinsSampling(
      cycle_length=EPOCH_LENGTH, iterations=ITERATIONS
  )

  # The optimal stretch factor and resulting bound on the weighted samples for
  # each noise multiplier in the sweep.
  kappas = [
      importance_sampling.compute_kappa(strategy, nm, C_COL) for nm in nm_sweep
  ]
  alphas = [importance_sampling.optimal_stretch(EPSILON, k) for k in kappas]
  support_bound = max(
      math.exp(importance_sampling.log_support_bound(alpha, EPSILON, kappa))
      for alpha, kappa in zip(alphas, kappas)
  )
  minimum_samples = delta_calculation.minimum_samples_to_calibrate(
      BASE_DELTA, DELTA, support_bound
  )
  print(
      f'Minimum samples to calibrate: {minimum_samples} (importance sampling,'
      f' support bound {support_bound:.3f}) vs'
      f' {delta_calculation.minimum_samples_to_calibrate(BASE_DELTA, DELTA)}'
      ' (plain Monte Carlo)'
  )

  # These calls are massively parallelizable! In addition, for larger numbers
  # of samples, we need multiple calls to get_importance_sampled_privacy_loss
  # with smaller values of num_samples to avoid out of memory errors. These
  # calls can also be massively parallelized. For practical applications, we
  # recommend parallelizing across these two dimensions in whatever manner best
  # fits your workflow.
  def _sample(nm, alpha, positive_sample):
    return importance_sampling.get_importance_sampled_privacy_loss(
        strategy=strategy,
        noise_multiplier=nm,
        c_col=C_COL,
        epsilon=EPSILON,
        alpha=alpha,
        positive_sample=positive_sample,
        num_samples=minimum_samples,
    )

  positive_samples = [_sample(nm, a, True) for nm, a in zip(nm_sweep, alphas)]
  negative_samples = [_sample(nm, a, False) for nm, a in zip(nm_sweep, alphas)]

  passes_verification, best_nm_index = (
      delta_calculation.perform_calibration_from_samples(
          EPSILON,
          DELTA,
          positive_samples=[s.privacy_loss for s in positive_samples],
          negative_samples=[s.privacy_loss for s in negative_samples],
          positive_log_weights=[s.log_weights for s in positive_samples],
          negative_log_weights=[s.log_weights for s in negative_samples],
          support_bound=support_bound,
      )
  )
  if passes_verification:
    # pyrefly: ignore[bad-index]
    print(f'Best noise multiplier: {nm_sweep[best_nm_index]}')
  else:
    # If all the verifications fail, we fall back to nm_upper_bound, which we
    # know achieves (EPSILON, BASE_DELTA)-DP.
    print(f'Best noise multiplier: {nm_upper_bound}')


if __name__ == '__main__':
  app.run(main)

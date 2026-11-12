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

"""Tests for reproducible synthetic logistic-regression benchmarks."""

from absl.testing import absltest
from examples import dp_logistic_regression
import jax
import numpy as np


class LogisticRegressionBenchmarkTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.addCleanup(np.random.set_state, np.random.get_state())

  def test_same_seed_is_independent_of_global_numpy_random_state(self):
    np.random.seed(1)
    first = dp_logistic_regression.create_benchmark(200, 5, seed=17)
    np.random.seed(2)
    second = dp_logistic_regression.create_benchmark(200, 5, seed=17)

    for first_leaf, second_leaf in zip(
        jax.tree.leaves(first), jax.tree.leaves(second)
    ):
      np.testing.assert_array_equal(first_leaf, second_leaf)

  def test_benchmark_does_not_advance_global_numpy_random_state(self):
    before = np.random.get_state()
    dp_logistic_regression.create_benchmark(200, 5, seed=17)
    after = np.random.get_state()

    self.assertEqual(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])
    self.assertEqual(before[2:], after[2:])

  def test_different_seeds_change_labels(self):
    _, _, first_labels = dp_logistic_regression.create_benchmark(
        200, 5, seed=17
    )
    _, _, second_labels = dp_logistic_regression.create_benchmark(
        200, 5, seed=18
    )

    self.assertFalse(np.array_equal(first_labels, second_labels))


if __name__ == '__main__':
  absltest.main()

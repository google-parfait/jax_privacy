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

from absl.testing import absltest
from absl.testing import parameterized
from jax_privacy import _validate


class InRangeTest(parameterized.TestCase):

  @parameterized.product(
      closed=['both', 'left', 'right', 'neither'], value=[0.0, 0.5, 1.0]
  )
  def test_endpoints(self, closed, value):
    lo_ok = value > 0 or closed in ('both', 'left')
    hi_ok = value < 1 or closed in ('both', 'right')
    if lo_ok and hi_ok:
      _validate.in_range(0, 1, closed=closed, x=value)
    else:
      with self.assertRaisesRegex(ValueError, 'Expected x='):
        _validate.in_range(0, 1, closed=closed, x=value)

  @parameterized.parameters(
      ('both', r'\[0, 1\]'),
      ('left', r'\[0, 1\)'),
      ('right', r'\(0, 1\]'),
      ('neither', r'\(0, 1\)'),
  )
  def test_error_message_shows_interval(self, closed, interval):
    with self.assertRaisesRegex(ValueError, interval):
      _validate.in_range(0, 1, closed=closed, x=2.0)

  def test_rejects_unknown_closed(self):
    with self.assertRaises(ValueError):
      _validate.in_range(0, 1, closed='top', x=0.5)

  def test_validates_all_values(self):
    with self.assertRaisesRegex(ValueError, 'Expected y='):
      _validate.in_range(0, 1, x=0.5, y=1.5)


class PositiveTest(parameterized.TestCase):

  @parameterized.parameters(1, 0.5, 1e-12)
  def test_accepts_finite_positive(self, value):
    _validate.positive(x=value)

  @parameterized.parameters(0, -1.0, float('nan'), float('inf'))
  def test_rejects_non_positive_or_non_finite(self, value):
    with self.assertRaisesRegex(ValueError, r'Expected x=.* > 0'):
      _validate.positive(x=value)


if __name__ == '__main__':
  absltest.main()

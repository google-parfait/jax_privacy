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

"""Tests for the data-loading example's unpadded Grain iterator."""

import contextlib
import importlib.util
import io
import pathlib
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np


_SPEC = importlib.util.spec_from_file_location(
    'data_loading',
    pathlib.Path(__file__).resolve().parents[1]
    / 'examples'
    / 'data_loading.py',
)
assert _SPEC is not None and _SPEC.loader is not None
data_loading = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(data_loading)


class DataLoadingExampleTest(parameterized.TestCase):

  @parameterized.parameters(2, 4)
  def test_plain_iterator_reports_all_examples_as_real(self, batch_size):
    records = [
        {'image': np.zeros((28, 28, 1), dtype=np.uint8), 'label': i % 10}
        for i in range(2 * batch_size)
    ]
    output = io.StringIO()
    with (
        flagsaver.as_parsed(
            batch_size=str(batch_size),
            num_batches='2',
            use_custom_iterator='false',
            max_workers='1',
        ),
        mock.patch.object(
            data_loading.tfds, 'data_source', return_value=records
        ),
        contextlib.redirect_stdout(output),
    ):
      data_loading.main(None)

    self.assertIn('Real Example Fraction: 1.0000', output.getvalue())


if __name__ == '__main__':
  absltest.main()

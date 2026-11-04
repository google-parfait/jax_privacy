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

"""Algorithms for Privacy-Preserving Machine Learning in JAX."""

from . import accounting
from . import auditing
from . import batch_selection
from . import execution_plan
from . import experimental
from . import matrix_factorization
from . import noise_addition
from . import optimizers
from . import training

# pylint: disable=g-importing-member
# Carefully selected member imports for the top-level public API.
from .clipping import clipped_grad
from .execution_plan import BandMFConfig
from .execution_plan import PerformanceFlags
from .training import DPTrainer

__version__ = '2.3.0.dev0'

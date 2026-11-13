.. Copyright 2026 DeepMind Technologies Limited.
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

###########################
 JAX Privacy documentation
###########################

.. include:: introduction.md
   :parser: myst_parser.sphinx_

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   overview
   installation

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   core_library

.. toctree::
   :maxdepth: 2
   :caption: Examples

   keras_api
   examples_guide
   _collections/examples/dp_sgd_flax_linen_mnist
   _collections/examples/dp_sgd_keras_gemma3_lora_finetuning_samsum
   _collections/examples/dp_sgd_keras_gemma3_synthetic_data

.. toctree::
   :maxdepth: 2
   :caption: Sharp Edges

   sharp_edges_dp_training_pitfalls
   sharp_edges_variable_batch_sizes
   sharp_edges_vmap_sharding
   sharp_edges_mixed_precision

.. toctree::
   :maxdepth: 2
   :caption: Technical Documentation

   history
   paper_reproductions
   batch_selection
   matrix_factorization
   troubleshooting
   library_design
   contribution_guide
   support

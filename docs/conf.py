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

"""Configuration file for the Sphinx documentation builder.

This file contains all the configuration settings for Sphinx to generate
the documentation for the jax_privacy library.
"""

# sphinx-build -b html . _build/html

import os
import sys
from typing import Protocol
from unittest import mock


class MockDataclassInstance(Protocol):
  pass


class MockTypeshed(mock.MagicMock):

  def __getattr__(self, name):
    if name == 'DataclassInstance':
      return MockDataclassInstance
    return super().__getattr__(name)


# Inject this module into sys.modules so Python finds it
sys.modules['_typeshed'] = MockTypeshed()

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'JAX Privacy'
copyright = '2026, Google DeepMind'  # pylint: disable=redefined-builtin
author = 'Google DeepMind'
release = '2.3.0.dev0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_book_theme',
    'myst_nb',
    'sphinx_collections',
    'sphinx.ext.doctest',
    'sphinx_autodoc_typehints',
]

# Enable MyST-Parser extensions for math rendering ($...$ and AMS environments).
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#dollar-delimited-math  # pylint: disable=line-too-long
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#math-jax-amsmath-environments  # pylint: disable=line-too-long
myst_enable_extensions = [
    'dollarmath',
    'amsmath',
]

autodoc_type_aliases = {
    'ArrayLike': 'jax.typing.ArrayLike',
    'ArrayTree': 'optax.ArrayTree',
}

autosummary_generate = True

# Configure autodoc settings
autodoc_typehints = 'signature'
autoclass_content = 'both'
autodoc_member_order = 'bysource'
napoleon_use_ivar = True

# -- Intersphinx configuration -----------------------------------------------
# Creates cross-reference links in pydocs rendered on readthedocs, for example
# :func:`jax.grad` will link to the JAX documentation page for jax.grad.
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
intersphinx_mapping = {
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'optax': ('https://optax.readthedocs.io/en/latest/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_theme_options = {
    'show_toc_level': 2,
    'repository_url': 'https://github.com/google-parfait/jax_privacy',
    'use_repository_button': True,
}

nb_execution_mode = 'off'
suppress_warnings = [
    'misc.highlighting_failure',
    'toc.not_included',
]

# We have to copy examples to include them in the docs. Without it won't work.
collections = {
    'examples': {
        'driver': 'copy_folder',
        'source': '../examples',  # Path from conf.py to your real examples
        'ignore': 'BUILD',
    }
}

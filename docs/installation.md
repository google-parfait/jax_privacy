<!-- Copyright 2026 DeepMind Technologies Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

# Installation

**Note:** to ensure that your installation is compatible with your local
accelerators such as a GPU, we recommend to first follow the corresponding
instructions to install [JAX](https://github.com/jax-ml/jax#installation).

## Recommended: Install from PyPI

We recommend installing the latest published release from PyPI:

```bash
pip install jax-privacy
```

## Alternative: Install from GitHub Head

If you need recent features or fixes that have not yet been included in a PyPI
release, you can install directly from GitHub head:

```bash
pip install git+https://github.com/google-parfait/jax_privacy
```

## Local Development Installation

This option is preferred if you want to build on top of our codebase or
contribute to the library.

1. Clone the repository:

   ```bash
   git clone https://github.com/google-parfait/jax_privacy
   ```

2. Install the package in editable mode so modifications to the code are
   reflected in imports:

   ```bash
   cd jax_privacy
   pip install -e .
   ```


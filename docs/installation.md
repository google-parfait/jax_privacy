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

## Recommended: Install from GitHub Head

We **highly recommend** installing JAX Privacy directly from GitHub head.
The repository is under active development, and installing from head ensures
you get the most recent features and the best version of the library.

```bash
pip install git+https://github.com/google-parfait/jax_privacy
```

### Dependency versions

* [**DP Accounting**](https://github.com/google/differential-privacy/tree/main/python/dp_accounting):
  JAX Privacy and DP Accounting are co-developed — we often add features to
  DP Accounting and then surface them in JAX Privacy, so installing from head
  ensures consistency.

* [**Optax**](https://github.com/google-deepmind/optax):
  Parts of JAX Privacy have been upstreamed to optax (optax.microbatching),
  and JAX Privacy depends on this recently-added code. Installing from head
  ensures updates to optax are reflected in JAX-Privacy.

## Alternative: Install from PyPI

You can also install JAX Privacy from PyPI (version 1.0 or 2.0), but note
that these releases will not have all the features available on GitHub head.

```bash
pip install jax-privacy
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


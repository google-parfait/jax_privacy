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

# Paper Reproductions Guide

The JAX-Privacy repository was initially released alongside the paper
[Unlocking High-Accuracy Differentially Private Image Classification through
Scale](https://arxiv.org/abs/2204.13650). To reproduce the experiments in this
paper exactly as they were run, you should clone an earlier version of the
repository. Instructions can be found at
[this README](https://github.com/google-parfait/jax_privacy/tree/0e9d93966452a9ca022b9af9080341b84035bfd4/jax_privacy/experiments/image_classification).

Moving forward, these experiments will no longer be maintained
for the following reasons:

## Changes to the JAX Ecosystem

We note that since JAX-Privacy has been released, the overall JAX ecosystem has
matured significantly, and some of the libraries and utilities used in this
experiment are no longer actively maintained or are considered deprecated, like
[jaxline](https://github.com/google-deepmind/jaxline),
[haiku](https://github.com/google-deepmind/dm-haiku),
and {func}`jax.pmap`.

We do not want to force or encourage users of JAX privacy to learn these
old libraries. We encourage users to refer to our small standalone
[examples](examples_guide), which are smaller and easier to understand
and use.

## Changes to JAX-Privacy maintainers

The maintainers of the library have changed since its initial release, and the
core library has undergone a major refactoring in v2.0 to simplify the APIs and
improve the user experience (see [History of JAX Privacy](history.md)). As the
experiments directory was far larger and more complex than the core library,
maintaining those files across the rewrite was impractical. We encourage the
community to develop end-to-end training pipelines on top of JAX Privacy
instead, taking inspiration from our [examples](examples_guide) and current best
practices.

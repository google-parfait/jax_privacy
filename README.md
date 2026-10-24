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

# JAX-Privacy: Algorithms for Privacy-Preserving Machine Learning in JAX

| [**Docs**](https://jax-privacy.readthedocs.io/) |
[**Overview**](https://jax-privacy.readthedocs.io/en/latest/overview.html) |
[**Paper**](https://arxiv.org/abs/2602.17861) | [**Citing**](#citing) |
[**Contact**](#contact)

JAX-Privacy is a library designed to simplify the deployment of robust and
performant mechanisms for differentially private (DP) machine learning in JAX.
Guided by design principles of usability, flexibility, and efficiency, it
provides modular, verified primitives alongside high-level training APIs for
both researchers requiring deep customization and practitioners seeking an
out-of-the-box experience.

For installation instructions, examples, and full API documentation, please
visit the [JAX Privacy documentation](https://jax-privacy.readthedocs.io/) or
read the [accompanying paper](https://arxiv.org/abs/2602.17861).

## Scope and Design Philosophy

JAX Privacy is a general-purpose Python library for differentially private
machine learning, with a focus on **DP-SGD-style mechanisms**. Rather than
implementing task-specific end-to-end pipelines, it provides both high-level
training APIs ([`training.py`][training-py] and the [Keras API][keras-doc])
and lower-level, composable building blocks that integrate into any JAX
training loop. For a detailed walkthrough of the five core building blocks and
the three API tiers, see the [Library Overview][overview-doc].

### What Is in Scope

*   **Composable DP-SGD Ingredients**: Pure Python/JAX modules for per-example
    gradient clipping, independent and correlated noise addition (including
    matrix factorization), batch selection strategies, privacy accounting, and
    empirical auditing.
*   **Modality- and Model-Agnostic Computation**: JAX Privacy works with any
    JAX model and dataset format. It does not need to know whether you are
    working with tabular data, sequence data, image data, or text; the only
    thing it cares about is that your inputs are PyTrees of JAX arrays with a
    leading batch axis.
*   **Zero Framework Lock-In**: JAX Privacy intentionally does not bring in any
    neural network or data-loading framework dependencies beyond core JAX (along
    with `optax` and `dp-accounting`), keeping it lightweight and compatible
    with any JAX ecosystem stack.

### What Is Out of Scope

*   **Models and Data Loaders**: Defining model architectures, tokenizers, and
    data loaders is out of scope for the library.
*   **Domain-Specific End-to-End Pipelines**: Task-specific workflows such as
    "DP Fine-Tuning of Language Models" or "DP Synthetic Data Generation" are
    not part of the core library, though users can easily use JAX Privacy to
    build them (see companion libraries like
    [DPSynth](https://github.com/google-parfait/dpsynth)).
*   **Binaries and Surrounding Infrastructure**: As a library, JAX Privacy
    consists purely of importable Python code and does not house standalone
    production binaries, job orchestration, or serving infrastructure. Runnable
    scripts and binaries demonstrating how to use the library live in a
    dedicated [`examples/`][examples-dir] directory.

[overview-doc]: https://jax-privacy.readthedocs.io/en/latest/overview.html
[keras-doc]: https://jax-privacy.readthedocs.io/en/latest/keras_api.html
[training-py]: https://github.com/google-parfait/jax_privacy/blob/main/jax_privacy/training.py
[examples-dir]: https://github.com/google-parfait/jax_privacy/tree/main/examples

## How to Cite This Repository <a id="citing"></a>

If you use JAX-Privacy in your work, please cite the accompanying paper:

```
@article{mckenna2026jaxprivacy,
  author = {McKenna, Ryan and Andrew, Galen and Balle, Borja and
Doroshenko, Vadym and Ganesh, Arun and Kong, Weiwei and Kurakin, Alex and
McMahan, Brendan and Pravilov, Mikhail},
  title = {{JAX}-{P}rivacy: A library for differentially private machine learning},
  journal = {arXiv preprint arXiv:2602.17861},
  url = {https://arxiv.org/abs/2602.17861},
  year = {2026},
}
```

To cite the software repository directly:

```
@software{jax-privacy2022github,
  author = {Balle, Borja and Berrada, Leonard and Charles, Zachary and
Choquette-Choo, Christopher A and De, Soham and Doroshenko, Vadym and Dvijotham,
Dj and Galen, Andrew and Ganesh, Arun and Ghalebikesabi, Sahra and Hayes, Jamie
and Kairouz, Peter and McKenna, Ryan and McMahan, Brendan and Pappu, Aneesh and
Ponomareva, Natalia and Pravilov, Mikhail and Rush, Keith and Smith, Samuel L
and Stanforth, Robert and Mishra, Chaitanya},
  title = {{JAX}-{P}rivacy: Algorithms for Privacy-Preserving Machine Learning in JAX},
  url = {http://github.com/google-parfait/jax_privacy},
  version = {2.3.0.dev0},
  year = {2026},
}
```

## Contact <a id="contact"></a>

If you have any questions or feedback, you can contact us via email:
jax-privacy-open-source@google.com.

## Acknowledgements

-   [NFNet codebase](https://github.com/google-deepmind/deepmind-research/tree/master/nfnets)
-   [DeepMind JAX Ecosystem](https://deepmind.google/blog/using-jax-to-accelerate-our-research/)

## License

All code is made available under the Apache 2.0 License. Model parameters are
made available under the Creative Commons Attribution 4.0 International (CC BY
4.0) License.

See https://creativecommons.org/licenses/by/4.0/legalcode for more details.

## Disclaimer

This is not an official Google product.


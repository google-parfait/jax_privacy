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

# Troubleshooting

If you don't find a solution for your problem here, look for similar
[issues on our GitHub](https://github.com/google-parfait/jax_privacy/issues).

This documentation is in active development and we are still gathering most
common pitfalls and issues that might arise during the use of our library. If
you have one, you are very welcome to contribute!

## Loss does not go down

Check whether it goes down with non-DP training. If it does not go down for
non-DP either, then try usual ML remediation techniques like changing learning
rate, using different optimizers. Such a problem might often happen if you use
16-bit mixed precision because more error is accumulated. In this case,
accumulating fewer gradients or adjusting precision might help (see the
[Mixed Precision Training guide](sharp_edges_mixed_precision)).

If it happens only for DP training, then it is worth noting that DP training in
general converges slower than usual training. In this case, try to do more
iterations, increase batch size (both physical and effective; see
[Handling Variable Batch Sizes](sharp_edges_variable_batch_sizes)), or
increase the learning rate. For a comprehensive overview of pitfalls that can
affect DP convergence and utility, consult
[Common Pitfalls in DP Training](sharp_edges_dp_training_pitfalls).

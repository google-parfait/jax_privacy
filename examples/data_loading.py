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

"""Demonstrates integration of jax_privacy.BatchSelectionStrategy <> PyGrain.

This file is meant to demonstrate how the BatchSelectionStrategy API can be
used with pygrain for efficient data loading from on-disk datasets. This example
can be forked for your own use cases, potentially with some customization, but
should work out-of-the-box as well.

By default, this data loader is configured to load all of the data into a single
JAX process. In multi-controller JAX setups, the default behavior means all
processes load all data. To distribute the data loading across multiple jax
processes, use the combination of shard_options=grain.ShardByJaxProcess() and
jax.make_array_from_process_local_data().
"""

import itertools
import time

from absl import app
from absl import flags
import grain.python as grain
from jax_privacy import _data_loading
from jax_privacy import batch_selection
import numpy as np
import tensorflow_datasets as tfds
import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 128, "Batch size for benchmarking.")
flags.DEFINE_integer("num_batches", 100, "Number of batches to benchmark.")
flags.DEFINE_bool("use_custom_iterator", True, "Use the custom iterator.")
flags.DEFINE_integer("max_workers", 8, "# workers to use for parallel loading.")


def main(_):
  max_workers = FLAGS.max_workers
  dataset = grain.MapDataset.source(
      tfds.data_source(
          "mnist",
          split="train",
          builder_kwargs={"file_format": "array_record"},
          data_dir=None,
      )
  )

  print(f"Starting benchmark with {FLAGS.batch_size=}, {FLAGS.num_batches=}...")

  if FLAGS.use_custom_iterator:
    strategy = batch_selection.CyclicPoissonSampling(
        iterations=FLAGS.num_batches,
        sampling_prob=FLAGS.batch_size / len(dataset),
    )
    iterator = _data_loading.batch_iterator(
        dataset,
        strategy,
        padding_example=dataset[0],
        rng=0,
        pad_to_multiple_of=32,
        max_workers=max_workers,
    )

  else:

    def map_fn(x):
      return x, np.zeros(FLAGS.batch_size, dtype=np.bool)

    options = grain.ReadOptions(num_threads=max_workers)
    iterator = itertools.islice(
        dataset.map(map_fn).to_iter_dataset(options).batch(FLAGS.batch_size),
        FLAGS.num_batches,
    )

  start_time = time.perf_counter()
  batch_sizes = set()
  real_examples = padded_examples = 0
  for batch, is_padding_example in tqdm.tqdm(iterator):
    del batch  # Unused.
    true_batch_size = (~is_padding_example).sum()
    padded_batch_size = is_padding_example.shape[0]
    real_examples += true_batch_size
    padded_examples += padded_batch_size
    batch_sizes.add(padded_batch_size)
  end_time = time.perf_counter()

  total_time = end_time - start_time
  batches_per_sec = FLAGS.num_batches / total_time
  elements_per_sec = (FLAGS.num_batches * FLAGS.batch_size) / total_time

  print("Benchmark results:")
  print(f"  Total time: {total_time:.4f} seconds")
  print(f"  Throughput: {batches_per_sec:.2f} batches/sec")
  print(f"  Throughput: {elements_per_sec:.2f} elements/sec")
  # Note: we do not always have to pay the price of padding examples. If using
  # microbatching, microbatches that contain all padding examples are skipped.
  # This metric is therefore an upper bound on the compute overhead of padding.
  print(f"  Real Example Fraction: {real_examples / padded_examples:.4f}")
  print(f"  Batch Sizes (Compilations): {sorted(batch_sizes)}")


if __name__ == "__main__":
  app.run(main)

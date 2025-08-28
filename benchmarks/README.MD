# Phase Folding Benchmarking Scripts

This folder contains scripts for benchmarking the phase folding optimization.

## Randomly Generating Circuits

The script `random_gen.py` is a tool for randomly generating circuits based
on a template, which parameters for the block length, rz weight (percentage of
rotations compared to other operations in subcircuits), and number of qubits.

A template should look mostly like a standard cudaq program, except that in lieu
of quantum operations, the following directives can be used to randomly generate
a circuit:

* `GEN-QALLOC: nqubits=<n>` will generate a `qvector` allocation with size `n`.
In addition, each of the qubits in the `qvector` will automatically be added to
a pool of qubits to be randomly selected from when generating operations.
* `GEN-BLOCK` will generate a sequence of random operations with the length
determined by the `--block-length` CLI option.
* `GEN-SUBCIRCUIT` will generate a sequence of random valid subcircuit operations,
(i.e., NOT, CNOT, Swap, and rz) with the length determined by the `--block-length`
CLI option.
* `GEN-MEASURES` will insert measurements for all qubits in the active pool.
* `GEN:<nqubits>` will be replaced with the number of quibts in the active pool.
* `GEN:<qubit>` will be replaced with a random qubit from the active pool.
* `GEN:<qvector>` will be replaced with a random `qvector` from the active pool.

The active pool of `qvector`s/qubits is tracked via curly braces, so it is important
to always use curly braces for blocks, even when they just have a single instruction.

## Benchmarking With Randomly Generating Subcircuits

The script `run_phase_folding_benchmarking.py` will randomly generate and benchmark
a suite of kernels based on `simple.template` and sets of configuration options
for each of the parameters for randomly generating a script: block length, rz
weight, and number of qubits. This script will construct multiple random
benchmarks with different random seeds for each configuration, and run each such
benchmark several times, including a warmup run to mitigate noise from caching.

For example, the following will randomly generate templates with block lengths of
100 and 200 (`--block-lengths 100 200`), rotation operation weights of 0.5 and 0.6
(`--rz-weights 0.5 0.6`), and 5 and 10 qubits each (`--n-qubits 5 10`), and 2 seeds
per configuration (`--n-seeds=2`) for a total of 16 different benchmarks. Each
benchmark will be run once (`--iterations=1`). Results per run will be dumped to
`raw.csv` (`--raw-data-file=raw.csv`, by default results per run won't be kept).
Summary results per configuration are dumped to `results.csv` by default, but this
can be overridden by providing a results file name.

```
python3 run_opt_benchmarks.py --block-lengths 100 200 --rz-weights 0.5 0.6 \
    --n-qubits 5 10 --seed 489 --n-seeds=2 --iterations=1 --raw-data-file=raw.csv
```

## Running Benchmarks in a Docker Contaienr

Finally, `run_phase_folding_benchmarking.sh` provides an example of how to run
the above scripts in the cuda-quantum development container and extract the
results.

# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import sys
import subprocess
import tempfile
import re
import argparse
from random_gen import generate_program
from scipy import stats

log_file = None


def log(command: str):
    log_file.write((command + "\n").encode())
    log_file.flush()


class Options:
    seed = 0
    block_length = 100
    rz_weight = 0.5
    n_qubits = 5

    def __str__(self):
        return "--seed={} --block-length={} --rz-weight={} --n-qubits={}".format(
            self.seed, self.block_length, self.rz_weight, self.n_qubits)


# Generate and compile benchmark executable from a template
def prepare_benchmark(template, outfile, options):
    program = generate_program(template, options.seed, options.rz_weight,
                               [options.block_length], [options.n_qubits])
    log("python3 random_gen.py " + template + " " + str(options))
    tmp_filename = template + ".cpp"
    with open(tmp_filename, 'w') as tmp_file:
        tmp_file.write(program)
    subprocess.run(
        ["nvq++", "--target=remote-mqpu", tmp_filename, "-o", outfile],
        check=True)
    log("nvq++  --target=remote-mqpu {} -o {}".format(tmp_filename, outfile))


# Run a single benchmark, capture time info
def run_benchmark(executable):

    def parse_launch_time(output):
        out = output.stdout
        #print(out)
        match = re.search('Total time: (.*)ms', out)
        launch_time = float(match.groups()[0])
        return launch_time

    def parse_JIT_time(output):
        out = output.stdout
        #print(out)
        match = re.search('JIT time: (.*)ms', out)
        jit_time = float(match.groups()[0])
        return jit_time

    def run_command(command):
        log(command)
        output = subprocess.run(command,
                                shell=True,
                                capture_output=True,
                                check=True,
                                text=True)
        return [parse_launch_time(output), parse_JIT_time(output)]

    [launch_time,
     jit_time] = run_command("CUDAQ_TIMING_TAGS=3,6 ./" + executable)
    [opt_launch_time, opt_jit_time
    ] = run_command("CUDAQ_TIMING_TAGS=3,6 CUDAQ_PHASE_FOLDING=true ./" +
                    executable)
    log("Launch time: {}, with opt: {}. JIT time: {}, with opt: {}".format(
        launch_time, opt_launch_time, jit_time, opt_jit_time))
    return [launch_time, jit_time, opt_launch_time, opt_jit_time]


# Generate and run benchmark based on template using options
# Runs with multiple random seeds to reduce the risk of outlier seeds
def benchmark(template,
              result_file,
              options,
              seeds=range(0, 50, 10),
              iterations=5):
    launch_times = []
    opt_launch_times = []
    jit_times = []
    opt_jit_times = []
    for seed in seeds:
        exec_file = "tmp.x"
        options.seed = seed
        prepare_benchmark(template, exec_file, options)
        # Do a warmup run to prepare the cache, etc...
        run_benchmark(exec_file)
        for _ in range(0, iterations + 1):
            [launch_time, jit_time, opt_launch_time,
             opt_jit_time] = run_benchmark(exec_file)
            launch_times.append(launch_time)
            jit_times.append(jit_time)
            opt_launch_times.append(opt_launch_time)
            opt_jit_times.append(opt_jit_time)
    launch_mean = stats.gmean(launch_times)
    launch_sem = stats.sem(launch_times)
    opt_launch_mean = stats.gmean(opt_launch_times)
    opt_launch_sem = stats.sem(opt_launch_times)
    jit_mean = stats.gmean(jit_times)
    jit_sem = stats.sem(jit_times)
    opt_jit_mean = stats.gmean(opt_jit_times)
    opt_jit_sem = stats.sem(opt_jit_times)
    data = [
        template, options.block_length, options.rz_weight, options.n_qubits,
        launch_mean, launch_sem, jit_mean, jit_sem, opt_launch_mean,
        opt_launch_sem, opt_jit_mean, opt_jit_sem
    ]
    result_file.write(",".join([str(x) for x in data]) + "\n")
    result_file.flush()


argparser = argparse.ArgumentParser(
    prog='Optimization Benchmarker',
    description=
    'Configurable benchmarking of optimization using randomly generated kernels',
    epilog='')
argparser.add_argument('--result-file', type=str, default="results.csv")
argparser.add_argument('--block-lengths', nargs="+", type=int)
argparser.add_argument('--rz-weights', nargs="+", type=float)
argparser.add_argument('--n-qubits', nargs="+", type=int)
argparser.add_argument('--seeds', nargs="+", type=int, default=[0])
argparser.add_argument('--iterations', type=int, default=5)

if __name__ == '__main__':
    log_file = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
    log(" ".join(sys.argv))
    args = argparser.parse_args()
    print("Logging to " + log_file.name)
    print("Outputting to " + args.result_file)
    with open(args.result_file, 'w') as result_file:
        result_file.write(
            "template, block len, rz weight, n qubits, launch mean, launch sem, jit mean, jit sem, launch opt mean, launch opt sem, jit opt mean, jit opt sem\n"
        )
        result_file.flush()
        for length in args.block_lengths:
            for rz_weight in args.rz_weights:
                for n_qubits in args.n_qubits:
                    options = Options()
                    options.block_length = int(length)
                    options.rz_weight = float(rz_weight)
                    options.n_qubits = int(n_qubits)
                    print(str(options))
                    benchmark("simple.template", result_file, options,
                              args.seeds, args.iterations)

    log_file.close()

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
result_file = None
raw_file = None


def log_command(command: str):
    log_file.write(("# " + command + "\n").encode())
    log_file.flush()


def log_out(output: str):
    log_file.write(output.encode())
    log_file.flush()


def write_result_line(csv_data: list):
    result_file.write(",".join([str(x) for x in csv_data]) + "\n")
    result_file.flush()


def write_raw_line(csv_data: list):
    if raw_file is None:
        return
    raw_file.write(",".join([str(x) for x in csv_data]) + "\n")
    raw_file.flush()


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
    log_command("python3 random_gen.py " + template + " " + str(options))
    tmp_filename = template + ".cpp"
    with open(tmp_filename, 'w') as tmp_file:
        tmp_file.write(program)
    subprocess.run(
        ["nvq++", "--target=remote-mqpu", tmp_filename, "-o", outfile],
        check=True)
    log_command("nvq++  --target=remote-mqpu {} -o {}".format(
        tmp_filename, outfile))


# Run a single benchmark, capture time info
def run_benchmark(executable):

    def parse_launch_time(output):
        #print(out)
        match = re.search('Total time: (.*)ms', output)
        launch_time = float(match.groups()[0])
        return launch_time

    def parse_JIT_time(output):
        #print(out)
        match = re.search('JIT time: (.*)ms', output)
        jit_time = float(match.groups()[0])
        return jit_time

    def parse_n_rzs(output):
        #print(out)
        match = re.search('# rzs: (\d*)', output)
        n_rzs = int(match.groups()[0])
        return n_rzs

    def run_command(command):
        log_command(command)
        output = subprocess.run(command,
                                shell=True,
                                capture_output=True,
                                check=True,
                                text=True).stdout
        log_out(output)
        results = {
            "launch": parse_launch_time(output),
            "jit": parse_JIT_time(output),
            "n_rzs": parse_n_rzs(output),
        }
        return results

    results = run_command("CUDAQ_TIMING_TAGS=3,6 CUDAQ_PHASE_FOLDING=false ./" +
                          executable)
    opt_results = run_command(
        "CUDAQ_TIMING_TAGS=3,6 CUDAQ_PHASE_FOLDING=true ./" + executable)
    #log_command("Launch time: {}, with opt: {}. JIT time: {}, with opt: {}".format(
    #    results["launch"], opt_results["launch"], results["jit"], opt_results["jit"]))
    return [results, opt_results]


# Generate and run benchmark based on template using options
# Runs with multiple random seeds to reduce the risk of outlier seeds
def benchmark(template, options, seeds=range(0, 50, 10), iterations=5):
    launch_times = []
    opt_launch_times = []
    jit_times = []
    opt_jit_times = []
    n_rzs = []
    opt_n_rzs = []
    for seed in seeds:
        exec_file = "tmp.x"
        options.seed = seed
        prepare_benchmark(template, exec_file, options)
        # Do a warmup run to prepare the cache, etc...
        # However, we take the number of rzs from this warmup, as it
        # is constant across runs for the same seed, unlike the times
        [results, opt_results] = run_benchmark(exec_file)
        n_rzs.append(results["n_rzs"])
        opt_n_rzs.append(opt_results["n_rzs"])
        for _ in range(0, iterations + 1):
            [results, opt_results] = run_benchmark(exec_file)
            launch_times.append(results["launch"])
            jit_times.append(results["jit"])
            opt_launch_times.append(opt_results["launch"])
            opt_jit_times.append(opt_results["jit"])
            raw_data = [
                template, seed, results["launch"], results["jit"],
                opt_results["launch"], opt_results["jit"], results["n_rzs"],
                opt_results["n_rzs"]
            ]
            write_raw_line(raw_data)
    raw_data = [launch_times, opt_launch_times, jit_times, opt_jit_times]
    csv_data = [template, seed]
    for datum in raw_data:
        csv_data += [stats.gmean(datum), stats.sem(datum)]
    csv_data += [stats.gmean(n_rzs), stats.gmean(opt_n_rzs)]
    write_result_line(csv_data)


argparser = argparse.ArgumentParser(
    prog='Optimization Benchmarker',
    description=
    'Configurable benchmarking of optimization using randomly generated kernels',
    epilog='')
argparser.add_argument('--result-file', type=str, default="results.csv")
argparser.add_argument('--raw-data-file', type=str, default=None)
argparser.add_argument('--block-lengths', nargs="+", type=int)
argparser.add_argument('--rz-weights', nargs="+", type=float)
argparser.add_argument('--n-qubits', nargs="+", type=int)
argparser.add_argument('--seeds', nargs="+", type=int, default=[0])
argparser.add_argument('--iterations', type=int, default=5)

if __name__ == '__main__':
    log_file = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
    log_command(" ".join(sys.argv))
    args = argparser.parse_args()
    print("Logging to " + log_file.name)
    print("Outputting to " + args.result_file)
    result_file = open(args.result_file, 'w')
    result_file.write(
        "template, seed, block len, rz weight, n qubits, launch mean, launch sem, launch opt mean, launch opt sem, jit opt mean, jit opt sem, jit mean, jit sem, # rzs, opt # rzs\n"
    )
    result_file.flush()
    if args.raw_data_file is not None:
        print("Outputting raw data to " + args.raw_data_file)
        raw_file = open(args.raw_data_file, 'w')
        raw_file.write(
            "template, seed, block len, rz weight, n qubits, launch mean, launch sem, launch opt mean, launch opt sem, jit opt mean, jit opt sem, jit mean, jit sem, # rzs, opt # rzs\n"
        )
        raw_file.flush()
    for length in args.block_lengths:
        for rz_weight in args.rz_weights:
            for n_qubits in args.n_qubits:
                options = Options()
                options.block_length = int(length)
                options.rz_weight = float(rz_weight)
                options.n_qubits = int(n_qubits)
                print(str(options))
                benchmark("simple.template", options, args.seeds,
                          args.iterations)

    log_file.close()

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
import random
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
def prepare_benchmark(template, outfile, options, target):
    log_command("python3 random_gen.py " + template + " " + str(options))
    program = generate_program(template, options.seed, options.rz_weight,
                               [options.block_length], [options.n_qubits])
    tmp_filename = template + ".cpp"
    with open(tmp_filename, 'w') as tmp_file:
        tmp_file.write(program)
    log_command(" ".join(
        ["nvq++", "--target=" + target, tmp_filename, "-o", outfile]))
    subprocess.run(" ".join(
        ["nvq++", "--target=" + target, tmp_filename, "-o", outfile]),
                   shell=True,
                   check=True)


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
def benchmark(template, options, seeds, iterations, target):
    launch_times = []
    opt_launch_times = []
    jit_times = []
    opt_jit_times = []
    n_rzs = []
    opt_n_rzs = []
    for seed in seeds:
        print("\tGenerating circuit with seed " + str(seed))
        exec_file = "tmp.x"
        options.seed = seed
        print("\t\tCompiling...")
        prepare_benchmark(template, exec_file, options, target)
        # Do a warmup run to prepare the cache, etc...
        # However, we take the number of rzs from this warmup, as it
        # is constant across runs for the same seed, unlike the times
        print("\t\tRunning warm up...")
        [results, opt_results] = run_benchmark(exec_file)
        n_rzs.append(results["n_rzs"])
        opt_n_rzs.append(opt_results["n_rzs"])
        for iteration in range(0, iterations):
            print("\t\tRunning iteration {} out of {}".format(
                iteration + 1, iterations))
            [results, opt_results] = run_benchmark(exec_file)
            launch_times.append(results["launch"])
            jit_times.append(results["jit"])
            opt_launch_times.append(opt_results["launch"])
            opt_jit_times.append(opt_results["jit"])
            raw_data = [
                template, seed, options.block_length, options.rz_weight,
                options.n_qubits, results["launch"], results["jit"],
                opt_results["launch"], opt_results["jit"], results["n_rzs"],
                opt_results["n_rzs"]
            ]
            write_raw_line(raw_data)
    raw_data = [launch_times, opt_launch_times, jit_times, opt_jit_times]
    csv_data = [
        template, options.block_length, options.rz_weight, options.n_qubits
    ]
    for datum in raw_data:
        csv_data += [stats.gmean(datum), stats.sem(datum)]
    csv_data += [stats.gmean(n_rzs), stats.gmean(opt_n_rzs)]
    write_result_line(csv_data)


argparser = argparse.ArgumentParser(
    prog='Optimization Benchmarker',
    description=
    'Configurable benchmarking of optimization using randomly generated kernels',
    epilog='')
argparser.add_argument('--result-file',
                       type=str,
                       default="results.csv",
                       help="The file to which results (mean, sem) are written")
argparser.add_argument(
    '--raw-data-file',
    type=str,
    default=None,
    help="If provided, dumps raw results to the provided file in csv format")
argparser.add_argument(
    '--block-lengths',
    nargs="+",
    type=int,
    help="Configuration variable: number of gates in the circuit")
argparser.add_argument(
    '--rz-weights',
    nargs="+",
    type=float,
    help=
    "Configuration variable: probability with which a given instruction is an rz"
)
argparser.add_argument('--n-qubits',
                       nargs="+",
                       type=int,
                       help="Configuration variable: number of qubits")
argparser.add_argument('--seed', type=int, default=0)
argparser.add_argument(
    '--n-seeds',
    type=int,
    default=5,
    help="The number of random circuits to generate for each configuration")
argparser.add_argument(
    '--iterations',
    type=int,
    default=3,
    help="The number of times to run each randomly generated circuit")
argparser.add_argument('--targets',
                       type=str,
                       nargs="+",
                       default=["remote-mqpu"],
                       help="A list of targets to run on")

if __name__ == '__main__':
    log_file = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
    log_command(" ".join(sys.argv))
    args = argparser.parse_args()

    random.seed(args.seed)
    print("Logging to " + log_file.name)
    print("Outputting to " + args.result_file)
    result_file = open(args.result_file, 'w')
    result_file.write(
        "template, block len, rz weight, n qubits, launch mean, launch sem, launch opt mean, launch opt sem, jit mean, jit sem, jit opt mean, jit opt sem, # rzs, opt # rzs\n"
    )
    result_file.flush()
    if args.raw_data_file is not None:
        print("Outputting raw data to " + args.raw_data_file)
        raw_file = open(args.raw_data_file, 'w')
        raw_file.write(
            "template, seed, block len, rz weight, n qubits, launch, launch opt, jit, jit opt, # rzs, opt # rzs\n"
        )
        raw_file.flush()
    for target in args.targets:
        for length in args.block_lengths:
            for rz_weight in args.rz_weights:
                for n_qubits in args.n_qubits:
                    options = Options()
                    options.block_length = int(length)
                    options.rz_weight = float(rz_weight)
                    options.n_qubits = int(n_qubits)
                    print(
                        "Running configuration: target={}, block-length={}, rz-weight={}, n-qubits={}"
                        .format(target, length, rz_weight, n_qubits))
                    seeds = [
                        random.randint(0, sys.maxsize)
                        for _ in range(0, args.n_seeds)
                    ]
                    benchmark("simple.template", options, seeds,
                              args.iterations, target)

    log_file.close()

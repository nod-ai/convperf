#!/usr/bin/env python3
# Main driver to run performance experiments
import argparse
import subprocess
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import os
import json
plt.style.use('ggplot')

BAR_WIDTH = 0.15
BAR_COLORS = {
    'xsmm': 'red',
    'iree': 'blue',
}

def run(cmd, args):
    bench_env = os.environ.copy()
    bench_env["OMP_NUM_THREADS"] = f"{args.num_threads}"
    cmd = ['numactl', '--cpunodebind=1', '-l'] + cmd
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=bench_env)
    output = p.communicate()
    runtimes = []
    for line in output[0].decode('utf-8').split('\n'):
        if 'real_time' in line:
            tokens = line.split()
            runtimes.append(int(tokens[1]))
        print(line)
    p.wait()
    return runtimes

def benchmark_iree(args):
    cmd = [args.benchmark_tool] \
        + ['-r', 'iree']
    if args.num_threads > 1:
        cmd += [f'--task_topology_max_group_count={args.num_threads-1}']
    return run(cmd, args)

def benchmark_xsmm(args):
    cmd = [args.benchmark_tool] \
        + ['-r', 'xsmm']
    return run(cmd, args)

def benchmark(args):
    runtimes = {"iree": [], "xsmm": []}
    for runner in args.runners.split(','):
        if runner == "iree":
            runtimes["iree"] = benchmark_iree(args)
        elif runner == "xsmm":
            runtimes["xsmm"] = benchmark_xsmm(args)
        else:
            print("Unsupported runner!")
    return runtimes

def shape_str(config, is_filter):
    if is_filter:
        N = config['F']
    else:
        N = config['N']
    C = config['C']
    H = config['H']
    W = config['W']
    p_str = ''
    for v in config['format']:
        if v == 'n' or v == 'f':
            p_str += f'{N}x'
        if v == 'h':
            p_str += f'{H}x'
        if v == 'w':
            p_str += f'{W}x'
        if v == 'c':
            p_str += f'{C}x'
    p_str = p_str[:-1]
    return p_str

def compute_flops(config):
    N = config["input"]["N"]
    Cout = config["output"]["C"]
    Cin = config["input"]["C"]
    Hout = config["output"]["H"]
    Wout = config["output"]["W"]
    Kh = config["filter"]["H"]
    Kw = config["filter"]["W"]
    flops = 2 * N * Cin * Cout * Hout * Wout * Kh * Kw
    return flops

def compute_labels_and_flops(configs):
    labels = []
    flops = []
    for config in configs:
        labels.append(shape_str(config["input"], False) + "_" + shape_str(config["filter"], True))
        flops.append(compute_flops(config))
    return labels, flops

def save_runtimes(args, runtimes):
    runtimes["benchmark_sizes"] = args.benchmark_sizes
    with open("runtimes.json", "w") as f:
        json.dump(runtimes, f, ensure_ascii=False, indent=4, sort_keys=True)

def visualize(args):
    generate_x = lambda i, labels : np.arange(len(labels)) + i * BAR_WIDTH
    with open(args.runtimes_file, "r") as f:
        data = json.load(f)
    with open(data["benchmark_sizes"], "r") as f:
        sizes = json.load(f)
    del data["benchmark_sizes"]
    labels, flops = compute_labels_and_flops(sizes["configs"])
    for i, method in enumerate(data.keys()):
        speed = [(y / x / 1e6) for x, y in zip(data[method], flops)] 
        print(f"MFLOPS[{method}]: {speed}")
        plt.bar(generate_x(i, labels), speed, BAR_WIDTH, label=method, color=BAR_COLORS[method])

    x_pos = [i + 0.5*(len(labels) - 1)*BAR_WIDTH for i in range(len(labels))]
    plt.xticks(x_pos, labels, rotation=90, fontsize=5)
    plt.xlabel('Convolution sizes')
    plt.ylabel('MFLOPS')
    plt.title("2D Convolution in fp32")
    plt.legend(loc='best')
    plt.savefig('convs.png', dpi=300, bbox_inches='tight')

def define_options(parser):
    parser.add_argument('--benchmark_tool', type=str, help='Path to benchmark tool')
    parser.add_argument('--runners', type=str, help='Methods to be benchmarked')
    parser.add_argument('--visualize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--benchmark_sizes', type=str, help='Path to benchmark sizes file')
    parser.add_argument('--runtimes_file', type=str, help='Path to runtimes file')
    parser.add_argument('--num_threads', type=int, help='Number of threads to run benchmark on')

parser = argparse.ArgumentParser()
define_options(parser)
args = parser.parse_args()

if args.visualize is None:
    runtimes = benchmark(args)
    save_runtimes(args, runtimes)
else:
    visualize(args)

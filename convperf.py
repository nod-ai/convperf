#!/usr/bin/env python3
# Main driver to run performance experiments
import argparse
import subprocess
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

BAR_WIDTH = 0.15
BAR_COLORS = {
    'xsmm': 'red',
    'iree': 'blue',
}

def run(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
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
        + ['-r', 'iree', \
           f'--task_topology_max_group_count={multiprocessing.cpu_count()}']
    return run(cmd)

def benchmark_xsmm(args):
    cmd = [args.benchmark_tool] \
        + ['-r', 'xsmm']
    return run(cmd)

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

def get_sizes(args):
    input_sizes = []
    filter_sizes = []
    str = 'Method,'
    with open(args.benchmark_sizes, "r") as f:
        for line in f.readlines():
            tokens = line.split(',')
            input_sizes.append(tokens[0])
            filter_sizes.append(tokens[1])
            str += tokens[0] + "_" + tokens[1] + ","
        str = str[:-1]
        str += "\n"
    return input_sizes, filter_sizes, str

def save_runtimes(args, runtimes):
    input_sizes, filter_sizes, str = get_sizes(args)
    with open("runtimes.csv", "w") as f:
        f.write(str)
        for method, runtime in runtimes.items():
            str = method + ","
            for time in runtime:
                str += f"{time},"
            str = str[:-1]
            str += "\n"
            f.write(str)

def visualize(args):
    generate_x = lambda i, labels : np.arange(len(labels)) + i * BAR_WIDTH
    labels = None
    with open("runtimes.csv", "r") as f:
        i = 0
        for line in f.readlines():
            tokens = line.rstrip().split(',')
            if i == 0:
                labels = tokens[1:]
                i += 1
                continue
            method = tokens[0]
            runtimes = [float(x)/1e6 for x in tokens[1:]]
            plt.bar(generate_x(i-1, labels), runtimes, BAR_WIDTH, label=method, color=BAR_COLORS[method])
            i += 1

    x_pos = [i + 0.5*(len(labels) - 1)*BAR_WIDTH for i in range(len(labels))]
    plt.xticks(x_pos, labels, rotation=90, fontsize=5)
    plt.xlabel('Convolution sizes')
    plt.ylabel('Execution time(ms)')
    plt.title("2D Convolution in fp32")
    plt.legend(loc='best')
    plt.savefig('convs.png', dpi=300, bbox_inches='tight')

def define_options(parser):
    parser.add_argument('--benchmark_tool', type=str, help='Path to benchmark tool')
    parser.add_argument('--runners', type=str, help='Methods to be benchmarked')
    parser.add_argument('--visualize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--benchmark_sizes', type=str, help='Path to benchmark sizes file')
    parser.add_argument('--runtimes_file', type=str, help='Path to runtimes file')

parser = argparse.ArgumentParser()
define_options(parser)
args = parser.parse_args()

if args.visualize is None:
    runtimes = benchmark(args)
    save_runtimes(args, runtimes)
else:
    visualize(args)

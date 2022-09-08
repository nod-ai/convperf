#!/usr/bin/env python3
# Main driver to run performance experiments
import argparse
import subprocess
import multiprocessing

def benchmark_iree(args):
    cmd = [args.benchmark_tool] \
        + ['-r', 'iree', \
           f'--task_topology_max_group_count={multiprocessing.cpu_count()}']
    p = subprocess.Popen(cmd)
    p.wait()

def benchmark_xsmm(args):
    cmd = [args.benchmark_tool] \
        + ['-r', 'xsmm']
    p = subprocess.Popen(cmd)
    p.wait()

def benchmark(args):
    for runner in args.runners.split(','):
        if runner == "iree":
            benchmark_iree(args)
        elif runner == "xsmm":
            benchmark_xsmm(args)
        else:
            print("Unsupported runner!")

def define_options(parser):
    parser.add_argument('--benchmark_tool', type=str, help='Path to benchmark tool')
    parser.add_argument('--runners', type=str, help='Methods to be benchmarked')

parser = argparse.ArgumentParser()
define_options(parser)
args = parser.parse_args()
benchmark(args)

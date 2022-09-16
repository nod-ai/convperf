#!/usr/bin/env python3
import argparse
import subprocess
import json

def compile(args):
    compile_flags = [
        "-iree-mlir-to-vm-bytecode-module",
        "-iree-hal-target-backends=llvm-cpu",
        "-iree-hal-benchmark-dispatch-repeat-count=100",
        "-iree-llvm-target-cpu-features=host",
        "-iree-flow-enable-fuse-padding-into-consumer-ops",
        "-iree-llvmcpu-enable-hoist-padding",
        "-iree-llvm-debug-symbols=false",
        "-iree-vm-bytecode-module-strip-source-map=true",
        "-iree-llvm-keep-linker-artifacts",
        "-iree-vm-emit-polyglot-zip=false",
        f"{args.mlir_file}" + ".mlir",
        "-o",
        f"{args.mlir_file}.vmfb",
    ]
    out = subprocess.PIPE
    if args.dump_output:
        compile_flags += [
            '-mlir-print-ir-after-all'
        ]
        out = open('mlir_dump.txt', 'w')
    combined = [args.compile_tool] + compile_flags
    print(' '.join(combined))
    subprocess.run(combined, check=True, stderr=out)
    if args.dump_output:
        out.close()

def configure_convolution(args):
    I = [None for _ in range(4)]
    O = [None for _ in range(4)]
    F = [None for _ in range(4)]
    U = [None for _ in range(4)]
    P = [(None, None) for _ in range(4)]
    ph = int(args.padding[0])
    pw = int(args.padding[1])
    for i, (letter, letter_f) in enumerate(zip(args.input_format, args.filter_format)):
        if letter == 'n':
            I[i] = args.N
            O[i] = args.N
            P[i] = (0, 0)
        if letter == 'c':
            I[i] = args.Cin
            O[i] = args.Cout
            P[i] = (0, 0)
        if letter_f == 'c':
            F[i] = args.Cin
        if letter == 'h':
            I[i] = args.Hin
            O[i] = args.Hout
            P[i] = (ph, ph)
        if letter_f == 'h':
            F[i] = args.Kh
        if letter == 'w':
            I[i] = args.Win
            O[i] = args.Wout
            P[i] = (pw, pw)
        if letter_f == 'w':
            F[i] = args.Kw
        if letter_f == 'f':
            F[i] = args.Cout
        U[i] = int(I[i]) + P[i][0] + P[i][1]

    D = args.dilations[0]
    S = args.strides[0]

    if int(args.padding[0]) == 0:
        conv_mlir = \
        f"func.func @conv2d_{I[0]}x{I[1]}x{I[2]}x{I[3]}_{F[0]}x{F[1]}x{F[2]}x{F[3]}(%arg0: tensor<{I[0]}x{I[1]}x{I[2]}x{I[3]}xf32>, %arg1: tensor<{F[0]}x{F[1]}x{F[2]}x{F[3]}xf32>) -> tensor<{O[0]}x{O[1]}x{O[2]}x{O[3]}xf32> {{\n" + \
        "  %cst_0 = arith.constant 0.000000e+00 : f32\n" + \
        f"  %0 = linalg.init_tensor [{O[0]}, {O[1]}, {O[2]}, {O[3]}] : tensor<{O[0]}x{O[1]}x{O[2]}x{O[3]}xf32>\n" + \
        f"  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<{O[0]}x{O[1]}x{O[2]}x{O[3]}xf32>) -> tensor<{O[0]}x{O[1]}x{O[2]}x{O[3]}xf32>\n" + \
        f"  %2 = linalg.conv_2d_{args.input_format}_{args.filter_format} {{dilations = dense<{D}> : tensor<2xi64>, strides = dense<{S}> : tensor<2xi64>}} ins(%arg0, %arg1 :" + \
        f" tensor<{I[0]}x{I[1]}x{I[2]}x{I[3]}xf32>, tensor<{F[0]}x{F[1]}x{F[2]}x{F[3]}xf32>)" + \
        f" outs(%1 : tensor<{O[0]}x{O[1]}x{O[2]}x{O[3]}xf32>) -> tensor<{O[0]}x{O[1]}x{O[2]}x{O[3]}xf32>\n" + \
        f"  return %2 : tensor<{O[0]}x{O[1]}x{O[2]}x{O[3]}xf32>\n" + \
        "}\n"
    else:
        conv_mlir = \
        f"func.func @conv2d_{I[0]}x{I[1]}x{I[2]}x{I[3]}_{F[0]}x{F[1]}x{F[2]}x{F[3]}(%arg0: tensor<{I[0]}x{I[1]}x{I[2]}x{I[3]}xf32>, %arg1: tensor<{F[0]}x{F[1]}x{F[2]}x{F[3]}xf32>) -> tensor<{O[0]}x{O[1]}x{O[2]}x{O[3]}xf32> {{\n" + \
        "  %cst_0 = arith.constant 0.000000e+00 : f32\n" + \
        f"  %0 = linalg.init_tensor [{O[0]}, {O[1]}, {O[2]}, {O[3]}] : tensor<{O[0]}x{O[1]}x{O[2]}x{O[3]}xf32>\n" + \
        f"  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<{O[0]}x{O[1]}x{O[2]}x{O[3]}xf32>) -> tensor<{O[0]}x{O[1]}x{O[2]}x{O[3]}xf32>\n" + \
        f"  %2 = tensor.pad %arg0 low[{P[0][0]}, {P[1][0]}, {P[2][0]}, {P[3][0]}] high[{P[0][1]}, {P[1][1]}, {P[2][1]}, {P[3][1]}] {{\n" + \
         "         ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):\n" + \
         "                tensor.yield %cst_0 : f32\n" + \
        f"       }} : tensor<{I[0]}x{I[1]}x{I[2]}x{I[3]}xf32> to tensor<{U[0]}x{U[1]}x{U[2]}x{U[3]}xf32>\n" + \
        f"  %3 = linalg.conv_2d_{args.input_format}_{args.filter_format} {{dilations = dense<{D}> : tensor<2xi64>, strides = dense<{S}> : tensor<2xi64>}} ins(%2, %arg1 :" + \
        f" tensor<{U[0]}x{U[1]}x{U[2]}x{U[3]}xf32>, tensor<{F[0]}x{F[1]}x{F[2]}x{F[3]}xf32>)" + \
        f" outs(%1 : tensor<{O[0]}x{O[1]}x{O[2]}x{O[3]}xf32>) -> tensor<{O[0]}x{O[1]}x{O[2]}x{O[3]}xf32>\n" + \
        f"  return %3 : tensor<{O[0]}x{O[1]}x{O[2]}x{O[3]}xf32>\n" + \
        "}\n"

    return conv_mlir

def compile_sizes(args):
    all_convs = ''
    with open(args.sizes_file, 'r') as f:
        data = json.load(f)
    for config in data["configs"]:
        args.N = config['input']['N']
        args.Cin = config['input']['C']
        args.Hin = config['input']['H']
        args.Win = config['input']['W']
        args.Cout = config['output']['C']
        args.Hout = config['output']['H']
        args.Wout = config['output']['W']
        args.Kh = config['filter']['H']
        args.Kw = config['filter']['W']
        args.strides = config['strides']
        args.padding = config['padding']
        args.dilations = config['dilations']
        args.input_format = config['input']['format']
        args.filter_format = config['filter']['format']
        conv = configure_convolution(args)
        all_convs += conv
    args.mlir_file = "convs"
    with open(args.mlir_file + ".mlir", "w") as f:
        f.write(all_convs)
    compile(args)

def define_options(parser):
    parser.add_argument('--sizes_file', type=str, help='File containing sizes to benchmark')
    parser.add_argument('--compile_tool', type=str, help='Path to iree-compile')
    parser.add_argument('--dump_output', action='store_true')

parser = argparse.ArgumentParser()
define_options(parser)
args = parser.parse_args()
compile_sizes(args)

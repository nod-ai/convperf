#!/usr/bin/env python3
import argparse
import subprocess

def compile(args):
    compile_flags = [
        "-iree-mlir-to-vm-bytecode-module",
        "-iree-hal-target-backends=llvm-cpu",
        "-iree-llvm-target-cpu-features=host",
        "-iree-llvmcpu-enable-hoist-padding",
        "-iree-llvm-debug-symbols=false",
        "-iree-vm-bytecode-module-strip-source-map=true",
        "-iree-vm-emit-polyglot-zip=false",
        f"{args.mlir_file}" + ".mlir",
        "-o",
        f"{args.mlir_file}.vmfb",
    ]
    combined = [args.compile_tool] + compile_flags
    print(' '.join(combined))
    subprocess.run(combined, check=True)

def configure_convolution(args):
    if args.input_format == "nhwc":
        I0 = args.N
        I1 = args.Hin
        I2 = args.Win
        I3 = args.Cin
        O0 = args.N
        O1 = args.Hout
        O2 = args.Wout
        O3 = args.Cout
    elif args.input_format == "nchw":
        I0 = args.N
        I1 = args.Cin
        I2 = args.Hin
        I3 = args.Win
        O0 = args.N
        O1 = args.Cout
        O2 = args.Hout
        O3 = args.Wout
    if args.filter_format == "hwcf":
        F0 = args.Kh
        F1 = args.Kw
        F2 = args.Cin
        F3 = args.Cout
    elif args.filter_format == "fchw":
        F0 = args.Cout
        F1 = args.Cin
        F2 = args.Kh
        F3 = args.Kw
    D = args.dilations[0]
    S = args.strides[0]

    conv_mlir = \
    f"func.func @conv2d_{I0}x{I1}x{I2}x{I3}_{F0}x{F1}x{F2}x{F3}(%arg0: tensor<{I0}x{I1}x{I2}x{I3}xf32>, %arg1: tensor<{F0}x{F1}x{F2}x{F3}xf32>) -> tensor<{O0}x{O1}x{O2}x{O3}xf32> {{\n" + \
    "  %cst_0 = arith.constant 0.000000e+00 : f32\n" + \
    f"  %0 = linalg.init_tensor [{O0}, {O1}, {O2}, {O3}] : tensor<{O0}x{O1}x{O2}x{O3}xf32>\n" + \
    f"  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<{O0}x{O1}x{O2}x{O3}xf32>) -> tensor<{O0}x{O1}x{O2}x{O3}xf32>\n" + \
    f"  %2 = linalg.conv_2d_{args.input_format}_{args.filter_format} {{dilations = dense<{D}> : tensor<2xi64>, strides = dense<{S}> : tensor<2xi64>}} ins(%arg0, %arg1 :" + \
    f" tensor<{I0}x{I1}x{I2}x{I3}xf32>, tensor<{F0}x{F1}x{F2}x{F3}xf32>)" + \
    f" outs(%1 : tensor<{O0}x{O1}x{O2}x{O3}xf32>) -> tensor<{O0}x{O1}x{O2}x{O3}xf32>\n" + \
    f"  return %2 : tensor<{O0}x{O1}x{O2}x{O3}xf32>\n" + \
    "}\n"

    return conv_mlir

def compile_sizes(args):
    all_convs = ''
    with open(args.sizes_file, 'r') as f:
        for line in f.readlines():
            params = line.rstrip().split(',')
            args.N, args.Cin, args.Hin, args.Win = params[0].split('x')
            args.Cout, args.Cin, args.Kh, args.Kw = params[1].split('x')
            args.N, args.Cout, args.Hout, args.Wout = params[2].split('x')
            args.strides = [params[3], params[4]]
            args.dilations = [params[7], params[8]]
            conv = configure_convolution(args)
            all_convs += conv
    args.mlir_file = "convs"
    with open(args.mlir_file + ".mlir", "w") as f:
        f.write(all_convs)
    compile(args)

def define_options(parser):
    parser.add_argument('--sizes_file', type=str, help='File containing sizes to benchmark')
    parser.add_argument('--compile_tool', type=str, help='Path to iree-compile')
    parser.add_argument('--input_format', type=str, help='Input format', choices=['nhwc', 'nchw'])
    parser.add_argument('--filter_format', type=str, help='Filter format', choices=['hwcf', 'fchw'])

parser = argparse.ArgumentParser()
define_options(parser)
args = parser.parse_args()
compile_sizes(args)

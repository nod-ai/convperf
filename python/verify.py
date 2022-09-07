# Code to verify accuracy of naive C++ based approach
# Here we validate the first convolution in Resnet50 vs PyTorch

import torch
import numpy as np

def get_shape_and_format(file_name):
    shape, format = open(file_name).readline().rstrip().split(',')
    shape = [int(x) for x in shape.split('x')]
    return shape, format

def get_tensor(file_name, shape, format):
    tensor = np.loadtxt(file_name, delimiter=',', skiprows=1)
    tensor = torch.from_numpy(np.reshape(tensor, shape))
    if format == "nhwc":
        tensor = tensor.permute(0, 3, 1, 2)
    if format == "hwcf":
        tensor = tensor.permute(3, 2, 0, 1)
    return tensor

input_shape, input_format = get_shape_and_format('input.csv')
output_shape, output_format = get_shape_and_format('output.csv')
filter_shape, filter_format = get_shape_and_format('filter.csv')

input = get_tensor('input.csv', input_shape, input_format)
output_c = get_tensor('output.csv', output_shape, output_format)
filter = get_tensor('filter.csv', filter_shape, filter_format)

output = torch.nn.functional.conv2d(input, filter, bias=None, stride=(2, 2),
                                    padding=(0, 0), dilation=(1, 1), groups=1)

error = torch.max(torch.abs(output - output_c))
print("Max error = ", error)
if error.item() > 1e-4:
    print("Fail!")
else:
    print("Success!")


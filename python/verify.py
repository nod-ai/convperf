# Code to verify accuracy of naive C++ based approach
# Here we validate the first convolution in Resnet50 vs PyTorch

import torch
import numpy as np

input_shape = np.loadtxt('input.csv', delimiter=',', max_rows=1, dtype=np.int32)
output_shape = np.loadtxt('output.csv', delimiter=',', max_rows=1, dtype=np.int32)
filter_shape = np.loadtxt('filter.csv', delimiter=',', max_rows=1, dtype=np.int32)

input = np.loadtxt('input.csv', delimiter=',', skiprows=1)
input = torch.from_numpy(np.reshape(input, input_shape))
output_c = np.loadtxt('output.csv', delimiter=',', skiprows=1)
output_c = torch.from_numpy(np.reshape(output_c, output_shape))
filter = np.loadtxt('filter.csv', delimiter=',', skiprows=1)
filter = torch.from_numpy(np.reshape(filter, filter_shape))

output = torch.nn.functional.conv2d(input, filter, bias=None, stride=(2, 2),
                                    padding=(0, 0), dilation=(1, 1), groups=1)

error = torch.max(torch.abs(output - output_c))
print("Max error = ", error)
if error.item() > 1e-4:
    print("Fail!")
else:
    print("Success!")


import torch
import numpy as np
from conv2d_OpenclMulti import Conv2DOpenCL

# Function to run the OpenCL convolution
def run_opencl_conv(input_array, filters, in_channels, out_channels, kernel_size, stride, padding):
    conv_opencl = Conv2DOpenCL(aocx_file_path='./conv2d/conv2d.aocx')  # Path to your .aocx file
    return conv_opencl.execute(input_array, filters, in_channels, out_channels, kernel_size, stride, padding)

# Function to run the PyTorch convolution
def run_pytorch_conv(input_tensor, filters, in_channels, out_channels, kernel_size, stride, padding):
    # Define the convolution layer in PyTorch
    conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    conv_layer.weight.data = torch.tensor(filters, dtype=torch.float32)
    return conv_layer(input_tensor)

# Test settings
batch_size = 100
in_channels = 4
out_channels = 8
input_h, input_w = 240, 240
kernel_size = 3
stride = 1
padding = 1

# Generate random input and filter data
np.random.seed(0)
input_array_np = np.random.randn(batch_size, in_channels, input_h, input_w).astype(np.float32)
filters_np = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32)

# Convert input_array and filters to PyTorch tensors
input_tensor_torch = torch.tensor(input_array_np, dtype=torch.float32)
filters_torch = torch.tensor(filters_np, dtype=torch.float32)

# Run OpenCL convolution
output_array_opencl = run_opencl_conv(input_array_np, filters_np, in_channels, out_channels, kernel_size, stride, padding)

# Run PyTorch convolution
output_tensor_torch = run_pytorch_conv(input_tensor_torch, filters_torch, in_channels, out_channels, kernel_size, stride, padding)

# Compare the results
print("OpenCL Output:")
print(output_array_opencl.shape)
print(output_array_opencl)

print("\nPyTorch Output:")
print(output_tensor_torch.shape)
print(output_tensor_torch.detach().numpy())

# Check if the results are similar
difference = np.abs(output_array_opencl - output_tensor_torch.detach().numpy())
max_difference = np.max(difference)
print("\nMaximum Difference between OpenCL and PyTorch output:", max_difference)

# You can assert that the results are close within a tolerance
assert max_difference < 1e-4, f"Results are not close. Max difference: {max_difference}"
print("\nTest Passed: OpenCL and PyTorch results are similar!")

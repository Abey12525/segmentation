import torch
import torch.nn as nn
import numpy as np
from conv2d_OpenclMulti import Conv2DOpenCL

def test_conv2d_comparison():
    # Parameters
    batch_size = 1
    in_channels = 3  # Number of input channels
    out_channels = 2  # Number of output channels
    height, width = 8, 8  # Dimensions of the input
    kernel_size = 3  # Testing 3x3 convolution
    stride = 1
    padding = 1

    # Generate random input tensor (PyTorch) and OpenCL
    input_tensor = torch.rand(batch_size, in_channels, height, width, dtype=torch.float32)
    input_array = input_tensor.numpy()  # Convert to numpy for OpenCL

    # Generate random filter tensor (PyTorch) and OpenCL
    filter_tensor = torch.rand(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32)
    filter_array = filter_tensor.numpy()  # Convert to numpy for OpenCL

    # Initialize the OpenCL Conv2D module with only kernel size
    aocx_file_path = './conv2d/conv2d.aocx'  # Path to the precompiled .aocx file
    conv2d_opencl = Conv2DOpenCL(aocx_file_path=aocx_file_path)

    # Compute output with OpenCL (passing in_channels, out_channels, stride, padding to execute)
    output_opencl = conv2d_opencl.execute(input_array, filter_array, in_channels=in_channels,
                                          out_channels=out_channels, kernel_size=(kernel_size, kernel_size), stride=stride, padding=padding)

    # Compute output with PyTorch
    conv2d_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=False)
    conv2d_layer.weight.data = filter_tensor  # Set the same weights for PyTorch
    output_torch = conv2d_layer(input_tensor).detach().cpu().numpy()

    # Compare the outputs
    difference = np.abs(output_opencl - output_torch)
    max_difference = difference.max()

    print("Output (OpenCL):\n", output_opencl)
    print("Output (PyTorch):\n", output_torch)
    print(f"Maximum difference between OpenCL and PyTorch outputs: {max_difference}")

    # Assert that the outputs are close within a tolerance
    assert np.allclose(output_opencl, output_torch, atol=1e-5), "Outputs do not match!"
    print("Outputs are similar within tolerance!")

if __name__ == "__main__":
    test_conv2d_comparison()

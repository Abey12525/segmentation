from conv2d_opencl import Conv2DOpenCL
import numpy as np
import torch
import torch.nn as nn

# Path to the precompiled OpenCL kernel
AOCX_FILE_PATH = "./conv2d/conv2d.aocx"

# Initialize the OpenCL Conv2D module
conv2d = Conv2DOpenCL(AOCX_FILE_PATH)

# Example to test OpenCL and PyTorch outputs for 1x1 convolution
def test_conv2d():
    # Define input tensor and filter
    batch_size = 1
    in_channels = 32  # Matches 1 * n_filters
    out_channels = 3  # Number of output channels
    height, width = 8, 8  # Dimensions of the input
    kernel_size = 3  # Testing 1x1 convolution

    # Generate random input tensor and weights
    input_tensor = torch.rand(batch_size, in_channels, height, width, dtype=torch.float32)
    filter_tensor = torch.rand(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32)

    # Compute output with PyTorch
    conv2d_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, bias=False)
    conv2d_layer.weight.data = filter_tensor  # Set weights
    output_torch = conv2d_layer(input_tensor).detach().cpu().numpy()

    # Compute output with OpenCL
    output_opencl = np.zeros((batch_size, out_channels, height, width), dtype=np.float32)
    for b in range(batch_size):  # Iterate over batch
        for out_c in range(out_channels):  # Iterate over output channels
            opencl_result = np.zeros((height, width), dtype=np.float32)
            for in_c in range(in_channels):  # Iterate over input channels
                opencl_result += conv2d.execute(
                    input_tensor[b, in_c].cpu().numpy(),  # Input for the specific channel
                    filter_tensor[out_c, in_c].cpu().numpy(),  # Corresponding filter
                    stride=1
                )
            output_opencl[b, out_c] = opencl_result  # Aggregate results for the batch

    # Compare outputs
    print("Output (PyTorch):\n", output_torch)
    print("Output (OpenCL):\n", output_opencl)
    difference = np.abs(output_torch - output_opencl)
    print("Difference:\n", difference)
    assert np.allclose(output_torch, output_opencl, atol=1e-5), "Outputs do not match!"

if __name__ == "__main__":
    test_conv2d()

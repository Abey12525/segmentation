import torch
import torch.nn as nn
import numpy as np

def test_conv2d_kernel_size_equivalence():
    # Parameters
    batch_size = 2
    in_channels = 16
    out_channels = 8
    height, width = 10, 10
    kernel_size = 3
    stride = 1
    padding = 0

    # Generate random input tensor
    input_tensor = torch.rand(batch_size, in_channels, height, width, dtype=torch.float32)

    # Initialize two Conv2d layers with equivalent configurations
    conv2d_tuple = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), stride=stride, padding=padding, bias=False)
    conv2d_int = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    # Ensure both layers have the same weights
    conv2d_int.weight.data = conv2d_tuple.weight.data.clone()

    # Compute outputs
    output_tuple = conv2d_tuple(input_tensor)
    output_int = conv2d_int(input_tensor)

    # Compare outputs
    difference = torch.abs(output_tuple - output_int)
    max_difference = difference.max().item()

    print(f"Output from kernel_size=(3,3):\n{output_tuple.shape}")
    print(f"Output from kernel_size=3:\n{output_int.shape}")
    print(f"Maximum difference: {max_difference}")

    # Assert outputs are the same
    assert torch.allclose(output_tuple, output_int, atol=1e-6), "Outputs do not match!"
    print("Both configurations produce the same output!")

if __name__ == "__main__":
    test_conv2d_kernel_size_equivalence()

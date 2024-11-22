import torch
import numpy as np
from UNetBlocks import UNetEncoderBlock, UNetDecoderBlock, conv2d_opencl

def test_unet_blocks():
    # Parameters for testing
    batch_size = 2
    in_channels = 4
    out_channels = 8
    height, width = 16, 16

    # Create dummy input
    input_tensor = torch.rand(batch_size, in_channels, height, width)

    # Initialize Encoder and Decoder Blocks
    encoder = UNetEncoderBlock(in_channels, out_channels)
    decoder = UNetDecoderBlock(out_channels, in_channels)

    # Test Encoder Block
    print("Testing UNetEncoderBlock...")
    encoded_output = encoder(input_tensor)
    assert encoded_output.shape == (batch_size, out_channels, height, width), \
        f"Encoder output shape mismatch! Expected: {(batch_size, out_channels, height, width)}, Got: {encoded_output.shape}"
    print("UNetEncoderBlock passed!")

    # Test Decoder Block
    print("Testing UNetDecoderBlock...")
    decoded_output = decoder(encoded_output)
    assert decoded_output.shape == (batch_size, in_channels, height, width), \
        f"Decoder output shape mismatch! Expected: {(batch_size, in_channels, height, width)}, Got: {decoded_output.shape}"
    print("UNetDecoderBlock passed!")

    # Validate OpenCL Outputs
    print("Validating OpenCL outputs...")
    input_array = input_tensor[0, 0].detach().cpu().numpy()  # First sample, first channel
    kernel_array = np.random.rand(3, 3).astype(np.float32)  # Example kernel

    # OpenCL Conv2D
    output_opencl = conv2d_opencl(input_array, kernel_array)
    print(f"OpenCL Conv2D output shape: {output_opencl.shape}")

    # Torch Conv2D for comparison
    torch_conv = torch.nn.Conv2d(1, 1, kernel_size=(3, 3), stride=1, padding=0, bias=False)
    torch_conv.weight.data = torch.tensor(kernel_array).unsqueeze(0).unsqueeze(0)  # Match Torch's weight shape
    torch_input = torch.tensor(input_array).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    output_torch = torch_conv(torch_input).squeeze().detach().numpy()
    print(f"Torch Conv2D output shape: {output_torch.shape}")

    # Compare OpenCL and Torch outputs
    diff = np.abs(output_opencl - output_torch)
    assert np.allclose(output_opencl, output_torch, atol=1e-5), \
        f"OpenCL and Torch outputs differ! Max difference: {diff.max()}"
    print("OpenCL and Torch Conv2D outputs match!")

if __name__ == "__main__":
    test_unet_blocks()

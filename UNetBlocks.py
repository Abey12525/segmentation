import torch.nn as nn 

class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU()):
        super().__init__()
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        batch_size, _, input_h, input_w = x.shape

        # Initialize output tensor
        output_1 = torch.zeros((batch_size, self.out_channels, input_h, input_w), dtype=torch.float32)
        output_2 = torch.zeros_like(output_1)

        # Use OpenCL conv2d for the first and second convolutions
        for b in range(batch_size):
            for c in range(self.out_channels):
                output_1[b, c] = torch.tensor(conv2d_opencl(
                    x[b, 0].detach().cpu().numpy(),  # First input channel
                    np.random.rand(3, 3).astype(np.float32)  # Example filter for OpenCL
                ))
                output_2[b, c] = torch.tensor(conv2d_opencl(
                    output_1[b, c].detach().cpu().numpy(),  # Output from the first convolution
                    np.random.rand(3, 3).astype(np.float32)  # Example filter for OpenCL
                ))

        return self.activation(output_2)  # Apply activation after the second convolution


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU()):
        super().__init__()
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        batch_size, _, input_h, input_w = x.shape

        # Initialize output tensor
        output_1 = torch.zeros((batch_size, self.out_channels, input_h, input_w), dtype=torch.float32)
        output_2 = torch.zeros_like(output_1)

        # Use OpenCL conv2d for the first and second convolutions
        for b in range(batch_size):
            for c in range(self.out_channels):
                output_1[b, c] = torch.tensor(conv2d_opencl(
                    x[b, 0].detach().cpu().numpy(),  # First input channel
                    np.random.rand(3, 3).astype(np.float32)  # Example filter for OpenCL
                ))
                output_2[b, c] = torch.tensor(conv2d_opencl(
                    output_1[b, c].detach().cpu().numpy(),  # Output from the first convolution
                    np.random.rand(3, 3).astype(np.float32)  # Example filter for OpenCL
                ))

        return self.activation(output_2)  # Apply activation after the second convolution

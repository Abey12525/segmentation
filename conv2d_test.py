import numpy as np
import pyopencl as cl
import torch
import torch.nn as nn

# OpenCL setup
def setup_opencl(aocx_file_path, kernel_name):
    """
    Set up OpenCL context, queue, and kernel from a precompiled .aocx file.

    Args:
        aocx_file_path (str): Path to the precompiled .aocx file.
        kernel_name (str): Name of the kernel function in the compiled binary.

    Returns:
        context, queue, kernel: OpenCL context, command queue, and kernel object.
    """
    platform = cl.get_platforms()[2]  # Select the first platform
    devices = platform.get_devices(device_type=cl.device_type.ACCELERATOR)  # Use FPGA or accelerator
    device = devices[0]  # Select the first available device
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # Load the precompiled kernel
    with open(aocx_file_path, 'rb') as f:
        aocx_data = f.read()

    program = cl.Program(context, [device], [aocx_data]).build()
    kernel = cl.Kernel(program, kernel_name)

    return context, queue, kernel

# OpenCL conv2d implementation
def conv2d_opencl(context, queue, kernel, input_array, filter_array, stride=1):
    """
    Perform 2D convolution using the FPGA-accelerated OpenCL kernel.

    Args:
        context: OpenCL context.
        queue: OpenCL command queue.
        kernel: Compiled OpenCL kernel object.
        input_array (np.ndarray): Input array (height x width).
        filter_array (np.ndarray): Filter array (height x width).
        stride (int): Stride of the convolution.

    Returns:
        np.ndarray: Output array after applying convolution.
    """
    input_h, input_w = input_array.shape
    filter_h, filter_w = filter_array.shape
    output_h = (input_h - filter_h) // stride + 1
    output_w = (input_w - filter_w) // stride + 1

    # Create OpenCL buffers
    mf = cl.mem_flags
    input_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_array)
    filter_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=filter_array)
    output_buf = cl.Buffer(context, mf.WRITE_ONLY, output_h * output_w * np.float32().nbytes)

    # Set kernel arguments
    kernel.set_arg(0, input_buf)
    kernel.set_arg(1, filter_buf)
    kernel.set_arg(2, output_buf)
    kernel.set_arg(3, np.int32(input_h))
    kernel.set_arg(4, np.int32(input_w))
    kernel.set_arg(5, np.int32(filter_h))
    kernel.set_arg(6, np.int32(filter_w))
    kernel.set_arg(7, np.int32(output_h))
    kernel.set_arg(8, np.int32(output_w))
    kernel.set_arg(9, np.int32(stride))

    # Execute the kernel
    global_size = (output_w, output_h)
    cl.enqueue_nd_range_kernel(queue, kernel, global_size, None)

    # Retrieve the output
    output_array = np.empty((output_h, output_w), dtype=np.float32)
    cl.enqueue_copy(queue, output_array, output_buf)
    return output_array

# Test function to compare outputs
def test_conv2d(aocx_file_path):
    # Define input and filter
    input_array = np.random.rand(8, 8).astype(np.float32)  # Input tensor
    filter_array = np.random.rand(3, 3).astype(np.float32)  # Filter tensor
    stride = 1

    # Load the OpenCL kernel
    context, queue, kernel = setup_opencl(aocx_file_path, "conv2d")

    # Compute output with OpenCL
    output_opencl = conv2d_opencl(context, queue, kernel, input_array, filter_array, stride)

    # Compute output with PyTorch
    input_tensor = torch.tensor(input_array).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    filter_tensor = torch.tensor(filter_array).unsqueeze(0).unsqueeze(0)  # Add in/out channel dimensions
    conv2d_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=stride, bias=False)
    conv2d_layer.weight.data = filter_tensor
    output_torch = conv2d_layer(input_tensor)
    output_torch = output_torch.squeeze().detach().numpy()

    # Compare outputs
    print("Input Array:\n", input_array)
    print("Filter Array:\n", filter_array)
    print("Output (OpenCL):\n", output_opencl)
    print("Output (PyTorch):\n", output_torch)
    print("Difference:\n", np.abs(output_opencl - output_torch))

# Run the test
if __name__ == "__main__":
    aocx_file_path = "./conv2d/conv2d.aocx"  # Path to the precompiled OpenCL binary
    test_conv2d(aocx_file_path)

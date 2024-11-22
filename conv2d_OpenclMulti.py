import pyopencl as cl
import numpy as np

class Conv2DOpenCL:
    def __init__(self, aocx_file_path='./conv2d/conv2d.aocx', device_id=2):
        """
        Initialize the Conv2D OpenCL module with a precompiled .aocx file.

        Args:
            aocx_file_path (str): Path to the precompiled .aocx file.
            device_id (int): Device ID for OpenCL platform.
        """
        self.aocx_file_path = aocx_file_path
        self.device_id = device_id
        self.context, self.queue, self.kernel = self._setup_opencl()

    def _setup_opencl(self):
        """
        Set up the OpenCL context, command queue, and kernel.

        Returns:
            context, queue, kernel: OpenCL context, command queue, and kernel object.
        """
        platform = cl.get_platforms()[self.device_id]
        devices = platform.get_devices(device_type=cl.device_type.ACCELERATOR)
        device = devices[0]
        context = cl.Context([device])
        queue = cl.CommandQueue(context)

        # Load the precompiled kernel
        with open(self.aocx_file_path, 'rb') as f:
            aocx_data = f.read()

        program = cl.Program(context, [device], [aocx_data]).build()
        kernel = cl.Kernel(program, "conv2d")

        return context, queue, kernel

    def execute(self, input_array, filters, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Perform 2D convolution using the FPGA-accelerated OpenCL kernel.

        Args:
            kernel_size (int or tuple): Size of the convolutional kernel.
            input_array (np.ndarray): Input array of shape (in_channels, height, width).
            filters (np.ndarray): Filter array of shape (out_channels, in_channels, kernel_h, kernel_w).
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride of the convolution. Default: 1.
            padding (int): Padding added to the input. Default: 0.

        Returns:
            np.ndarray: Output array of shape (out_channels, output_h, output_w).
        """
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        _, in_channels, input_h, input_w = input_array.shape
        out_channels, _, filter_h, filter_w = filters.shape

        # Apply padding
        padded_h = input_h + 2 * padding
        padded_w = input_w + 2 * padding
        padded_input = np.zeros((in_channels, padded_h, padded_w), dtype=np.float32)
        padded_input[:, padding:padding + input_h, padding:padding + input_w] = input_array

        # Calculate output dimensions
        output_h = (padded_h - filter_h) // stride + 1
        output_w = (padded_w - filter_w) // stride + 1

        # Initialize output array
        output_array = np.zeros((out_channels, output_h, output_w), dtype=np.float32)

        # Perform convolution for each output channel
        for out_c in range(out_channels):
            result = np.zeros((output_h, output_w), dtype=np.float32)
            for in_c in range(in_channels):
                result += self._conv2d_single_channel(
                    padded_input[in_c],  # Single channel input
                    filters[out_c, in_c],  # Single channel filter
                    stride
                )
            output_array[out_c] = result

        return output_array

    def _conv2d_single_channel(self, input_channel, filter_channel, stride):
        """
        Perform convolution for a single input and filter channel using OpenCL.

        Args:
            input_channel (np.ndarray): Single input channel.
            filter_channel (np.ndarray): Single filter channel.

        Returns:
            np.ndarray: Output array for the single channel.
        """
        input_h, input_w = input_channel.shape
        filter_h, filter_w = filter_channel.shape
        output_h = (input_h - filter_h) // stride + 1
        output_w = (input_w - filter_w) // stride + 1

        # Create OpenCL buffers
        mf = cl.mem_flags
        input_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_channel)
        filter_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=filter_channel)
        output_buf = cl.Buffer(self.context, mf.WRITE_ONLY, output_h * output_w * np.float32().nbytes)

        # Set kernel arguments
        self.kernel.set_arg(0, input_buf)
        self.kernel.set_arg(1, filter_buf)
        self.kernel.set_arg(2, output_buf)
        self.kernel.set_arg(3, np.int32(input_h))
        self.kernel.set_arg(4, np.int32(input_w))
        self.kernel.set_arg(5, np.int32(filter_h))
        self.kernel.set_arg(6, np.int32(filter_w))
        self.kernel.set_arg(7, np.int32(output_h))
        self.kernel.set_arg(8, np.int32(output_w))
        self.kernel.set_arg(9, np.int32(stride))

        # Execute the kernel
        global_size = (output_w, output_h)
        cl.enqueue_nd_range_kernel(self.queue, self.kernel, global_size, None)

        # Retrieve the output
        output_array = np.empty((output_h, output_w), dtype=np.float32)
        cl.enqueue_copy(self.queue, output_array, output_buf)
        return output_array

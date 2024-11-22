import pyopencl as cl
import numpy as np

class Conv2DOpenCL:
    def __init__(self, aocx_file_path='./conv2d/conv2d.aocx', device_id=2 ):
        """
        Initialize the Conv2D OpenCL module with a precompiled .aocx file.

        Args:
            aocx_file_path (str): Path to the precompiled .aocx file.
        """
        self.device_id = device_id
        self.aocx_file_path = aocx_file_path
        self.context, self.queue, self.kernel = self._setup_opencl()

    def _setup_opencl(self):
        """
        Set up the OpenCL context, command queue, and kernel.

        Returns:
            context, queue, kernel: OpenCL context, command queue, and kernel object.
        """
        # Select the first platform and accelerator device
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

    def execute(self, input_array, filter_array, stride=1):
        """
        Perform 2D convolution using the FPGA-accelerated OpenCL kernel.

        Args:
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
        input_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_array)
        filter_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=filter_array)
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

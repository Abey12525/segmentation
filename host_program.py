import pyopencl as cl
import numpy as np

# OpenCL kernel code as a string (matching the given kernel)
kernel_source = """
__kernel void vector_add(__global const float* x, 
                         __global const float* y, 
                         __global float* restrict z)
{
    // Get index of the work item
    int index = get_global_id(0);

    // Add the vector elements
    z[index] = x[index] + y[index];
}
"""

def main():
    # Step 1: Set up OpenCL platform and device
    platform = cl.get_platforms()[2]  # Get the first platform (can be customized)
    devices = platform.get_devices(device_type=cl.device_type.ACCELERATOR)
    device = devices[0]  # Select the first available FPGA accelerator device

    # Step 2: Create OpenCL context and queue
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # Step 3: Create buffers for input and output data
    N = 10
    x = np.ones(N, dtype=np.float32)  # Input array x (all ones)
    y = np.full(N, 3.0, dtype=np.float32)  # Input array y (all twos)
    z = np.zeros(N, dtype=np.float32)  # Output array z (to store the result)

    # Step 4: Create memory buffers for inputs and outputs
    buffer_x = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)
    buffer_y = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=y)
    buffer_z = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, z.nbytes)

    aocx_file_path = "./vector_add1.aocx"  # Path to the .aocx file
    with open(aocx_file_path, 'rb') as f:
        aocx_data = f.read()

    # Step 5: Load the kernel and compile it
    program = cl.Program(context,[device], [aocx_data]).build()
    kernel = cl.Kernel(program, "vector_add")

    # Step 6: Set kernel arguments
    kernel.set_arg(0, buffer_x)
    kernel.set_arg(1, buffer_y)
    kernel.set_arg(2, buffer_z)

    # Step 7: Execute the kernel
    global_size = (N,)
    local_size = None  # Let OpenCL choose local work size
    cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size).wait()

    # Step 8: Read the result back from the FPGA
    cl.enqueue_copy(queue, z, buffer_z).wait()

    # Step 9: Display the result
    print("Input array x:", x)
    print("Input array y:", y)
    print("Result array z:", z)

if __name__ == "__main__":
    main()

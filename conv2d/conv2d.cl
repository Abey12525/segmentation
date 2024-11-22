__kernel void conv2d(__global const float* input, 
                     __global const float* filter, 
                     __global float* output, 
                     const int input_h, const int input_w, 
                     const int filter_h, const int filter_w,
                     const int output_h, const int output_w,
                     const int stride) {
    int out_x = get_global_id(0);
    int out_y = get_global_id(1);

    if (out_x < output_w && out_y < output_h) {
        float value = 0.0f;

        for (int fy = 0; fy < filter_h; fy++) {
            for (int fx = 0; fx < filter_w; fx++) {
                int in_x = out_x * stride + fx;
                int in_y = out_y * stride + fy;

                if (in_x < input_w && in_y < input_h) {
                    value += input[in_y * input_w + in_x] * filter[fy * filter_w + fx];
                }
            }
        }
        output[out_y * output_w + out_x] = value;
    }
}

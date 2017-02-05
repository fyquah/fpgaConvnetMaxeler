/* TODO(fyq14) : Use C++ protobuf instead of this.  */

static void generic_load(std::string filename, int count, double **output)
{
    std::ifstream fin(filename.c_str());
    for (int i = 0 ; i < count ; i++) {
        fin >> output[i];
    }
    fin.close();
}


namespace convnet {

void *load_conv_kernels(
        std::string filename, conv_layer_t layer, double **output)
{
    generic_load(filename, N, output);
}


void *load_conv_bias(
        std::string filename, conv_layer_t layer, double **output)
{
    generic_load(filename, total_kernel_weights(layer), output);
}


uint32_t total_kernel_weights(const conv_layer_t & layer);
{
    return layer.kernel_size * layer.kernel_size
            * layer.num_inputs * layer.num_outputs;
}

}

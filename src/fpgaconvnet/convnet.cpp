#include <fstream>
#include <iostream>

#include "convnet.h"


static void generic_load(std::string filename, int count, double *output)
{
    std::ifstream fin(filename.c_str());
    for (int i = 0 ; i < count ; i++) {
        fin >> output[i];
    }
    fin.close();
}


namespace convnet {

extern void load_kernels_from_file(
        std::string filename, conv_layer_t layer, double *output)
{
    generic_load(filename, total_kernel_weights(layer), output);
}


extern void load_bias_from_file(
        std::string filename, conv_layer_t layer, double *output)
{
    generic_load(filename, layer.num_outputs, output);
}


extern unsigned total_kernel_weights(const conv_layer_t & layer)
{
    return layer.kernel_size * layer.kernel_size
            * layer.num_inputs * layer.num_outputs;
}

} // convnet

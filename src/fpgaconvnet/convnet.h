#include <iostream>
#include <string>

namespace convnet {

struct conv_layer_t {
    uint32_t kernel_size;
    uint32_t num_inputs;
    uint32_t num_outputs;
};

void load_kernels_from_file(std::string filename, conv_layer_t, double **);
void load_bias_from_file(std::string filename, conv_layer_t, double **);
uint32_t total_kernel_weights(const conv_layer_t & layer);

}

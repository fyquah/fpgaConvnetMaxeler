#include <string>

namespace convnet {

/* TODO(fyq14) : Use C++ protobuf instead of this.  */
struct conv_layer_t {
    unsigned kernel_size;
    unsigned num_inputs;
    unsigned num_outputs;
};

extern void load_kernels_from_file(
        std::string filename, conv_layer_t layer, double *output);
extern void load_bias_from_file(
        std::string filename, conv_layer_t layer, double *output);
extern unsigned total_kernel_weights(const conv_layer_t & layer);

} // convnet

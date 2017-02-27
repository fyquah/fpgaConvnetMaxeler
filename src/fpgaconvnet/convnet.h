#ifndef CONVNET_H
#define CONVNET_H

#include <fcntl.h>
#include <vector>

#include "MaxSLiCInterface.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "fpgaconvnet/protos/parameters.pb.h"


namespace fpgaconvnet {


protos::Network load_network_proto(const std::string & filename);


void load_kernels_from_file(
    std::string filename,
    const protos::LayerParameter & layer,
    double *output
);
void load_bias_from_file(
    std::string filename,
    const protos::LayerParameter & layer,
    double *output
);


void report_conv_performance(
        const protos::Network & network,
        uint64_t N,
        timeval t_begin,
        timeval t_end
);
void verify_conv_output(
        const protos::Network & network,
        uint64_t N,
        float *conv_out,
        std::string filename
);


/*
 * Reallign the kernels such that we have:
 * [kernel[c][0], kernel[c][convFactor], kernel[c][2 * convFactor] ...,
 *  kernel[c][1], kernel[c][convFactor + 1], kernel[c][2 * convFactor + 2], ...,
 *  ]
 *  akin reshape(kernels, (convFactors, -1)).T
 */
void allign_and_place_kernel_weights(
        const protos::LayerParameter & layer,
        double *dest_base,
        double *src_base
);

void max_set_layer_weights(
        max_actions_t *action,
        const protos::LayerParameter & layer,
        double *kernels,
        double *bias
);

void set_log_prefix(const std::string & prefix);

uint64_t calc_total_kernel_weights(const protos::LayerParameter & layer);


/* On most cases, this is the class abstracts most of the crazy reallignment
 * calls that is needed.
 */
class Convnet {
private:
    protos::Network network_params;
    std::vector<double*> kernels;
    std::vector<double*> worker_kernels;
    std::vector<double*> bias;
    std::vector<protos::LayerParameter> conv_layer_params;

    uint64_t input_size;
    uint64_t output_size;

    max_engine_t *dfe;
    max_file_t *max_file;

    Convnet(const Convnet &) {}
public:
    Convnet(const protos::Network & network_params,
            max_file_t *max_file,
            const char* load_spec = "*");
    ~Convnet ();

    void load_weights_from_files(std::vector<std::string> filenames);

    void max_init_weights();
    void max_load_input_data(const std::vector<float> & images, uint64_t N);
    std::vector<float> max_run_inference(
            uint64_t N, const std::vector<float> & images);
    std::vector<float> max_retrieve_features(uint64_t N);
};


} // fpgaconvnet

#endif

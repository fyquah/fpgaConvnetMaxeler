#ifndef CONVNET_H
#define CONVNET_H

#include <fcntl.h>
#include <vector>

#include "MaxSLiCInterface.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "fpgaconvnet/protos/parameters.pb.h"


namespace fpgaconvnet {

enum file_format_t {
    FORMAT_TXT,
    FORMAT_BINARY
};

protos::Network load_network_proto(const std::string & filename);


void load_kernels_from_file(
    std::string filename,
    const protos::LayerParameter & layer,
    float *output
);
void load_bias_from_file(
    std::string filename,
    const protos::LayerParameter & layer,
    float *output
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
        std::string filename,
        file_format_t file_format = FORMAT_TXT
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
        float *dest_base,
        float *src_base
);

void set_log_prefix(const std::string & prefix);

uint64_t calc_total_kernel_weights(const protos::LayerParameter & layer);


class Exception : public std::exception {
private:
    std::string message;
public:
    Exception(const std::string & message);
    virtual ~Exception() throw();
    virtual const char* what() const throw();
};


/* On most cases, this is the class abstracts most of the crazy reallignment
 * calls that is needed.
 */
class Convnet {
private:
    protos::Network network_params;
    std::vector<float*> kernels;
    std::vector<float*> worker_kernels;
    std::vector<float*> bias;
    std::vector<float*> queue_weights;
    std::vector<protos::LayerParameter> conv_layer_params;

    uint64_t input_size;
    uint64_t output_size;
    bool initialized_weights;

    max_engine_t *dfe;
    max_file_t *max_file;

    Convnet(const Convnet &) {}
    void set_layer_weights(
        max_actions_t *action,
        const protos::LayerParameter & layer,
        float *kernels,
        float *bias
    );

public:
    Convnet(const protos::Network & network_params,
            max_file_t *max_file,
            const char* load_spec = "*");
    ~Convnet ();

    void load_weights_from_files(
            std::vector<std::string> filenames, file_format_t file_type);
    void randomize_weights();

    void max_init_weights();
    void max_load_input_data(const std::vector<float> & images, uint64_t N);
    std::vector<float> max_run_inference(
            uint64_t N, const std::vector<float> & images, const bool benchmark);
    std::vector<float> max_retrieve_features(uint64_t N);
};


} // fpgaconvnet

#endif

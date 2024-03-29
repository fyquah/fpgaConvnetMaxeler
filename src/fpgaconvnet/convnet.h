#ifndef CONVNET_H
#define CONVNET_H

#include <fcntl.h>
#include <vector>
#include <utility>

#include "MaxSLiCInterface.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "fpgaconvnet/protos/parameters.pb.h"
#include "fpgaconvnet/common.h"


namespace fpgaconvnet {

enum file_format_t {
    FORMAT_TXT,
    FORMAT_BINARY
};

protos::Network load_network_proto(const std::string & filename);


void load_float_array_from_binary_file(
    std::string filename,
    const int size,
    float *output);

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
uint64_t calc_conv_in_size(const protos::Network & network);


/* On most cases, this is the class abstracts most of the crazy reallignment
 * calls that is needed.
 */
class Convnet {
private:
    protos::Network network_params;
    std::vector<float*> kernels;
    std::vector<fixed_point_t*> worker_kernels;
    std::vector<fixed_point_t*> bias;
    std::vector<fixed_point_t*> queue_weights;

    const char *load_spec;

    max_engine_t *dfe;          /* Used only when the bitstream uses  1 fpga */
    max_engarray_t *dfe_array;  /* Used only when the bitstream uses >1 fpga */
    std::vector<std::vector<max_file_t *> > max_files;
    max_file_t* lmem_maxfile;
    std::map<std::pair<int, int>, int> fpga_input_size;
    std::map<std::pair<int, int>, int> fpga_output_size;
    int m_last_executed_bitstream;

    Convnet(const Convnet &) {}
    void set_layer_weights(
        max_actions_t *action,
        const protos::LayerParameter & layer,
        fixed_point_t *kernels,
        fixed_point_t *bias
    );
    void constructor(
            const protos::Network & network_params,
            std::vector<std::vector<max_file_t*>> max_files,
            const char* load_spec = "*");

    /* Returns a vector denoting the range of layers served by each binary. */
    std::vector<std::pair<int, int>> get_range_list();

    uint64_t get_address_byte_offset(uint64_t N);
    uint64_t get_input_address_for_bitstream(unsigned bitstream, uint64_t N);
    uint64_t get_input_stream_size_for_bitstream(unsigned bitstream, uint64_t N);
    uint64_t get_output_address_for_bitstream(unsigned bitstream, uint64_t N);
    uint64_t get_output_stream_size_for_bitstream(unsigned bitstream, uint64_t N);
    unsigned get_num_fpga_for_bitstream(unsigned bitstream);
    unsigned get_num_bitstreams();

    void max_run_single_bitstream(
            uint64_t N, unsigned bitstream_id, double *p_timetaken,
            const void *input, void *output);

public:
    Convnet(const protos::Network & network_params,
            max_file_t* max_files,
            const char* load_spec);
    Convnet(const protos::Network & network_params,
            std::vector<max_file_t*> max_files,
            const char* load_spec);
    Convnet(const protos::Network & network_params,
            std::vector<std::vector<max_file_t*>> max_files,
            const char* load_spec);
    ~Convnet ();

    void load_weights_from_files(
            std::vector<std::string> filenames, file_format_t file_type);
    void randomize_weights();
    void max_init_weights();

    void max_write_to_lmem(
            const unsigned dfe_index,
            const void *data,
            const uint64_t addr,
            const uint64_t num_bytes);
    void max_read_from_lmem(
            const unsigned dfe_index,
            void *data,
            const uint64_t addr,
            const uint64_t num_bytes);

    void max_load_input_data(const float * images, uint64_t N);
    void max_read_output_data(float *images, uint64_t N);

    std::vector<float> max_run_inference(
            uint64_t N, const std::vector<float> & images, const bool benchmark);
    std::vector<float> max_run_inference(
            uint64_t N, const std::vector<float> & images,
            const bool benchmark, double * p_time_taken);

    std::vector<float> max_retrieve_features(uint64_t N);

    /// Runs inference using only part of the bitstreams. Helps debugging.
    /** Debugging utilities:
     *
     * [max_run_inference_with_single_bitstream]
     * */
    std::vector<float> max_run_inference_with_single_bitstream(
            uint64_t N,
            const std::vector<float> & images,
            const unsigned bitstream_id);

};

void dump_latencies(std::string filename, std::vector<double> times);


} // fpgaconvnet

#endif

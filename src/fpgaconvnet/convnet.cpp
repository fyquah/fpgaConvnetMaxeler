#include <cmath>
#include <cstring>
#include <exception>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/time.h>

#include <Eigen/Dense>

#include "fpgaconvnet/common.h"
#include "fpgaconvnet/convnet.h"
#include "fpgaconvnet/protos/parameters.pb.h"

// Generated from compiling maxfile for interacting with LMem
#include "lmem.h"

static void generic_load(std::string filename, int count, float *output)
{
    std::ifstream fin(filename.c_str());

    if (!fin.is_open()) {
        throw fpgaconvnet::Exception("Cannot open " + filename);
    }

    for (int i = 0 ; i < count ; i++) {
        fin >> output[i];
    }
    fin.close();
}


static void generic_load_binary(std::string filename, int count, float *output)
{
    std::ifstream fin(filename.c_str());

    for (int i = 0 ; i < count ; i++) {
        // note, works only on little endian machines.
        fin.read((char*) &output[i], 4);
    }
    fin.close();
}

namespace fpgaconvnet {


void set_log_prefix(const std::string & prefix)
{
    logging::log_prefix(prefix);
}

uint64_t calc_conv_in_size(const protos::Network & network)
{
    return calculation::conv_in_size(network);
}


uint64_t total_rom_size(const protos::LayerParameter & layer)
{
    return calculation::total_rom_size(layer);
}


static inline uint16_t
cast_float_to_fixed(const float arg) {

    const int num_frac_bits = 12;
    // const int num_int_bits = 4;
    const float fixed_point_one = (1 << num_frac_bits);

    int x = (int) (arg * fixed_point_one);

    /* GCC gurantees that a right shift is a ASR rather than a LSR
     * for signed integers.
     */
    uint16_t int_bits = (uint16_t) (x >> num_frac_bits);
    uint16_t frac_bits = (x & ((1 << num_frac_bits) - 1));

    return ((uint16_t) (int_bits << num_frac_bits)) | (frac_bits);
}


static void copy_float_to_fixed(fixed_point_t *dest, float *src, int size)
{
    for (int i = 0 ; i < size ; i++) {
        dest[i] = cast_float_to_fixed(src[i]);
    }
}


/* Utilities for loading weights into C-arrrays. */
void load_float_array_from_binary_file(
    std::string filename,
    const int size,
    float *output)
{
    generic_load_binary(filename, size, output);
}


void load_kernels_from_file(
    std::string filename,
    const protos::LayerParameter & layer,
    float *output
)
{
    generic_load(filename, calculation::total_kernel_weights(layer), output);
}


void load_bias_from_file(
    std::string filename,
    const protos::LayerParameter & layer,
    float *output
)
{
    generic_load(filename, layer.num_outputs(), output);
}


void load_kernels_from_binary_file(
    std::string filename,
    const protos::LayerParameter & layer,
    float *output
)
{
    std::ifstream fin(filename.c_str());
    int total_kernel_size = layer.conv().kernel_size() * layer.conv().kernel_size();
    int total_kernels = layer.num_outputs() / layer.conv().group() * layer.num_inputs();

    if (!fin.is_open()) {
        throw fpgaconvnet::Exception("Cannot open " + filename);
    }

    for (int i = 0 ; i < total_kernels ; i++) {
        // note, works only on little endian machines.
        fin.read((char*) &output[i * total_kernel_size], total_kernel_size * sizeof(float));
    }
    fin.close();
}


void load_bias_from_binary_file(
    std::string filename,
    const protos::LayerParameter & layer,
    float *output
)
{
    generic_load_binary(filename, layer.num_outputs(), output);
}


void report_conv_performance(const protos::Network & network, uint64_t N, double delta)
{
    double throughput = double(N) / delta * 1000000;
    double total_ops = calculation::ops(network);
   
    logging::stdout(logging::INFO)
            << "Time taken for " << N << " feature extractions  = "
            << delta << "microseconds" << std::endl;
    logging::stdout(logging::INFO)
            << "Actual Throughput (images per second) = " << throughput << std::endl;
    logging::stdout(logging::INFO)
            << "Actual GOps = " << throughput * total_ops / 1e9 << std::endl;
    calculation::explain_throughput(network);
}


void verify_conv_output(
        const protos::Network & network,
        uint64_t N,
        float *conv_out,
        std::string filename,
        file_format_t format)
{
    std::ifstream fin(filename.c_str());
    uint32_t total_pixels = 0;
    float total_error = 0.0;
    float total_pixel_magnitude = 0.0;

    const protos::LayerParameter & final_layer = network.layer(network.layer_size() - 1);
    const unsigned conv_out_size = (final_layer.output_height() *
                               final_layer.output_width() *
                               final_layer.num_outputs());
    uint64_t ctr = 0;

    for (uint32_t i = 0 ; i < std::min(N, 10ul) ; i++) {
        std::cout << "verify_conv_output IMAGE_NUMBER : " << i << std::endl;
        for (uint32_t j = 0 ; j < conv_out_size; j++) {
            float expected = 0.0;
            float obtained = conv_out[conv_out_size * i + j];

            if (format == FORMAT_TXT) {
                fin >> expected;
            } else {
                fin.read((char*) &expected, 4);
            }

            if (fin.eof()) {
                fin.clear();
                logging::stdout(logging::WARNING)
                    << "Verifier terminated early!" << std::endl;
                break;
            }

            total_error += std::abs(obtained  - expected);
            total_pixel_magnitude += std::abs(obtained);
            total_pixels += 1;

            if (isnan(obtained) || std::abs(obtained - expected) > 0.01) {
                ctr++;
                logging::stdout(logging::DEBUG) << j << "\t| BAD: Obtained " << obtained << ", expected " << expected << std::endl;
            }
            // else {
            //     logging::stdout(WARNING) << j << "\t| OKAY: Obtained " << obtained << ", expected " << expected << std::endl;
            // }
        }
    }
    logging::stdout(logging::INFO)
        << "Average pixel_error = "
        << float(total_error) / float(total_pixels) << std::endl;
    logging::stdout(logging::WARNING)
        << "Average pixel magnitude = "
        << float(total_pixel_magnitude) / float(total_pixels) << std::endl;
    logging::stdout(logging::WARNING)
        << "Number of pixels with error > 0.01 = " << ctr 
        << " out of "
        << total_pixels << std::endl;
    fin.close();
}

/* Don't even ask me how this works. This is dark magic to me by now. */
void special_allign_and_place_kernel_weights(
        const protos::LayerParameter & layer,
        float *dest_base,
        float *src_base
)
{
    const uint64_t conv_ff = layer.conv().conv_folding_factor();
    const uint64_t kernel_dim = layer.conv().kernel_size();
    const uint64_t worker_factor = layer.conv().worker_factor();
    const uint64_t rom_per_worker =
            calculation::total_rom_size(layer) / layer.conv().worker_factor();

    /* This for loop makes dest_base into
     *
     * wf * scheduler_iterations * cff * kff
     */
    for (uint64_t i = 0 ; i < worker_factor ; i++) {
        float *dest = dest_base + (i * rom_per_worker);
        float *src = src_base + (i * kernel_dim * kernel_dim);

        for (uint64_t w = 0; w < calculation::scheduler_iterations(layer); w++) {
            const int worker_iter = w;  // the w-th channel that the worker's handling.

            if (i + worker_iter * worker_factor >= layer.num_inputs()) {
                continue;
            }

            for (uint64_t channel = 0
                    ; channel < layer.num_outputs() / layer.conv().group()
                    ; channel++) {
                const int k =
                        (worker_iter * worker_factor
                         / (layer.num_inputs() / layer.conv().group()));
                const int x =
                        (worker_iter * worker_factor)
                        % (layer.num_inputs() / layer.conv().group());
                const int src_offset =
                        (((k * layer.num_outputs() / layer.conv().group()
                           + channel)
                           * (layer.num_inputs() / layer.conv().group()))
                          + x)
                        * kernel_dim * kernel_dim;
                const int conv_iters = calculation::convolution_iterations(layer);
                const int scheduler_iters = calculation::scheduler_iterations(layer);
                const int dest_offset =
                        ((worker_iter * conv_iters)
                         + ((channel % conv_ff) * conv_iters * scheduler_iters)
                         + (channel / conv_ff))
                        * layer.conv().kernel_folding_factor()
                        * calculation::kernel_iterations(layer);

                /*
                 * Useful piece of code to visualize the kernels weights
                 *    -> worker position mapping.
                std::cout << "src_offset = " 
                        << src_offset / (kernel_dim * kernel_dim)
                        << " -> "
                        << dest_offset / (kernel_dim * kernel_dim)
                        << "; " << *(src + src_offset)
                        << "; " << *(src + src_offset + 1)
                        << "; " << *(src + src_offset + 2)
                        << std::endl;
                */
                std::memcpy(
                        dest + dest_offset,
                        src + src_offset,
                        sizeof(float) * kernel_dim * kernel_dim);
            }
        }
    }
}

void allign_and_place_cpu_initialized_kernel_weights(
        const protos::LayerParameter & layer,
        fixed_point_t *dest_base,
        float *src_base
)
{
    const uint64_t total_rom_size = calculation::total_rom_size(layer);
    float *tmp = new float[total_rom_size];
    const uint64_t rom_per_worker =
            total_rom_size / layer.conv().worker_factor();
    const uint64_t rom_per_conv =
            rom_per_worker / layer.conv().conv_folding_factor();
    const uint64_t total_iterations = calculation::total_iterations(layer);

    special_allign_and_place_kernel_weights(
            layer,
            tmp,
            src_base);

    for (unsigned iter = 0 ; iter < total_iterations ; iter++) {
        uint64_t addr = iter * calculation::total_multipliers(layer);

        for (unsigned worker = 0 ; worker < layer.conv().worker_factor() ; worker++) {
            for (unsigned conv = 0 ; conv < layer.conv().conv_folding_factor() ; conv++) {
                uint64_t offset =
                        (iter * layer.conv().kernel_folding_factor())
                        + (worker * rom_per_worker)
                        + (conv * rom_per_conv);

                /*
                 * Some useful debugging code
                std::cout
                    << "Worker = " << worker
                    << " conv = " << conv
                    << " iter = " << iter
                    << " offset = " << offset
                    << " addr = " << addr
                    << " ; " << *(tmp + offset)
                    << " ; " << *(tmp + offset + 1)
                    << " ; " << *(tmp + offset + 2)
                    << std::endl;
                */

                copy_float_to_fixed(
                        dest_base + addr,
                        tmp + offset,
                        layer.conv().kernel_folding_factor());
                addr += layer.conv().kernel_folding_factor();

            }
        }
    }

    delete[] tmp;
}


void allign_and_place_lmem_initialized_kernel_weights(
        const protos::LayerParameter & layer,
        fixed_point_t *dest_base,
        float *src_base
)
{
    const uint64_t total_rom_size = calculation::total_rom_size(layer);
    float *tmp = new float[total_rom_size];
    const uint64_t rom_per_worker =
            total_rom_size / layer.conv().worker_factor();
    const uint64_t rom_per_conv =
            rom_per_worker / layer.conv().conv_folding_factor();
    const uint64_t total_iterations = calculation::total_iterations(layer);

    special_allign_and_place_kernel_weights(
            layer,
            tmp,
            src_base);

    for (uint64_t iter = 0 ; iter < total_iterations ; iter++) {
        uint64_t addr = iter * calculation::weights_vector_size(layer);

        for (unsigned worker = 0 ; worker < layer.conv().worker_factor() ; worker++) {
            for (unsigned conv = 0 ; conv < layer.conv().conv_folding_factor() ; conv++) {
                uint64_t offset =
                        (iter * layer.conv().kernel_folding_factor())
                        + (worker * rom_per_worker)
                        + (conv * rom_per_conv);

                copy_float_to_fixed(
                        dest_base + addr,
                        tmp + offset,
                        layer.conv().kernel_folding_factor());
                addr += layer.conv().kernel_folding_factor();
            }
        }
    }

    delete[] tmp;
}

void Convnet::randomize_weights()
{
    for (unsigned i = 0 ; i < kernels.size() ; i++) {
        if (kernels[i] == NULL) {
            continue;
        }

        auto conv_layer = network_params.layer(i);
        uint64_t total_weights = calculation::total_kernel_weights(conv_layer);
        uint64_t total_bias = conv_layer.num_outputs();

        for (unsigned j = 0 ; j < total_weights; j++) {
            kernels[i][j] = (float) math::rng(-0.75, 0.75);
        }

        for (unsigned j = 0 ; j < total_bias ; j++) {
            bias[i][j] = (float) math::rng(-0.75, 0.75);
        }
    }
}


void Convnet::set_layer_weights(
        max_actions_t *action,
        const protos::LayerParameter & layer,
        fixed_point_t *worker_kernels,
        fixed_point_t *bias
)
{
    /* Requirements:
     * - stream to be a multiple of 16 bytes = 4 * 4 floats
     * - As input vector is of size layer.conv().kernel_folding_factor(), let's call this number
     *   kff.
     */
    char buffer[30];
    sprintf(buffer, "kernel_%d", layer.layer_id());

    if (calculation::is_layer_cpu_initialized(layer)) {
        uint64_t stream_size = calculation::cpu_weights_stream_size(layer);
        uint16_t *values = new uint16_t[stream_size];

        queue_weights.push_back(values);
        std::memcpy(
                values,
                worker_kernels,
                sizeof(fixed_point_t) * stream_size);
        max_queue_input(
                action, buffer, values,
                sizeof(fixed_point_t) * stream_size);

    }

    sprintf(buffer, "bias_%d", layer.layer_id());
    max_queue_input(
            action, buffer, bias,
            sizeof(fixed_point_t) * calculation::bias_stream_size(layer));
}

Convnet::Convnet(
        const protos::Network & network_params,
        std::vector<std::vector<max_file_t*> > max_files,
        const char* load_spec)
{
    constructor(network_params, max_files, load_spec);
}

Convnet::Convnet(
        const protos::Network & network_params,
        std::vector<max_file_t*> max_files,
        const char* load_spec)
{
    std::vector<std::vector<max_file_t*> > b_max_files;
    b_max_files.push_back(max_files);
    constructor(network_params, b_max_files, load_spec);
}


Convnet::Convnet(
        const protos::Network & network_params,
        max_file_t* max_file,
        const char* load_spec)
{
    std::vector<std::vector<max_file_t*> > max_files(1);
    max_files.back().push_back(max_file);
    constructor(network_params, max_files, load_spec);
}

unsigned Convnet::get_num_fpga_for_bitstream(unsigned bitstream)
{
    return max_files[bitstream].size();
}

unsigned Convnet::get_num_bitstreams()
{
    return max_files.size();
}

void Convnet::constructor(
        const protos::Network & arg_network_params,
        std::vector<std::vector<max_file_t*>> arg_max_files,
        const char* arg_load_spec)
{
    network_params = arg_network_params;
    max_files = arg_max_files;
    load_spec = arg_load_spec;
    dfe = NULL;
    dfe_array = NULL;
    lmem_maxfile = lmem_init();
    m_last_executed_bitstream = -1;

    for (auto it = network_params.layer().begin();
            it != network_params.layer().end();
            it++) {
        if (it->has_conv()) {
            uint64_t worker_kernel_total_size =
                    calculation::weights_vector_size(*it)
                    * calculation::total_iterations(*it);
            uint64_t bias_total_size = calculation::bias_stream_size(*it);

            kernels.push_back(new float[calculation::total_kernel_weights(*it)]);
            bias.push_back(new fixed_point_t[bias_total_size]);
            worker_kernels.push_back(new fixed_point_t[worker_kernel_total_size]);
        } else {
            kernels.push_back(NULL);
            bias.push_back(NULL);
            worker_kernels.push_back(NULL);
        }

        // since the layers are traversed in order, this is guranteed to write
        // the correct results in the end
        fpga_output_size[std::make_pair(it->bitstream_id(), it->fpga_id())] =
                it->output_height() * it->output_width() * it->num_outputs();
    }

    for (auto it = network_params.layer().rbegin();
            it != network_params.layer().rend();
            it++) {
        fpga_input_size[std::make_pair(it->bitstream_id(), it->fpga_id())] =
                it->input_height() * it->input_width() * it->num_inputs();
    }
}

Convnet::~Convnet ()
{
    for (unsigned i = 0 ; i < kernels.size() ; i++) {
        if (kernels[i] != NULL) {
            delete[] kernels[i];
        }
        if (bias[i] != NULL) {
            delete[] bias[i];
        }
        if (worker_kernels[i] != NULL) {
            delete[] worker_kernels[i];
        }
    }
    for (unsigned i = 0 ; i < queue_weights.size(); i++) {
        delete[] queue_weights[i];
    }
    if (dfe) {
        max_unload(dfe);
    }
    if (dfe_array) {
        max_unload_array(dfe_array);
    }
}

void Convnet::load_weights_from_files(
        std::vector<std::string> filenames, file_format_t file_type)
{
    logging::stdout(logging::INFO) << "Loading weights from file." << std::endl;
    int j = 0;

    for (int i = 0 ; i < network_params.layer_size(); i++) {
        if (kernels[i] == NULL) {
            continue;
        }

        auto layer = network_params.layer(i);

        float *bias_tmp = new float[calculation::bias_stream_size(layer)];

        if (file_type == FORMAT_TXT) {
            load_kernels_from_file(
                    filenames[j++], layer, kernels[i]);
            load_bias_from_file(
                    filenames[j++], layer, bias_tmp);
        } else {
            load_kernels_from_binary_file(
                    filenames[j++], layer, kernels[i]);
            load_bias_from_binary_file(
                    filenames[j++], layer, bias_tmp);
        }

        copy_float_to_fixed(
                bias[i], bias_tmp, calculation::bias_stream_size(layer));

        delete[] bias_tmp;
    }
    logging::stdout(logging::INFO) << "Alligning weights." << std::endl;
    for (int i = 0 ; i < network_params.layer_size(); i++) {
        auto layer = network_params.layer(i);

        if (kernels[i] != NULL) {
            if (calculation::is_layer_cpu_initialized(layer)) {
                allign_and_place_cpu_initialized_kernel_weights(
                        layer, worker_kernels[i], kernels[i]);
            } else {
                allign_and_place_lmem_initialized_kernel_weights(
                        layer, worker_kernels[i], kernels[i]);
            }
        }
    }
    logging::stdout(logging::INFO) << "Done!" << std::endl;
}


void Convnet::max_init_weights()
{
    std::vector<int8_t*> buffer_ptrs;

    for (int i = 0; i < network_params.layer_size() ; i++) {
        if (kernels[i] == NULL) {
            continue;
        }

        auto layer = network_params.layer(i);

        if (calculation::is_layer_cpu_initialized(layer)) {
            logging::stdout(logging::INFO) << "layer "
                    << layer.layer_id()
                    << " is host-initialized. Skipping .." << std::endl;
            continue;
        }

        max_actions_t *write_action = max_actions_init(
                lmem_maxfile, "writeLMem");
        const uint64_t address =
                layer.conv().weight_address_base();
        const uint64_t stream_size =
                calculation::total_iterations(layer)
                * calculation::weights_vector_size(layer)
                * sizeof(fixed_point_t);

        logging::stdout(logging::INFO)
                << "Initializing weights in LMEM at layer "
                << layer.layer_id()
                << " [fpga_id = " << layer.fpga_id() << "]"
                << std::endl;
        logging::stdout(logging::INFO)
                << "Address = " << address << std::endl;
        logging::stdout(logging::INFO)
                << "Stream size (in bytes) = " << stream_size << std::endl;
        max_set_param_uint64t(write_action, "start", address);
        max_set_param_uint64t(write_action, "size", stream_size);
        max_queue_input(
                write_action,
                "data_in",
                (void*) &worker_kernels[i][0],
                stream_size);

        dfe = max_load(lmem_maxfile, load_spec);
        max_run(dfe, write_action);
        max_actions_free(write_action);
        max_unload(dfe);
        dfe = NULL;

        logging::stdout(logging::INFO) << "Done!" << std::endl;
    }

}

std::vector<float> Convnet::max_run_inference(
        uint64_t N,
        const std::vector<float> & images,
        const bool benchmark
)
{
    double p;
    return max_run_inference(N, images, benchmark, &p);
}


std::vector<std::pair<int, int>>
Convnet::get_range_list()
{
    std::vector<std::pair<int, int>> v;
    unsigned begin = 0;

    for (int i = 0; i < network_params.layer_size(); i++) {
        if (i == network_params.layer_size() - 1 ||
                network_params.layer(i).bitstream_id() != network_params.layer(i + 1).bitstream_id()) {
            v.push_back(std::make_pair(begin, i));
            begin = i + 1;
        }

    }

    return v;
}

/* TODO: This is a pessimistic estimate of the required offset. */
uint64_t
Convnet::get_address_byte_offset(uint64_t N)
{
    uint64_t x = 0;

    for (int i = 0; i < network_params.layer_size() ; i++) {
        uint64_t y, z;
        auto layer = network_params.layer(i);

        y = layer.input_height() * layer.input_width() * layer.num_inputs();
        if (i == 0) {
            y = y * sizeof(float);
        } else {
            y = y * sizeof(fixed_point_t);
        }

        z = layer.output_height() * layer.output_width() * layer.num_outputs();
        if (i == network_params.layer_size() - 1) {
            z = z * sizeof(float);
        } else {
            z = z * sizeof(fixed_point_t);
        }

        x = std::max(x, std::max(y, z));
    }

    if (x % 384 == 0) {
        return N * x;
    } else {
        return N * (x / 384 + 1) * 384;
    }
}

uint64_t BASE_OFFSET = 384 * 100;

uint64_t
Convnet::get_input_address_for_bitstream(unsigned bitstream, uint64_t N)
{
    if (bitstream % 2 == 0) {
        return BASE_OFFSET;
    } else {
        return BASE_OFFSET + get_address_byte_offset(N);
    }
}


uint64_t
Convnet::get_output_address_for_bitstream(unsigned bitstream, uint64_t N)
{
    if (bitstream % 2 == 1) {
        return BASE_OFFSET;
    } else {
        return BASE_OFFSET + get_address_byte_offset(N);
    }
}


uint64_t
Convnet::get_input_stream_size_for_bitstream(unsigned bitstream, uint64_t N)
{
    const unsigned idx = get_range_list()[bitstream].first;
    const auto layer = network_params.layer(idx);
    const unsigned num_values =
        layer.input_height() * layer.input_width() * layer.num_inputs();

    if (idx == 0) {
        return N * num_values * sizeof(float);
    } else {
        return N * num_values * sizeof(fixed_point_t);
    }
}


uint64_t
Convnet::get_output_stream_size_for_bitstream(unsigned bitstream, uint64_t N)
{
    const unsigned idx = get_range_list()[bitstream].second;
    const auto layer = network_params.layer(idx);
    const unsigned num_values =
        layer.output_height() * layer.output_width() * layer.num_outputs();

    if (int(idx) == network_params.layer_size() - 1) {
        return N * num_values * sizeof(float);
    } else {
        return N * num_values * sizeof(fixed_point_t);
    }
}


void
Convnet::max_load_input_data(const float *images, uint64_t N)
{
    const uint64_t stream_size = get_input_stream_size_for_bitstream(0, N);
    const uint64_t address = get_input_address_for_bitstream(0, N);

    logging::stdout(logging::INFO)
        << "Writing " << N << " images with stream size of "
        << stream_size << " bytes to address "
        << address
        << std::endl;

    dfe = max_load(lmem_maxfile, load_spec);
    max_actions_t *write_action = max_actions_init(lmem_maxfile, "writeLMem");
    max_set_param_uint64t(write_action, "start", address);
    max_set_param_uint64t(write_action, "size", stream_size);
    max_queue_input(
            write_action,
            "data_in",
            (void*) &images[0],
            stream_size);
    max_run(dfe, write_action);
    max_actions_free(write_action);
    max_unload(dfe);
    dfe = NULL;
    m_last_executed_bitstream = -1;
}

void
Convnet::max_read_output_data(float * images, uint64_t N)
{
    const uint64_t stream_size = get_output_stream_size_for_bitstream(
            get_num_bitstreams() - 1, N);
    const uint64_t address = get_output_address_for_bitstream(
            get_num_bitstreams() - 1, N);

    dfe = max_load(lmem_maxfile, load_spec);
    logging::stdout(logging::INFO)
        << "Reading " << N << " images with stream size of "
        << stream_size << " bytes from address "
        << address
        << std::endl;

    max_actions_t *read_action = max_actions_init(
            lmem_maxfile, "readLMem");
    max_set_param_uint64t(read_action, "start", address);
    max_set_param_uint64t(read_action, "size", stream_size);
    max_queue_output(
            read_action,
            "data_out",
            (void*) &images[0],
            stream_size);
    max_run(dfe, read_action);
    max_actions_free(read_action);
    max_unload(dfe);
    dfe = NULL;
    m_last_executed_bitstream = -1;
}

void
Convnet::max_run_single_bitstream(
        uint64_t N, unsigned bitstream_id, double *p_timetaken)
{
    const unsigned num_fpgas = get_num_fpga_for_bitstream(bitstream_id);
    const bool initialised_weights = m_last_executed_bitstream == int(bitstream_id);

    max_actions_t **actions = new max_actions_t*[num_fpgas];
    timeval t_begin;
    timeval t_end;
    t_begin.tv_sec  = 0.0;
    t_begin.tv_usec = 0.0;
    t_end.tv_sec  = 0.0;
    t_end.tv_usec = 0.0;

    for (unsigned i = 0 ; i < num_fpgas ; i++) {
        actions[i] = max_actions_init(max_files[bitstream_id][i], "default");
    }

    for (int i = 0; i < network_params.layer_size() ; i++) {
        auto it = &network_params.layer(i);
        auto layer = *it;

        if (it->bitstream_id() != bitstream_id) {
            continue;
        }

        if (it->has_conv()) {
            char buffer[30];
            sprintf(buffer, "kernel_%d", it->layer_id());
            max_actions_t *action = actions[it->fpga_id()];

            if (initialised_weights) {
                logging::stdout(logging::INFO)
                        << "Host-initialized weights has been set "
                        << "in previous calls."
                        << std::endl;
                if (calculation::is_layer_cpu_initialized(layer)) {
                    sprintf(buffer, "kernel_%d", layer.layer_id());
                    max_queue_input(action, buffer, NULL, 0);
                }
                sprintf(buffer, "bias_%d", layer.layer_id());
                max_queue_input(action, buffer, NULL, 0);

            } else {
                logging::stdout(logging::INFO)
                        << "Passing in host-initialized weights "
                        << "(This should only be done once)."
                        << buffer << std::endl;
                set_layer_weights(
                        action, *it, worker_kernels[i], bias[i]);
            }


        } else if (it->has_lrn()) {
            /* assuming binomial approximation used. */
            char buffer[30];
            const float beta = it->lrn().beta();
            const float alpha = it->lrn().alpha();
            const float k = it->lrn().k();
            const float local_size = it->lrn().local_size();

            sprintf(buffer, "approx_factor_%d", it->layer_id());
            max_set_param_double(
                    actions[it->fpga_id()],
                    buffer,
                    -beta * alpha / local_size);

            sprintf(buffer, "approx_left_%d", it->layer_id());
            max_set_param_double(
                    actions[it->fpga_id()],
                    buffer,
                    std::pow(k, -beta));
        }
    }

    for (unsigned i = 0 ; i < num_fpgas ; i++) {
        max_set_param_uint64t(actions[i], "N", N);

        if (initialised_weights) {
            max_set_param_uint64t(actions[i], "init", 0);
        } else {
            max_set_param_uint64t(actions[i], "init", 1);
        }
    }

    const uint64_t addressIn = get_input_address_for_bitstream(
            bitstream_id, N);
    const uint64_t addressOut = get_output_address_for_bitstream(
            bitstream_id, N);

    max_set_param_uint64t(actions[0], "addressIn", addressIn);
    max_set_param_uint64t(actions[num_fpgas - 1], "addressOut", addressOut);

    fpgaconvnet::logging::stdout(logging::INFO)
        << "input address = " << addressIn << "\n";
    fpgaconvnet::logging::stdout(logging::INFO)
        << "output address = " << addressOut << "\n";

#ifdef __SIM__
    void *tmp_buffer_in = NULL;
    void *tmp_buffer_out = NULL;

    for (unsigned i = 0; i < num_fpgas ; i++) {
        dfe = max_load(max_files[bitstream_id][i], load_spec);

        logging::stdout(logging::INFO) << "Simulating FPGA " << i << " ..." << std::endl;

        if (i > 0) {
            logging::stdout(logging::INFO) << "Mocking maxring input" << std::endl;
            max_queue_input(actions[i],
                            "mock_maxring_in",
                            tmp_buffer_in,
                            N * fpga_input_size[std::make_pair(bitstream_id, i)] * 2);
        }

        if (i < num_fpgas - 1) {
            logging::stdout(logging::INFO) << "Mocking maxring output" << std::endl;
            tmp_buffer_out = malloc(
                    N * fpga_output_size[std::make_pair(bitstream_id, i)] * 2);
            max_queue_output(actions[i],
                            "mock_maxring_out",
                            tmp_buffer_out,
                            N * fpga_output_size[std::make_pair(bitstream_id, i)] * 2);
        }
        max_run(dfe, actions[i]);
        max_unload(dfe);

        tmp_buffer_in = tmp_buffer_out;
        tmp_buffer_out = NULL;
    }
    dfe = NULL;

    if (tmp_buffer_out != NULL) {
        free(tmp_buffer_out);
    }
    if (tmp_buffer_in != NULL) {
        free(tmp_buffer_in);
    }
#else
    if (num_fpgas == 1) {
        __sync_synchronize();
        gettimeofday(&t_begin, NULL);
        dfe = max_load(max_files[bitstream_id][0], load_spec);
        gettimeofday(&t_end, NULL);
        __sync_synchronize();
        logging::stdout(logging::INFO)
            << "max_load took "
            << compute_time_difference(t_begin, t_end)
            << " microseconds\n";

        __sync_synchronize();
        gettimeofday(&t_begin, NULL);
        max_run(dfe, actions[0]);
        gettimeofday(&t_end, NULL);
        __sync_synchronize();
        max_unload(dfe);
        dfe = NULL;

    } else {
        dfe_array = max_load_mixed_array(
                (max_file_t**) &max_files[bitstream_id][0], num_fpgas, load_spec);
        max_actarray_t *act_array = max_mixed_actarray_init(
                &max_files[bitstream_id][0], num_fpgas);
        for (unsigned i = 0 ; i < num_fpgas ; i++) {
            max_set_action(act_array, i, actions[i]);
        }

        __sync_synchronize();
        gettimeofday(&t_begin, NULL);
        max_run_array(dfe_array, act_array);
        gettimeofday(&t_end, NULL);
        __sync_synchronize();

        max_unload_array(dfe_array);
    }
#endif

    delete[] actions;

    *p_timetaken = compute_time_difference(t_begin, t_end);
    m_last_executed_bitstream = bitstream_id;
}


std::vector<float> Convnet::max_run_inference(
        uint64_t N,
        const std::vector<float> & images,
        const bool benchmark,
        double *p_time_taken
)
{
    const auto last_layer = network_params.layer(
            network_params.layer_size() - 1);
    const uint64_t output_size =
            last_layer.output_height() * last_layer.output_width()
            * last_layer.num_outputs();
    std::vector<float> ret(N * output_size , 0);
    double time_taken = 0.0;

    /* 1. Load images into off-chip memory. */
    logging::stdout(logging::INFO)
        << "Loading images to off-chip memory ... " << std::endl;
    max_load_input_data(&images[0], N);
    logging::stdout(logging::INFO) << "-- DONE!" << std::endl;

    /* 2. Run inference */
    logging::stdout(logging::INFO)
        << "Running feature extractions ... " << std::endl;
    for (unsigned i = 0 ; i < get_num_bitstreams() ; i++) {
        logging::Indentation indent;

        logging::stdout(logging::INFO)
            << "Running bitstream " << i << " ... " << std::endl;

        logging::Indentation more_indent;
        double this_time_taken;
        max_run_single_bitstream(N, i, &this_time_taken);
        max_run_single_bitstream(N, i, &this_time_taken);

        time_taken += this_time_taken;
    }
    fpgaconvnet::logging::stdout() << "-- DONE" << std::endl;

    // 3. Load data from off-chip memory back to the host
    max_read_output_data(&ret[0], N);
    fpgaconvnet::logging::stdout() << "-- DONE" << std::endl;

    // 4. Report the results and write the time taken
    if (benchmark) {
        report_conv_performance(network_params, N, time_taken);
    }
    *p_time_taken = time_taken;
    return ret;
}

void
dump_latencies(std::string filename, std::vector<double> times)
{
    std::ofstream o(filename.c_str());
    o << "[";
    for (unsigned i = 0 ; i < times.size() ; i++) {
        o << times[i] << ", ";
    }
    o << "]";
    o.close();
}

} // fpgaconvnet

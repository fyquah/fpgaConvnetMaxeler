#include <cmath>
#include <cstring>
#include <exception>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/time.h>

#include <Eigen/Dense>

#include "fpgaconvnet/convnet.h"
#include "fpgaconvnet/protos/parameters.pb.h"


static double rng(const double lo, const double hi)
{
    return lo + (hi - lo) * ((double)rand()/(double)RAND_MAX);
}


static uint64_t gcd(uint64_t a, uint64_t b)
{
    if (b > a) {
        std::swap(a, b);
    }
    if (b == 0) {
        return a;
    } else {
        return gcd(b, a % b);
    }
}

static uint64_t lcm(uint64_t a, uint64_t b)
{
    return a / gcd(a, b) * b;
}


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

// Logging stuff
static const int DEBUG = 0;
static const int INFO = 1;
static const int WARNING = 2;
static const int ERROR = 3;
static const char* level_strings[] = {
        "DEBUG", "INFO", "WARNING", "ERROR"
};

static std::string LOG_PREFIX = "default";

static std::ostream& log_stdout(int level = INFO)
{
    if (level > 4) {
        level = 4;
    } else if (level < 0) {
        level = 0;
    }
    return std::cout << "[" << LOG_PREFIX << "\t "
            << level_strings[level] << "]\t";
}


void set_log_prefix(const std::string & prefix)
{
    LOG_PREFIX = prefix;
}

static uint64_t div_ceil(uint64_t a, uint64_t b)
{
    if (a % b == 0) {
        return a / b;
    } else {
        return a / b + 1;
    }
}


static uint64_t calc_total_multipliers(const protos::LayerParameter & layer)
{
    return layer.conv().worker_factor()
            * layer.conv().conv_folding_factor()
            * layer.conv().kernel_folding_factor();
}


/* The number of iterations to convolve a kernel with a sliding window. */
static uint64_t calc_kernel_iterations(const protos::LayerParameter & layer)
{
    uint64_t kernelDim = layer.conv().kernel_size();
    return div_ceil(kernelDim * kernelDim, layer.conv().kernel_folding_factor());
}

/* The number of convolution cycles to process a single sliding window. */
static uint64_t calc_convolution_iterations(const protos::LayerParameter & layer)
{
    return div_ceil(layer.num_outputs(),
                    layer.conv().conv_folding_factor()); 
}

/* The number of cycles to process all input channels of a sliding window. */ 
static uint64_t calc_scheduler_iterations(const protos::LayerParameter & layer)
{
    return div_ceil(layer.num_inputs(), layer.conv().worker_factor());
}


static uint64_t calc_total_iterations(const protos::LayerParameter &layer)
{
    return calc_scheduler_iterations(layer)
            * calc_convolution_iterations(layer) * calc_kernel_iterations(layer);
}


uint64_t calc_total_kernel_weights(const protos::LayerParameter & layer)
{
    if (!layer.has_conv()) {
        return 0;
    }
    auto & conv = layer.conv();
    return layer.num_inputs() * layer.num_outputs()
            * conv.kernel_size() * conv.kernel_size();
}


uint64_t calc_conv_in_size(const protos::Network & network)
{
    return network.layer(0).num_inputs()
	* network.layer(0).input_height()
	* network.layer(0).input_width();
}


uint64_t calc_total_rom_size(const protos::LayerParameter & layer)
{
    return layer.conv().worker_factor()
            * layer.conv().conv_folding_factor()
            * layer.conv().kernel_folding_factor()
            * calc_total_iterations(layer);
}


static uint64_t calc_weights_vector_size(
        const protos::LayerParameter & layer)
{
    int stream_chunk_size = 384 / sizeof(fixed_point_t);
    uint64_t weights_per_iter =
            layer.conv().worker_factor()
            * layer.conv().conv_folding_factor()
            * layer.conv().kernel_folding_factor();
    return div_ceil(weights_per_iter, stream_chunk_size) * stream_chunk_size;
}



static bool is_layer_cpu_initialized(const protos::LayerParameter & layer)
{
    return layer.conv().bram_factor()
        >= (layer.num_inputs() * layer.num_outputs());
}

static uint64_t calc_bias_stream_size(const protos::LayerParameter & layer)
{
    uint64_t stream_chunk_size = 16 / sizeof(fixed_point_t);
    return div_ceil(layer.num_outputs(), stream_chunk_size)
        * stream_chunk_size;
}

static uint64_t calc_cpu_weights_stream_size(
        const protos::LayerParameter & layer)
{
    uint64_t stream_chunk_size = 16 / sizeof(fixed_point_t);
    uint64_t multiple_base =
        lcm(stream_chunk_size, layer.conv().kernel_folding_factor())
        / layer.conv().kernel_folding_factor();

    uint64_t total_rom_size = calc_total_rom_size(layer);
    uint64_t num_iters =
        total_rom_size / layer.conv().kernel_folding_factor();
    uint64_t padded_num_iters =
        div_ceil(num_iters, multiple_base) * multiple_base;
    uint64_t padded_rom_size =
        padded_num_iters * layer.conv().kernel_folding_factor();

    return padded_rom_size;
}


static void copy_float_to_fixed(fixed_point_t *dest, float *src, int size)
{
    const int num_frac_bits = 12;
    const int num_int_bits = 4;
    const float fixed_point_one = (1 << num_frac_bits);

    for (int i = 0 ; i < size ; i++) {
        int x = (int) (src[i] * fixed_point_one);

        /* GCC gurantees that a right shift is a ASR rather than a LSR
         * for signed integers.
         */
        uint16_t int_bits = (uint16_t) (x >> num_frac_bits);
        uint16_t frac_bits = (x & ((1 << num_frac_bits) - 1));

        dest[i] =
            ((uint16_t) (int_bits << num_frac_bits))
            | (frac_bits);
    }
}


protos::Network load_network_proto(const std::string & filename)
{
    protos::Network network;
    int fd = open(filename.c_str(), O_RDONLY);

    google::protobuf::io::FileInputStream fstream(fd);
    google::protobuf::TextFormat::Parse(&fstream, &network);

    int i = 0;
    for (auto it = network.mutable_layer()->begin(); it != network.mutable_layer()->end() ; it++, i++) {
        if (it != network.mutable_layer()->begin()) {
            auto prev_it = it - 1;
            it->set_num_inputs(prev_it->num_outputs());
            it->set_input_height(prev_it->output_height());
            it->set_input_width(prev_it->output_width());
        }

        it->set_layer_id(i);
        if (it->has_conv()) {
            it->set_output_height(
                    (it->input_height() - it->conv().kernel_size() + 2 * it->conv().pad())
                    / it->conv().stride() + 1);
            it->set_output_width(
                    (it->input_width() - it->conv().kernel_size() + 2 * it->conv().pad())
                    / it->conv().stride() + 1);

        } else if (it->has_pool()) {
            uint32_t stride;

            if (it->pool().has_stride()) {
                stride = it->pool().stride();

            } else {
                stride = it->pool().dim();

            }

            it->set_num_outputs(it->num_inputs());
            it->set_output_height(div_ceil(it->input_height(), stride));
            it->set_output_width(div_ceil(it->input_width(), stride));

        } else if (it->has_lrn()) {
            it->set_num_outputs(it->num_inputs());
            it->set_output_height(it->input_height());
            it->set_output_width(it->input_width());

        } else {
            throw fpgaconvnet::Exception("Unknown layer " + std::to_string((long long unsigned) i));

        }
    }
    log_stdout() << network.DebugString() << std::endl;
    return network;
}

/* Utilities for loading weights into C-arrrays. */
void load_float_array_from_binary_file(
    std::string filename,
    const int size,
    float *output)
{
    generic_load(filename, size, output);
}


void load_kernels_from_file(
    std::string filename,
    const protos::LayerParameter & layer,
    float *output
)
{
    generic_load(filename, calc_total_kernel_weights(layer), output);
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
    int total_kernels = layer.num_outputs() * layer.num_inputs();
    float *buffer = new float[total_kernel_size];

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


void report_conv_performance(
    const protos::Network & network,
    uint64_t N,
    timeval t_begin,
    timeval t_end
)
{
    double begin = double(t_begin.tv_sec) * 1000000 + double(t_begin.tv_usec);
    double end = double(t_end.tv_sec) * 1000000 + double(t_end.tv_usec);
    double delta = end - begin;
    double throughput = double(N) / delta * 1000000;
    double total_ops = 0.0;
   
    for (auto it = network.layer().begin() ; it != network.layer().end() ; it++) {
        if (it->has_conv()) {
            total_ops += (
                    2 * double(it->conv().kernel_size() * it->conv().kernel_size())
                    * double(it->output_height() * it->output_width())
                    * double(it->num_inputs() * it->num_outputs()));
        }
    }

    log_stdout(INFO) << "Time taken for " << N << " feature extractions  = "
             << delta << std::endl;
    log_stdout(INFO) << "Throughput (images per second) = "
            << throughput << std::endl;
    log_stdout(INFO) << "GOps = " << throughput * total_ops / 1e9 << std::endl;
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

    const protos::LayerParameter & final_layer = network.layer(network.layer_size() - 1);
    const int conv_out_size = (final_layer.output_height() *
                               final_layer.output_width() *
                               final_layer.num_outputs());

    for (uint32_t i = 0 ; i < std::min(N, 10ul) ; i++) {
        for (uint32_t j = 0 ; j < conv_out_size; j++) {
            float expected;
            float obtained = conv_out[conv_out_size * i + j];

            if (format == FORMAT_TXT) {
                fin >> expected;
            } else {
                fin.read((char*) &expected, 4);
            }

            if (fin.eof()) {
                fin.clear();
                log_stdout(WARNING) << "Verifier terminated early!" << std::endl;
                break;
            }

            total_error += std::abs(obtained  - expected);
            total_pixels += 1;

            if (std::abs(obtained - expected) > 0.01) {
                log_stdout(WARNING) << j << "\t| ERROR: Obtained " << obtained << ", expected " << expected << std::endl;
            }
            // else {
            //     log_stdout(WARNING) << j << "\t| OKAY: Obtained " << obtained << ", expected " << expected << std::endl;
            // }
        }
    }
    log_stdout(INFO) << "Average pixel_error = " << float(total_error) / float(total_pixels) << std::endl;
    fin.close();
}

/* Don't even ask me how this works. This is dark magic to me by now. */
void special_allign_and_place_kernel_weights(
        const protos::LayerParameter & layer,
        float *dest_base,
        float *src_base
)
{
    const uint64_t kernel_ff = layer.conv().kernel_folding_factor();
    const uint64_t conv_ff = layer.conv().conv_folding_factor();
    const uint64_t kernel_dim = layer.conv().kernel_size();
    const uint64_t worker_factor = layer.conv().worker_factor();
    const uint64_t total_iter = calc_total_iterations(layer);
    const uint64_t rom_per_worker =
            calc_total_rom_size(layer) / layer.conv().worker_factor();

    /* This for loop makes dest_base into
     *
     * wf * scheduler_iterations * cff * kff
     */
    for (int i = 0 ; i < worker_factor ; i++) {
        float *dest = dest_base + (i * rom_per_worker);
        float *src = src_base + (i * kernel_dim * kernel_dim);

        for (int w = 0; w < calc_scheduler_iterations(layer); w++) {
            const int worker_iter = w;  // the w-th channel that the worker's handling.

            if (i + worker_iter * worker_factor >= layer.num_inputs()) {
                continue;
            }

            for (int channel = 0 ; channel < layer.num_outputs() ; channel++) {
                const int src_offset =
                        (channel * layer.num_inputs() + worker_iter * worker_factor)
                        * kernel_dim * kernel_dim;
                const int conv_iters = calc_convolution_iterations(layer);
                const int scheduler_iters = calc_scheduler_iterations(layer);
                const int dest_offset =
                        ((worker_iter * conv_iters)
                         + ((channel % conv_ff) * conv_iters * scheduler_iters)
                         + (channel / conv_ff))
                        * layer.conv().kernel_folding_factor() * calc_kernel_iterations(layer);

                /*
                 * Useful piece of code to visualize the kernels weights
                 *    -> worker position mapping.
                 * std::cout << "src_offset = " 
                 *         << src_offset / (kernel_dim * kernel_dim)
                 *         << " -> "
                 *         << dest_offset / (kernel_dim * kernel_dim)
                 *         << std::endl;
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
    const uint64_t total_rom_size = calc_total_rom_size(layer);
    float *tmp = new float[total_rom_size];
    const uint64_t rom_per_worker =
            total_rom_size / layer.conv().worker_factor();
    const uint64_t rom_per_conv =
            rom_per_worker / layer.conv().conv_folding_factor();
    const uint64_t total_iterations = calc_total_iterations(layer);

    special_allign_and_place_kernel_weights(
            layer,
            tmp,
            src_base);

    for (int iter = 0 ; iter < total_iterations ; iter++) {
        uint64_t addr = iter * calc_total_multipliers(layer);

        for (int worker = 0 ; worker < layer.conv().worker_factor() ; worker++) {
            for (int conv = 0 ; conv < layer.conv().conv_folding_factor() ; conv++) {
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


void allign_and_place_lmem_initialized_kernel_weights(
        const protos::LayerParameter & layer,
        fixed_point_t *dest_base,
        float *src_base
)
{
    const uint64_t total_rom_size = calc_total_rom_size(layer);
    float *tmp = new float[total_rom_size];
    const uint64_t rom_per_worker =
            total_rom_size / layer.conv().worker_factor();
    const uint64_t rom_per_conv =
            rom_per_worker / layer.conv().conv_folding_factor();
    const uint64_t total_iterations = calc_total_iterations(layer);

    special_allign_and_place_kernel_weights(
            layer,
            tmp,
            src_base);

    for (int iter = 0 ; iter < total_iterations ; iter++) {
        uint64_t addr = iter * calc_weights_vector_size(layer);

        for (int worker = 0 ; worker < layer.conv().worker_factor() ; worker++) {
            for (int conv = 0 ; conv < layer.conv().conv_folding_factor() ; conv++) {
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
    for (int i = 0 ; i < conv_layer_params.size() ; i++) {
        auto conv_layer = conv_layer_params[i];
        uint64_t total_weights = calc_total_kernel_weights(conv_layer);
        uint64_t total_bias = conv_layer.num_outputs();

        for (int j = 0 ; j < total_weights; j++) {
            kernels[i][j] = (float) rng(-0.75, 0.75);
        }

        for (int j = 0 ; j < total_bias ; j++) {
            bias[i][j] = (float) rng(-0.75, 0.75);
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

    if (initialized_weights) {
        log_stdout(INFO)
                << "Host-initialized weights has been set in previous calls."
                << std::endl;
        if (is_layer_cpu_initialized(layer)) {
            sprintf(buffer, "kernel_%d", layer.layer_id());
            max_queue_input(action, buffer, NULL, 0);
        }
        sprintf(buffer, "bias_%d", layer.layer_id());
        max_queue_input(action, buffer, NULL, 0);

    } else {
        log_stdout(INFO)
                << "Passing in host-initialized weights (This will only be done once)."
                << std::endl;

        if (is_layer_cpu_initialized(layer)) {
            uint64_t stream_size = calc_cpu_weights_stream_size(layer);
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
                sizeof(fixed_point_t) * calc_bias_stream_size(layer));
    }

}

Convnet::Convnet(
        const protos::Network & network_params,
        std::vector<max_file_t*> max_files,
        const char* load_spec)
{
    constructor(network_params, max_files, load_spec);
}


Convnet::Convnet(
        const protos::Network & network_params,
        max_file_t* max_file,
        const char* load_spec)
{
    std::vector<max_file_t*> max_files;
    max_files.push_back(max_file);
    constructor(network_params, max_files, load_spec);
}

void Convnet::constructor(
        const protos::Network & arg_network_params,
        std::vector<max_file_t*> arg_max_files,
        const char* arg_load_spec)
{
    network_params = arg_network_params;
    max_files = arg_max_files;
    initialized_weights = false;
    num_fpgas = max_files.size();
    load_spec = arg_load_spec;
    fpga_input_size = std::vector<int>(num_fpgas);
    fpga_output_size = std::vector<int>(num_fpgas);

    if (num_fpgas == 1) {
        dfe_array = NULL;
        dfe = max_load(max_files[0], load_spec);
    } else {
        dfe = NULL;
#ifdef __SIM__
        dfe_array = NULL;
#else
        dfe_array = max_load_mixed_array((max_file_t**) &max_files[0], num_fpgas, load_spec);
#endif
    }

    for (auto it = network_params.layer().begin();
            it != network_params.layer().end();
            it++) {
        if (it->has_conv()) {
            uint64_t worker_kernel_total_size =
                    calc_weights_vector_size(*it) * calc_total_iterations(*it);
            uint64_t bias_total_size = calc_bias_stream_size(*it);

            conv_layer_params.push_back(*it);
            kernels.push_back(new float[calc_total_kernel_weights(*it)]);
            bias.push_back(new fixed_point_t[bias_total_size]);
            worker_kernels.push_back(new fixed_point_t[worker_kernel_total_size]);
        }

        fpga_output_size[it->fpga_id()] =
                it->output_height() * it->output_width() * it->num_outputs();
    }

    fpga_input_size[0] =
            network_params.layer(0).input_height()
            * network_params.layer(0).input_width()
            * network_params.layer(0).num_inputs();
    for (int i = 1 ; i < num_fpgas ; i++) {
        fpga_input_size[i] = fpga_output_size[i - 1];
    }

    const protos::LayerParameter &first_layer = network_params.layer(0);
    const protos::LayerParameter &final_layer = network_params.layer(
            network_params.layer_size() - 1);

    input_size =
            first_layer.input_height() * first_layer.input_width()
            * first_layer.num_inputs();

    output_size =
            final_layer.output_height() * final_layer.output_width()
            * final_layer.num_outputs();
}

Convnet::~Convnet ()
{
    for (int i = 0 ; i < conv_layer_params.size() ; i++) {
        delete[] kernels[i];
        delete[] bias[i];
        delete[] worker_kernels[i];
    }
    for (int i = 0 ; i < queue_weights.size(); i++) {
        delete[] queue_weights[i];
    }
    if (dfe) {
        max_unload(dfe);
    }
    if (dfe_array) {
        max_unload_array(dfe_array);
    }
}

void Convnet::load_weights_from_files(std::vector<std::string> filenames, file_format_t file_type)
{
    log_stdout(INFO) << "Loading weights from file." << std::endl;
    for (int i = 0 ; i < conv_layer_params.size(); i++) {
        float *bias_tmp = new float[calc_bias_stream_size(
                conv_layer_params[i])];

        if (file_type == FORMAT_TXT) {
            load_kernels_from_file(
                    filenames[2 * i], conv_layer_params[i], kernels[i]);
            load_bias_from_file(
                    filenames[2 * i + 1], conv_layer_params[i], bias_tmp);
        } else {
            load_kernels_from_binary_file(
                    filenames[2 * i], conv_layer_params[i], kernels[i]);
            load_bias_from_binary_file(
                    filenames[2 * i + 1], conv_layer_params[i], bias_tmp);
        }

        copy_float_to_fixed(
                bias[i], bias_tmp, calc_bias_stream_size(conv_layer_params[i]));

        delete[] bias_tmp;
    }
    log_stdout(INFO) << "Alligning weights." << std::endl;
    for (int i = 0; i < conv_layer_params.size() ; i++) {
        if (is_layer_cpu_initialized(conv_layer_params[i])) {
            allign_and_place_cpu_initialized_kernel_weights(
                    conv_layer_params[i], worker_kernels[i], kernels[i]);
        } else {
            allign_and_place_lmem_initialized_kernel_weights(
                    conv_layer_params[i], worker_kernels[i], kernels[i]);
        }
    }
    log_stdout(INFO) << "Done!" << std::endl;
}


void Convnet::max_init_weights()
{
    uint64_t start = 0;
    std::vector<int8_t*> buffer_ptrs;

    for (int i = 0; i < conv_layer_params.size() ; i++) {
        if (is_layer_cpu_initialized(conv_layer_params[i])) {
            log_stdout(INFO) << "layer "
                    << conv_layer_params[i].layer_id()
                    << " is host-initialized. Skipping .." << std::endl;
            continue;
        }

        auto & layer = conv_layer_params[i];
        max_actions_t *write_action = max_actions_init(
                max_files[conv_layer_params[i].fpga_id()], "writeLMem");
        const uint64_t address =
                conv_layer_params[i].conv().weight_address_base();
        const uint64_t stream_size =
                calc_total_iterations(layer)
                * calc_weights_vector_size(layer)
                * sizeof(fixed_point_t);

        log_stdout(INFO) << "Initializing weights in LMEM at layer "
            << conv_layer_params[i].layer_id()
            << " [fpga_id = " << conv_layer_params[i].fpga_id() << "]"
            << std::endl;
        log_stdout(INFO) << "Address = " << address << std::endl;
        log_stdout(INFO) << "Stream size (in bytes) = " << stream_size << std::endl;
        max_set_param_uint64t(write_action, "start", address);
        max_set_param_uint64t(write_action, "size", stream_size);
        max_queue_input(
                write_action,
                "weights_in",
                (void*) &worker_kernels[i][0],
                stream_size);

        if (num_fpgas == 1) {
            max_run(dfe, write_action);
        }
#ifdef __SIM__
        else {
            dfe = max_load(max_files[conv_layer_params[i].fpga_id()], load_spec);
            max_run(dfe, write_action);
            max_unload(dfe);
            dfe = NULL;
        }
#else
        // TODO(fyq14): Complete this
        else {
            max_run(dfe, write_action);
        }
#endif

        max_actions_free(write_action);
        log_stdout(INFO) << "Done!" << std::endl;
    }

}


std::vector<float> Convnet::max_run_inference(
        uint64_t N,
        const std::vector<float> & images,
        const bool benchmark
)
{
    timeval t_begin, t_end;
    const uint64_t address_images = 0;
    const uint64_t address_features = N * input_size * sizeof(float);
    max_actions_t **actions = new max_actions_t*[num_fpgas];
    std::vector<float> ret((N) * output_size , 0);
    std::vector<bool> has_conv(num_fpgas, 0);

    for (int i = 0 ; i < num_fpgas ; i++) {
        actions[i] = max_actions_init(max_files[i], "default");
    }

    log_stdout(INFO) << "Setting up feature extraction actions ... " << std::endl;

    int i = 0;
    for (auto it = network_params.layer().begin();
            it != network_params.layer().end();
            it++) {
        if (it->has_conv()) {
            set_layer_weights(
                    actions[it->fpga_id()], *it, worker_kernels[i], bias[i]);
            has_conv[it->fpga_id()] = 1;
            i++;
        } else if (it->has_lrn()) {
            /* assuming binomial approximation used. */
            char buffer[30];

            sprintf(buffer, "approx_factor_%d", it->layer_id());
            max_set_param_double(
                    actions[it->fpga_id()],
                    buffer,
                    -it->lrn().alpha() * it->lrn().beta() / float(it->lrn().local_size()));
        }
    }

    for (int i = 0 ; i < num_fpgas ; i++) {
        max_set_param_uint64t(actions[i], "N", N);
        if (has_conv[i]) {
            if (initialized_weights) {
                max_set_param_uint64t(actions[i], "init", 0);
            } else {
                max_set_param_uint64t(actions[i], "init", 1);
            }
        }
    }


    initialized_weights = 1;

    max_queue_input(actions[0],
                    "fromcpu",
                    (void*) &images[0],
                    N * input_size * sizeof(float));
    max_queue_output(actions[num_fpgas - 1],
                     "tocpu",
                     (void*) &ret[0],
                     N * output_size * sizeof(float));

    log_stdout(INFO) << "Running feature extraction ... " << std::endl;

#ifdef __SIM__
    void *tmp_buffer_in;
    void *tmp_buffer_out;

    for (int i = 0; i < num_fpgas ; i++) {
        log_stdout(INFO) << "Running on DFE " << i << " ..." << std::endl;

        if (num_fpgas > 1) {
            dfe = max_load(max_files[i], load_spec);
        }

        if (i > 0) {
            max_queue_input(actions[i],
                            "mock_maxring_in",
                            tmp_buffer_in,
                            N * fpga_input_size[i] * 2);
        }

        if (i < num_fpgas - 1) {
            tmp_buffer_out = malloc(N * fpga_output_size[i] * 2);
            max_queue_output(actions[i],
                            "mock_maxring_out",
                            tmp_buffer_out,
                            N * fpga_output_size[i] * 2);
        }
        max_run(dfe, actions[i]);

        if (num_fpgas > 1) {
            max_unload(dfe);
        }

        tmp_buffer_in = tmp_buffer_out;
        tmp_buffer_out = NULL;
    }
    dfe = NULL;

    if (tmp_buffer_out != NULL) {
        delete[] tmp_buffer_out;
    }
    if (tmp_buffer_in != NULL) {
        delete[] tmp_buffer_in;
    }
#else
    if (num_fpgas == 1) {
        __sync_synchronize();
        gettimeofday(&t_begin, NULL);
        max_run(dfe, actions[0]);
        gettimeofday(&t_end, NULL);
        __sync_synchronize();

    } else {
        max_actarray_t *act_array = max_mixed_actarray_init(&max_files[0], num_fpgas);
        for (int i = 0 ; i < num_fpgas ; i++) {
            max_set_action(act_array, i, actions[i]);
        }

        __sync_synchronize();
        gettimeofday(&t_begin, NULL);
        max_run_array(dfe_array, act_array);
        gettimeofday(&t_end, NULL);
        __sync_synchronize();
    }
#endif

    if (benchmark) {
        report_conv_performance(network_params, N, t_begin, t_end);
    }
    log_stdout(INFO) << "Done!" << std::endl;

    delete[] actions;

    return ret;
}


Exception::Exception(const std::string & message)
    : message(message)
{
}


Exception::~Exception() throw()
{
}


const char* Exception::what() const throw()
{
    return message.c_str();
}

} // fpgaconvnet

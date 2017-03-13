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

static bool is_layer_cpu_initialized(const protos::LayerParameter & layer)
{
    return layer.conv().bram_factor() >= (layer.num_inputs() * layer.num_outputs());
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

            if (std::abs(obtained - expected) > 0.001) {
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

void allign_and_place_cpu_initialized_kernel_weights(
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


void allign_and_place_lmem_initialized_kernel_weights(
        const protos::LayerParameter & layer,
        float *dest_base,
        float *src_base
)
{
    const uint64_t total_rom_size = calc_total_rom_size(layer);
    float *tmp = new float[total_rom_size];
    const uint64_t rom_per_worker =
            total_rom_size / layer.conv().worker_factor();
    const uint64_t rom_per_conv =
            rom_per_worker
            / (calc_scheduler_iterations(layer) * layer.conv().conv_folding_factor());
    const uint64_t total_iterations = calc_total_iterations(layer);

    allign_and_place_cpu_initialized_kernel_weights(
            layer,
            tmp,
            src_base);

    uint64_t addr = 0;
    for (int iter = 0 ; iter < total_iterations ; iter++) {
        for (int worker = 0 ; worker < layer.conv().worker_factor() ; worker++) {
            float *copy_base = tmp + rom_per_worker;

            for (int conv = 0 ; conv < layer.conv().conv_folding_factor() ; conv++) {

                std::memcpy(
                        dest_base + addr,
                        copy_base + (conv * rom_per_conv),
                        sizeof(float) * layer.conv().kernel_folding_factor());
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
        float *worker_kernels,
        float *bias
)
{
    /* Requirements:
     * - stream to be a multiple of 16 bytes = 4 * 4 floats
     * - As input vector is of size layer.conv().kernel_folding_factor(), let's call this number
     *   kff.
     */
    char buffer[30];
    uint64_t multiple_base = lcm(4, layer.conv().kernel_folding_factor())
            / layer.conv().kernel_folding_factor();
    uint64_t total_rom_size = calc_total_rom_size(layer);
    uint64_t num_iters = total_rom_size / layer.conv().kernel_folding_factor();
    uint64_t padded_num_iters = div_ceil(num_iters, multiple_base) * multiple_base;
    uint64_t padded_rom_size =
            padded_num_iters * layer.conv().kernel_folding_factor();

    sprintf(buffer, "kernel_%d", layer.layer_id());

    if (initialized_weights) {
        max_queue_input(action, buffer, NULL, 0);

    } else {
        float *values = new float[padded_rom_size];

        queue_weights.push_back(values);
        std::memcpy(values, worker_kernels, sizeof(float) * total_rom_size);
        max_queue_input(
                action,
                buffer,
                values,
                sizeof(float) * padded_rom_size);

    }

    sprintf(buffer, "bias_%d", layer.layer_id());
    max_queue_input(action, buffer, bias,
                    sizeof(float) * div_ceil(layer.num_outputs(), 4) * 4);
}


Convnet::Convnet(
        const protos::Network & network_params,
        max_file_t *max_file,
        const char* load_spec) :
    network_params(network_params), max_file(max_file),
    initialized_weights(false),
    dfe(max_load(max_file,load_spec))
{
    for (auto it = network_params.layer().begin();
            it != network_params.layer().end();
            it++) {
        if (it->has_conv()) {
            conv_layer_params.push_back(*it);
            kernels.push_back(new float[calc_total_kernel_weights(*it)]);
            bias.push_back(new float[div_ceil(it->num_outputs(), 4) * 4]);
            worker_kernels.push_back(new float[div_ceil(calc_total_rom_size(*it), 96) * 96]);
            memset(worker_kernels.back(), 0,
                   sizeof(float) * calc_total_kernel_weights(*it));
        }
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
}

void Convnet::load_weights_from_files(std::vector<std::string> filenames, file_format_t file_type)
{
    for (int i = 0 ; i < conv_layer_params.size(); i++) {
        if (file_type == FORMAT_TXT) {
            load_kernels_from_file(filenames[2 * i], conv_layer_params[i], kernels[i]);
            load_bias_from_file(filenames[2 * i + 1], conv_layer_params[i], bias[i]);
        } else {
            load_kernels_from_binary_file(filenames[2 * i], conv_layer_params[i], kernels[i]);
            load_bias_from_binary_file(filenames[2 * i + 1], conv_layer_params[i], bias[i]);
        }
    }
    for (int i = 0; i < conv_layer_params.size() ; i++) {
        if (is_layer_cpu_initialized(conv_layer_params[i])) {
            allign_and_place_cpu_initialized_kernel_weights(
                    conv_layer_params[i], worker_kernels[i], kernels[i]);
        } else {
            allign_and_place_lmem_initialized_kernel_weights(
                    conv_layer_params[i], worker_kernels[i], kernels[i]);
        }
    }
}


void Convnet::max_init_weights()
{
    uint64_t start = 0;
    std::vector<int8_t*> buffer_ptrs;

    for (int i = 0; i < conv_layer_params.size() ; i++) {
        if (is_layer_cpu_initialized(conv_layer_params[i])) {
            log_stdout(INFO) << "layer " << i << " is host-initialized." << std::endl;
            continue;
        }

        auto & layer = conv_layer_params[i];
        max_actions_t *write_action = max_actions_init(max_file, "writeLMem");
        const uint64_t stream_size =
                div_ceil(calc_total_rom_size(layer) * sizeof(float), 384) * 384;

        log_stdout(INFO) << "Initializing weights in LMEM at layer " << i << std::endl;
        max_set_param_uint64t(write_action, "start", 0);
        max_set_param_uint64t(write_action, "size", stream_size);
        max_queue_input(
                write_action,
                "weights_in",
                (void*) &worker_kernels[i],
                stream_size);
        max_run(dfe, write_action);
        max_actions_free(write_action);
        log_stdout(INFO) << "Done!" << std::endl;
    }

}


void Convnet::max_load_input_data(const std::vector<float> & images, uint64_t N)
{
    const uint64_t address_images = 0;
    const uint64_t address_features = N * input_size * sizeof(float);
    max_actions_t *load_action = max_actions_init(max_file, "load_data");

    log_stdout(INFO) << "Copying sample data to off-chip memory." << std::endl;
    max_set_param_uint64t(load_action, "address", address_images);
    max_set_param_uint64t(load_action, "size", N * input_size);
    max_queue_input(
            load_action, "fromcpu",
            (void*) &images[0], sizeof(float) * N * input_size);
    // max_disable_validation(load_action);
    max_run(dfe, load_action);
    max_actions_free(load_action);
}


std::vector<float> Convnet::max_run_inference(
        uint64_t N,
        const std::vector<float> & images,
        const bool benchmark
)
{
    const uint64_t address_images = 0;
    const uint64_t address_features = N * input_size * sizeof(float);
    max_actions_t *run_action = max_actions_init(max_file, "default");
    timeval t_begin, t_end;
    std::vector<float> ret((N) * output_size , 0);

    log_stdout(INFO) << "Running Feature Extraction ... " << std::endl;
    max_set_param_uint64t(run_action, "N", N);

    int i = 0;
    bool has_cpu_init_conv = 0;
    for (auto it = network_params.layer().begin();
            it != network_params.layer().end();
            it++) {
        if (it->has_conv() && is_layer_cpu_initialized(*it)) {
            has_cpu_init_conv = 1;
            set_layer_weights(
                    run_action, *it, worker_kernels[i], bias[i]);
            i++;
        } else if (it->has_lrn()) {
            /* assuming binomial approximation used. */
            char buffer[30];

            sprintf(buffer, "approx_factor_%d", it->layer_id());
            max_set_param_double(
                    run_action,
                    buffer,
                    -it->lrn().alpha() * it->lrn().beta() / float(it->lrn().local_size()));
        }
    }

    if (initialized_weights) {
        max_set_param_uint64t(run_action, "init", 0);
    } else {
        max_set_param_uint64t(run_action, "init", 1);
    }
    initialized_weights = 1;


    std::cout << "input stream size = "
            << images.size() * sizeof(float) << std::endl;
    max_queue_input(run_action,
                    "fromcpu",
                    (void*) &images[0],
                    images.size() * sizeof(float));
    max_queue_output(run_action,
                     "tocpu",
                     (void*) &ret[0],
                     N * output_size * sizeof(float));
    __sync_synchronize();
    gettimeofday(&t_begin, NULL);
    max_run(dfe, run_action);
    gettimeofday(&t_end, NULL);
    __sync_synchronize();
    max_actions_free(run_action);

    if (benchmark) {
        report_conv_performance(network_params, N, t_begin, t_end);
    }

    return ret;
}

std::vector<float> Convnet::max_retrieve_features(uint64_t N) {
    const uint64_t address_features = N * input_size * sizeof(float);
    std::vector<float> features(N * output_size);

    log_stdout(INFO) << "Copying features from off-chip memory." << std::endl;
    max_actions_t *read_action = max_actions_init(max_file, "get_results");
    max_set_param_uint64t(read_action, "address", address_features);
    max_set_param_uint64t(read_action, "size", N * output_size);
    max_queue_output(
            read_action,
            "tocpu",
            &features[0],
            N * output_size * sizeof(float));
    // max_disable_validation(read_action);
    max_run(dfe, read_action);

    return features;
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

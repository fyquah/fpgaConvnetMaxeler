#include <cmath>
#include <cstring>

#include <fstream>
#include <iostream>
#include <string>
#include <sys/time.h>

#include <Eigen/Dense>

#include "fpgaconvnet/convnet.h"
#include "fpgaconvnet/protos/parameters.pb.h"


static void generic_load(std::string filename, int count, double *output)
{
    std::ifstream fin(filename.c_str());
    for (int i = 0 ; i < count ; i++) {
        fin >> output[i];
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
            it->set_output_height(it->input_height() - it->conv().kernel_size() + 1);
            it->set_output_width(it->input_width() - it->conv().kernel_size() + 1);
        } else {
            it->set_num_outputs(it->num_inputs());
            it->set_output_height(it->input_height() / it->pool().dim());
            it->set_output_width(it->input_width() / it->pool().dim());
        }
    }
    log_stdout() << network.DebugString() << std::endl;
    return network;
}

/* Utilities for loading weights into C-arrrays. */
void load_kernels_from_file(
    std::string filename,
    const protos::LayerParameter & layer,
    double *output
)
{
    generic_load(filename, calc_total_kernel_weights(layer), output);
}


void load_bias_from_file(
    std::string filename,
    const protos::LayerParameter & layer,
    double *output
)
{
    generic_load(filename, layer.num_outputs(), output);
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
        std::string filename)
{
    std::ifstream fin(filename.c_str());
    uint32_t total_pixels = 0;
    float total_error = 0.0;

    const protos::LayerParameter & final_layer = network.layer(network.layer_size() - 1);
    const int conv_out_size = (final_layer.output_height() *
                               final_layer.output_width() *
                               final_layer.num_outputs());

    for (uint32_t i = 0 ; i < N ; i++) {
        for (uint32_t j = 0 ; j < conv_out_size; j++) {
            float expected;
            float obtained = conv_out[conv_out_size * i + j];
            fin >> expected;

            if (fin.eof()) {
                fin.clear();
                fin.seekg(0, std::ios::beg);
                log_stdout(INFO) << "Verifier moving to the beginning of file!"
                        << std::endl;
                fin >> expected;
            }

            total_error += std::abs(obtained  - expected);
            total_pixels += 1;

            if (std::abs(obtained - expected) > 0.01) {
                log_stdout(WARNING) << "Error > 0.01 while verifying output!" << std::endl;
            }
        }
    }
    log_stdout(INFO) << "pixel_error = " << float(total_error) / float(total_pixels) << std::endl;
    fin.close();
}

void allign_and_place_kernel_weights(
        const protos::LayerParameter & layer,
        double *dest_base,
        double *src_base
)
{
    const uint64_t kernel_ff = layer.conv().kernel_folding_factor();
    const uint64_t conv_ff = layer.conv().conv_folding_factor();
    const uint64_t kernel_dim = layer.conv().kernel_size();
    const uint64_t worker_factor = layer.conv().worker_factor();
    const uint64_t total_iter = calc_total_iterations(layer);

    for (int i = 0 ; i < worker_factor ; i++) {
        int total = ((layer.num_inputs() / worker_factor)
                * kernel_dim * kernel_dim
                * layer.num_outputs());
        double *dest = dest_base + (i * total);
        double *src = src_base + (i * kernel_dim * kernel_dim);

        for (int w = 0; w < layer.num_inputs() / worker_factor; w++) {
            const int worker_iter = w;  // the w-th channel that the worker's handling.

            for (int channel = 0 ; channel < layer.num_outputs() ; channel++) {
                const int src_offset =
                        (channel * layer.num_inputs() + worker_iter * worker_factor)
                        * kernel_dim * kernel_dim;
                const int dest_offset =
                        ((worker_iter * (layer.num_outputs() / conv_ff))
                         + (channel % conv_ff) * (layer.num_outputs() / conv_ff) * (layer.num_inputs() / worker_factor)
                         + (channel / conv_ff))
                        * kernel_dim * kernel_dim;
                
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
                        sizeof(double) * kernel_dim * kernel_dim);
            }
        }
    }

    /* At this point, the dimension of dest_base is
     *      num_worker * conv_factor * total_iter * kernel_factor
     *
     * We want to change it to:
     *      num_worker * conv_factor * kernel_factor * total_iter
     */
    double *tmp = new double[kernel_ff * total_iter];

    for (uint64_t worker = 0; worker < worker_factor; worker++) {
        for (uint64_t conv = 0; conv < conv_ff; conv++) {
            uint64_t offset = ((worker * conv_ff) + conv) * kernel_ff * total_iter;
            double *ptr = dest_base + offset;

            std::memcpy(tmp, ptr, kernel_ff * total_iter * sizeof(double));

            Eigen::Map<Eigen::MatrixXd> matrix_const(tmp, total_iter, kernel_ff);
            Eigen::Map<Eigen::MatrixXd> matrix(ptr, kernel_ff, total_iter);

            for (int i = 0 ; i < total_iter ; i++) {
                for (int j = 0 ; j < kernel_ff ; j++) {
                    ptr[j * total_iter + i] = tmp[i * kernel_ff + j];
                }
            }
        }
    }

    delete[] tmp;
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


void max_set_single_rom(
        max_actions_t *action,
        const protos::LayerParameter & layer,
        double *worker_kernels,
        uint64_t worker,
        uint64_t conv,
        uint64_t mult)
{
    char buffer[100];
    uint64_t rom_size = calc_total_iterations(layer);

    sprintf(buffer, "layer_%d_kernels_%d_%d_%d", layer.layer_id(), worker, conv, 100);
    for (int i = 0 ; i < rom_size ; i++) {
        uint64_t index;

        index = worker;
        index = (index * layer.conv().conv_folding_factor()) + conv;
        index = (index * layer.conv().kernel_folding_factor()) + mult;
        index = (index * rom_size) + i;
        max_set_param_array_double(action, buffer, worker_kernels[index], i);
    }
}


void max_set_layer_weights(
        max_actions_t *action,
        const protos::LayerParameter & layer,
        double *worker_kernels,
        double *bias
)
{
    char buffer[30];

    for (int worker = 0 ; worker < layer.conv().worker_factor() ; worker++) {
        for (int conv = 0; conv < layer.conv().conv_folding_factor(); conv++) {
            for (int mult = 0; mult < layer.conv().kernel_folding_factor(); mult++) {
                max_set_single_rom(
                        action, layer, worker_kernels,
                        worker, conv, mult);
            }
        }
    }

    for (int i = 0 ; i < layer.num_outputs() ; i++) {
        sprintf(buffer, "bias_%d", layer.layer_id());
        max_set_param_array_double(action, buffer, bias[i], i);
    }
}


Convnet::Convnet(
        const protos::Network & network_params,
        max_file_t *max_file,
        const char* load_spec) :
    network_params(network_params), max_file(max_file),
    dfe(max_load(max_file,load_spec))
{
    for (auto it = network_params.layer().begin();
            it != network_params.layer().end();
            it++) {
        if (it->has_conv()) {
            conv_layer_params.push_back(*it);
            kernels.push_back(new double[calc_total_kernel_weights(*it)]);
            bias.push_back(new double[it->num_outputs()]);
            worker_kernels.push_back(
                    new double[calc_total_kernel_weights(*it)]);
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
    if (dfe) {
        max_unload(dfe);
    }
}

void Convnet::load_weights_from_files(std::vector<std::string> filenames)
{
    for (int i = 0 ; i < conv_layer_params.size(); i++) {
        load_kernels_from_file(filenames[2 * i], conv_layer_params[i], kernels[i]);
        load_bias_from_file(filenames[2 * i + 1], conv_layer_params[i], bias[i]);
    }
    for (int i = 0; i < conv_layer_params.size() ; i++) {
        allign_and_place_kernel_weights(
                conv_layer_params[i], worker_kernels[i], kernels[i]);
    }
}

void Convnet::max_init_weights()
{
    log_stdout(INFO) << "Initializing net weights in DFE." << std::endl;
    max_actions_t *memory_action = max_actions_init(max_file, "init_convnet");

    int i = 0;
    for (auto it = network_params.layer().begin();
            it != network_params.layer().end();
            it++) {
        if (it->has_conv()) {
            max_set_layer_weights(
                    memory_action, *it, worker_kernels[i], bias[i]);
            i++;
        }
    }
    // max_disable_validation(memory_action);
    max_run(dfe, memory_action);
    max_actions_free(memory_action);
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

void Convnet::max_run_inference(uint64_t N) {
    const uint64_t address_images = 0;
    const uint64_t address_features = N * input_size * sizeof(float);
    max_actions_t *run_action = max_actions_init(max_file, "default");
    timeval t_begin, t_end;

    log_stdout(INFO) << "Running Feature Extraction ... " << std::endl;
    max_set_param_uint64t(run_action, "N", N);
    max_set_param_uint64t(run_action, "address_images", address_images);
    max_set_param_uint64t(run_action, "address_features", address_features);
    // max_disable_validation(run_action);
    __sync_synchronize();
    gettimeofday(&t_begin, NULL);
    max_run(dfe, run_action);
    gettimeofday(&t_end, NULL);
    __sync_synchronize();
    max_actions_free(run_action);
    report_conv_performance(network_params, N, t_begin, t_end);
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

} // fpgaconvnet

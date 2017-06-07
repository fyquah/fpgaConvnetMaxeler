#include <fcntl.h>

#include <cmath>

#include <iostream>

#include "fpgaconvnet/common.h"
#include "fpgaconvnet/protos/parameters.pb.h"



namespace fpgaconvnet
{

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

            uint32_t unstrided_height = it->input_height() - it->pool().dim();
            uint32_t unstrided_width = it->input_width() - it->pool().dim();

            it->set_num_outputs(it->num_inputs());
            it->set_output_height(math::div_ceil(unstrided_height, stride) + 1);
            it->set_output_width(math::div_ceil(unstrided_width, stride) + 1);

        } else if (it->has_lrn()) {
            it->set_num_outputs(it->num_inputs());
            it->set_output_height(it->input_height());
            it->set_output_width(it->input_width());

        } else {
            throw fpgaconvnet::Exception("Unknown layer " + std::to_string((long long unsigned) i));

        }
    }
    logging::stdout() << network.DebugString() << std::endl;
    return network;
}


// Exception stuff
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


// Logging stuff
namespace logging {

static const char* level_strings[] = {
        "DEBUG", "INFO", "WARNING", "ERROR"
};

static std::string LOG_PREFIX = "default";
static std::string INDENTATION = "";

std::ostream& stdout(int level)
{
    if (level > 4) {
        level = 4;
    } else if (level < 0) {
        level = 0;
    }
    return std::cout << "[" << LOG_PREFIX << "\t "
            << level_strings[level] << "]\t" << INDENTATION;
}

void log_prefix(const std::string & prefix)
{
    LOG_PREFIX = prefix;
}

void indent()
{
    INDENTATION.append("> ");
}

void dedent()
{
    INDENTATION = INDENTATION.substr(0, INDENTATION.length() - 2);
}


Indentation::Indentation()
{
    indent();
}

Indentation::~Indentation()
{
    dedent();
}

}  // logging


namespace math {

double rng(const double lo, const double hi)
{
    return lo + (hi - lo) * ((double)rand()/(double)RAND_MAX);
}


uint64_t gcd(uint64_t a, uint64_t b)
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

uint64_t lcm(uint64_t a, uint64_t b)
{
    return a / gcd(a, b) * b;
}


uint64_t div_ceil(uint64_t a, uint64_t b)
{
    if (a % b == 0) {
        return a / b;
    } else {
        return a / b + 1;
    }
}



}  // math


namespace calculation {

double throughput(const protos::Network & network)
{
    double frequency = network.frequency() * 1e6;
    double cycle_length = 1.0 / frequency;
    double size =
        network.layer(0).input_height() * network.layer(0).input_width();

    double input_image_bytes =
        size * network.layer(0).num_inputs() * sizeof(fixed_point_t);

    /* In terms of images per clock cycle */
    double throughput = PCIE_BANDWIDTH / input_image_bytes * cycle_length;
    int prev_fpga = 0;

    for (int i = 0 ; i < network.layer_size() ; i++) {
        auto layer = network.layer(i);
        const double input_size =
                layer.input_height() * layer.input_width();

        const double output_size =
                layer.output_height() * layer.output_width();

        const double area_compression = output_size / input_size;
        double layer_throughput = 0.0;

        if (layer.has_conv()) {
            const double scheduler_throughput =
                area_compression / layer.conv().worker_factor();

            const double computation_throughput =
                1.0 / (calculation::total_iterations(layer) * output_size);

            layer_throughput = std::min(
                    scheduler_throughput, computation_throughput);

        } else if (layer.has_pool()) {
            layer_throughput =
                1.0 / (calculation::total_iterations(layer) * input_size);

        } else if (layer.has_lrn()) {
            layer_throughput =
                1.0 / (calculation::total_iterations(layer) * input_size);

        }

        throughput = std::min(throughput, layer_throughput);
        if (layer.fpga_id() != prev_fpga) {
            const uint64_t image_bytes =
                network.layer(i).input_height()
                * network.layer(i).input_width()
                * network.layer(i).num_inputs()
                * sizeof(fpgaconvnet::fixed_point_t);

            throughput = std::min(
                    throughput,
                    double(fpgaconvnet::calculation::MAXRING_BANDWIDTH)
                    / double(image_bytes));
        }
        prev_fpga = layer.fpga_id();
    }

    return throughput * frequency;
}


double ops(const protos::Network & network)
{
    double total_ops = 0.0;

    for (auto it = network.layer().begin() ; it != network.layer().end() ; it++) {
        if (it->has_conv()) {
            total_ops += (
                    2 * double(it->conv().kernel_size() * it->conv().kernel_size())
                    * double(it->output_height() * it->output_width())
                    * double(it->num_inputs() * it->num_outputs() / it->conv().group()));
        } else if (it->has_pool()) {
            total_ops += it->output_height() * it->output_width() * it->num_inputs();

        } else if (it->has_lrn()) {
            total_ops += it->input_height() * it->input_width() * it->num_inputs();

        }
    }

    return total_ops;

}


uint64_t total_multipliers(const protos::LayerParameter & layer)
{
    return layer.conv().worker_factor()
            * layer.conv().conv_folding_factor()
            * layer.conv().kernel_folding_factor();
}


uint64_t kernel_iterations(const protos::LayerParameter & layer)
{
    uint64_t kernelDim = layer.conv().kernel_size();
    return math::div_ceil(kernelDim * kernelDim, layer.conv().kernel_folding_factor());
}


uint64_t convolution_iterations(const protos::LayerParameter & layer)
{
    return math::div_ceil(layer.num_outputs() / layer.conv().group(),
                    layer.conv().conv_folding_factor()); 
}


uint64_t total_iterations(const protos::LayerParameter &layer)
{
    if (layer.has_conv()) {
        return scheduler_iterations(layer)
                * convolution_iterations(layer)
                * kernel_iterations(layer);

    } else if (layer.has_pool()) {
        return math::div_ceil(
                layer.num_inputs(),
                layer.pool().channel_folding_factor());


    } else if (layer.has_lrn()) {
        return math::div_ceil(
                layer.num_inputs(),
                layer.lrn().channel_folding_factor());


    }
}


uint64_t scheduler_iterations(const protos::LayerParameter & layer)
{
    return math::div_ceil(layer.num_inputs(), layer.conv().worker_factor());
}


uint64_t total_kernel_weights(const protos::LayerParameter & layer)
{
    if (!layer.has_conv()) {
        return 0;
    }
    auto & conv = layer.conv();
    return layer.num_inputs()
            * layer.num_outputs() / layer.conv().group()
            * conv.kernel_size() * conv.kernel_size();
}


uint64_t conv_in_size(const protos::Network & network)
{
    return network.layer(0).num_inputs()
	* network.layer(0).input_height()
	* network.layer(0).input_width();
}


uint64_t total_rom_size(const protos::LayerParameter & layer)
{
    return layer.conv().worker_factor()
            * layer.conv().conv_folding_factor()
            * layer.conv().kernel_folding_factor()
            * total_iterations(layer);
}

uint64_t weights_vector_size(
        const protos::LayerParameter & layer)
{
    int stream_chunk_size = 384 / sizeof(fixed_point_t);
    uint64_t weights_per_iter =
            layer.conv().worker_factor()
            * layer.conv().conv_folding_factor()
            * layer.conv().kernel_folding_factor();
    return math::div_ceil(weights_per_iter, stream_chunk_size) * stream_chunk_size;
}


bool is_layer_cpu_initialized(const protos::LayerParameter & layer)
{
    return
        !layer.conv().has_bram_factor()
        || (layer.conv().bram_factor()
                >= (layer.num_inputs()
                    * (layer.num_outputs() / layer.conv().group())));
}



uint64_t bias_stream_size(const protos::LayerParameter & layer)
{
    uint64_t stream_chunk_size = 16 / sizeof(fixed_point_t);
    return math::div_ceil(layer.num_outputs(), stream_chunk_size)
        * stream_chunk_size;
}

uint64_t cpu_weights_stream_size(
        const protos::LayerParameter & layer)
{
    uint64_t stream_chunk_size = 16 / sizeof(fixed_point_t);
    uint64_t multiple_base =
        math::lcm(stream_chunk_size, layer.conv().kernel_folding_factor())
        / layer.conv().kernel_folding_factor();

    uint64_t num_iters =
        total_rom_size(layer) / layer.conv().kernel_folding_factor();
    uint64_t padded_num_iters =
        math::div_ceil(num_iters, multiple_base) * multiple_base;
    uint64_t padded_rom_size =
        padded_num_iters * layer.conv().kernel_folding_factor();

    return padded_rom_size;
}


}  // calculation
}  // fpgaconvnet

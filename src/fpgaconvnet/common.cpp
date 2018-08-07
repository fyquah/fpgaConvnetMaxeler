#include <fcntl.h>

#include <cmath>

#include <iostream>
#include <fstream>

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
static unsigned current_level = 1;
static std::ofstream unopened_ofstream;

std::ostream& stdout(int level)
{
    if (level > 4) {
        level = 4;
    } else if (level < 0) {
        level = 0;
    }

    if (level >= current_level) {
        return std::cout << "[" << LOG_PREFIX << "\t "
                << level_strings[level] << "]\t" << INDENTATION;
    } else {
        return unopened_ofstream;  // hack
    }
}

void set_level(int level)
{
    current_level = level;
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

const protos::LayerParameter get_bitstream_first_layer(
        const protos::Network & network, const unsigned bitstream_id)
{
    if (bitstream_id == -1) {
        return network.layer(0);
    }

    for (int i = 0 ; i < network.layer_size() ; i++) {
        if (network.layer(i).bitstream_id() == bitstream_id) {
            return network.layer(i);
        }
    }

    assert(false);
}


const protos::LayerParameter get_bitstream_final_layer(
        const protos::Network & network, const unsigned bitstream_id)
{
    if (bitstream_id == -1) {
        return network.layer(network.layer_size() - 1);
    }

    for (int i =  network.layer_size() - 1; i >= 0 ; i--) {
        if (network.layer(i).bitstream_id() == bitstream_id) {
            return network.layer(i);
        }
    }

    assert(false);
}

const double
compute_input_num_bytes(const protos::LayerParameter l)
{
    return l.num_inputs() * l.input_height() * l.input_width();
}

const double 
compute_output_num_bytes(const protos::LayerParameter l)
{
    return l.num_outputs() * l.output_height() * l.output_width();
}


const double
bandwidth_throughput_limit(
        const protos::Network & network, const unsigned bitstream_id)
{
    // TODO: Allow user to configure PCIE or LMEM?
    /* In terms of images per clock cycle */
    const unsigned total_values =
        (compute_input_num_bytes(get_bitstream_first_layer(network, bitstream_id))
         + compute_output_num_bytes(get_bitstream_final_layer(network, bitstream_id)));
    const unsigned num_bytes = total_values * sizeof(fixed_point_t);
    return LMEM_BANDWIDTH / double(num_bytes);
}



bool
throughput_t::operator<(const throughput_t & o) const
{
    return throughput < o.throughput;
}


bool
throughput_t::operator>(const throughput_t & o) const
{
    return throughput > o.throughput;
}


throughput_t pipeline_throughput(
        const protos::Network & network, const unsigned bitstream_id)
{
    double frequency = network.frequency() * 1e6;
    double cycle_length = 1.0 / frequency;
    double size =
        network.layer(0).input_height() * network.layer(0).input_width();

    throughput_t ret;
    ret.throughput = bandwidth_throughput_limit(network, bitstream_id);
    ret.bottleneck_type = BOTTLENECK_IO;
    int prev_fpga = 0;

    std::vector<protos::LayerParameter> relevant_layers;
    for (int i = 0 ; i < network.layer_size() ; i++) {
        auto layer = network.layer(i);
        const bool should_do_layer =
            bitstream_id == -1
            || !network.allow_runtime_reconfiguration()
            || bitstream_id == layer.bitstream_id();
        if (should_do_layer) {
            relevant_layers.push_back(network.layer(i));
        }
    }

    for (int i = 0 ; i < relevant_layers.size() ; i++) {
        auto layer = relevant_layers[i];

        const double input_size =
                layer.input_height() * layer.input_width();

        const double output_size =
                layer.output_height() * layer.output_width();

        double layer_throughput_ = 0.0;

        if (layer.has_conv()) {
            const double scheduler_throughput =
                1.0 / (scheduler_iterations(layer) * input_size);

            const double computation_throughput =
                1.0 / (calculation::total_iterations(layer) * output_size);

            layer_throughput_ = std::min(
                    scheduler_throughput, computation_throughput);

        } else if (layer.has_pool()) {
            layer_throughput_ =
                1.0 / (calculation::total_iterations(layer) * input_size);

        } else if (layer.has_lrn()) {
            layer_throughput_ =
                1.0 / (calculation::total_iterations(layer) * input_size);

        } else {
            assert(false);

        }

        layer_throughput_ = layer_throughput_ * frequency;
        throughput_t layer_throughput;
        layer_throughput.throughput = layer_throughput_ ,
        layer_throughput.bottleneck_type = BOTTLENECK_COMPUTE,
        layer_throughput.bottleneck.compute.layer_id = layer.layer_id();

        ret = std::min(ret, layer_throughput);

        if (layer.fpga_id() != prev_fpga) {
            const uint64_t image_bytes =
                network.layer(i).input_height()
                * network.layer(i).input_width()
                * network.layer(i).num_inputs()
                * sizeof(fpgaconvnet::fixed_point_t);
            const double maxring_throughput_ =
                double(fpgaconvnet::calculation::MAXRING_BANDWIDTH)
                / double(image_bytes);
            throughput_t maxring_throughput;
            maxring_throughput.throughput = maxring_throughput_,
            maxring_throughput.bottleneck_type = BOTTLENECK_MAXRING,
            maxring_throughput.bottleneck.maxring.layer_id = 
                    relevant_layers[i - 1].layer_id();
            

            ret = std::min(ret, maxring_throughput);
        }
        prev_fpga = layer.fpga_id();
    }


    return ret;
}


unsigned optimal_num_parallel_pipelines(
    const protos::Network & reference_network,
    unsigned bitstream_id)
{
    protos::Network network;
    if (bitstream_id == -1) {
        network = reference_network;
    } else {
        network = split_by_bitstreams(reference_network)[bitstream_id];
    }

    throughput_t t = pipeline_throughput(
          network, bitstream_id);
    return std::min(
            std::floor(double(network.num_fpga_available()) / network.num_fpga_used()),
            std::ceil(bandwidth_throughput_limit(network, bitstream_id) / t.throughput));
}


throughput_t
effective_throughput(const protos::Network & network, const unsigned bitstream_id)
{
    unsigned num_parallel_pipelines = optimal_num_parallel_pipelines(
            network, bitstream_id);
    throughput_t pt = pipeline_throughput(network, bitstream_id);

    if (num_parallel_pipelines * pt.throughput < bandwidth_throughput_limit(network, bitstream_id)) {
      pt.throughput = pt.throughput * num_parallel_pipelines;
      return pt;
    } else {
      throughput_t ret;
      ret.throughput = bandwidth_throughput_limit(network, bitstream_id);
      ret.bottleneck_type = BOTTLENECK_IO;
      
      return ret;
    }
}


double
real_throughput(const protos::Network & network)
{
    if (network.allow_runtime_reconfiguration()) {
        double inverse_throughput = 0.0;
        for (int i = 0; i <= network.layer(network.layer_size() - 1).bitstream_id() ; i++) {
            inverse_throughput += 1.0 / effective_throughput(network, i).throughput;
        }
        return 1.0 / inverse_throughput;
    } else {
        return effective_throughput(network, -1).throughput;
    }
}


double min_num_fpga_real_throughput(const protos::Network & ref)
{
  auto network = ref;
  network.set_num_fpga_available(min_num_fpga_needed(ref));
  return real_throughput(network);
}


void explain_throughput(const protos::Network & network)
{
    using namespace fpgaconvnet;
    const double total_ops = fpgaconvnet::calculation::ops(network);

    logging::stdout()
        << "Projected Image Throughput (with "
        << network.num_fpga_available()
        << " FPGAS) = "
        << real_throughput(network)
        << " images/s\n";
    logging::stdout()
        << "Projected Ops Throughput (with "
        << network.num_fpga_available()
        << " FPGAS) = "
        << real_throughput(network) * total_ops * 1e-9
        << " GOp/s\n";
    fpgaconvnet::logging::stdout()
        << "Projected Image Throughput (with "
        << fpgaconvnet::calculation::min_num_fpga_needed(network)
        << " FPGAs) = "
        << min_num_fpga_real_throughput(network)
        << " images/s\n";
    logging::stdout()
        << "Projected Ops Throughput (with "
        << fpgaconvnet::calculation::min_num_fpga_needed(network)
        << " FPGAS) = "
        << min_num_fpga_real_throughput(network) * total_ops * 1e-9
        << " GOp/s\n";

    logging::Indentation indent;

    for (int i = 0; i <= network.layer(network.layer_size() - 1).bitstream_id() ; i++) {
        logging::stdout() << "Bitstream " << i << "\n";
        logging::Indentation indent;

        const auto effective_throughput =
            fpgaconvnet::calculation::effective_throughput(network, i);
        const auto pipeline_throughput =
            fpgaconvnet::calculation::pipeline_throughput(network, i);

        logging::stdout()
            << "Num parallel pipelines " << optimal_num_parallel_pipelines(network, i) << "\n";
        logging::stdout()
            << "Pipeline Throughput "
            << pipeline_throughput
            << "\n";
        logging::stdout()
            << "Effective Throughput = "
            << effective_throughput
            << "\n";
    }

    // TODO
}

unsigned
min_num_fpga_needed(const protos::Network & network)
{
  const auto subnetworks = split_by_bitstreams(network);
  assert(subnetworks.size() > 0);
  unsigned best = 1;
  for (int i = 0; i < subnetworks.size() ; i++) {
    best = std::max(best, subnetworks[i].num_fpga_used());
  }
  return best;
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

    assert(false);
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

protos::Network
insert_fpga_positions(protos::Network network, std::vector<int> v)
{
    unsigned num_fpga_used = 1;

    for (unsigned i = 0 ;i < network.layer_size() ; i++) {
        assert (0 <= v[i] && v[i] < network.num_fpga_available());
        network.mutable_layer(i)->set_fpga_id(v[i]);
        num_fpga_used = std::max(num_fpga_used, v[i] + 1u);
    }
    network.set_num_fpga_used(num_fpga_used);
    return network;
}

std::vector<protos::Network>
split_by_bitstreams(protos::Network ref)
{
    std::vector<protos::Network> subnetworks(
            ref.layer(ref.layer_size() - 1).bitstream_id() + 1, ref);

    for (unsigned i = 0 ; i < subnetworks.size() ; i++) {
        subnetworks[i].clear_layer();
    }

    for (auto it = ref.layer().begin(); it != ref.layer().end() ; it++) {
        unsigned i = it->bitstream_id();
        *subnetworks[i].add_layer() = *it;

        if (subnetworks[i].has_num_fpga_used()) {
            subnetworks[i].set_num_fpga_used(
                std::max(subnetworks[i].num_fpga_used(), it->fpga_id() + 1));
        } else {
            subnetworks[i].set_num_fpga_used(it->fpga_id() + 1);
        }
    }
    return subnetworks;
}


}  // fpgaconvnet

static std::ostream&
print_bottleneck(std::ostream & o, const fpgaconvnet::calculation::throughput_t & t)
{
    using namespace fpgaconvnet::calculation;

    switch (t.bottleneck_type) {
      case BOTTLENECK_COMPUTE:
        return o << "COMPUTE[Layer " << t.bottleneck.compute.layer_id << "]";
      
      case BOTTLENECK_IO:
        return o << "IO";

      case BOTTLENECK_MAXRING:
        return o << "MAXRING[Layer " << t.bottleneck.maxring.layer_id << "]";

      default:
        assert(false);
    }
}


std::ostream&
operator<<(std::ostream & o, const fpgaconvnet::calculation::throughput_t & t)
{
    o << t.throughput << " images / s (Bottleneck = ";
    return print_bottleneck(o, t) << ")";
}

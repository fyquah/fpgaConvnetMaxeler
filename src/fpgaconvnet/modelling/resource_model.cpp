#include <cmath>
#include <cstring>

#include <sstream>
#include <vector>

#include <fpgaconvnet/modelling/resource_model.h>


namespace fpgaconvnet {
namespace resource_model {

static resource_t
resource_add(const resource_t left, const resource_t right)
{
    resource_t ret = {
        .lut  = left.lut + right.lut,
        .bram = left.bram + right.bram,
        .dsp  = left.dsp + right.dsp,
    };
    return ret;
}


std::string
resource_to_string(const resource_t & res)
{
    thread_local static char buffer[1000];
    std::sprintf(
            buffer, "[resource lut = %.3f, dsp = %.3f, bram = %.3f]",
            res.lut, res.dsp, res.bram);
    return std::string(buffer);
}

std::string
resource_to_string(const std::vector<resource_t> & resources)
{
    std::stringstream ss;

    for (int i = 0 ; i < resources.size() ; i++) {
        ss << "- [fpga "  << i << ": "
            << resource_to_string(resources[i]) << "]\n";
    }

    return ss.str();
}


static resource_t
conv_resource(const protos::LayerParameter & layer)
{
    assert(sizeof(fixed_point_t) == 2);

    const uint64_t wf  = layer.conv().worker_factor();
    const uint64_t cff = layer.conv().conv_folding_factor();
    const uint64_t kff = layer.conv().kernel_folding_factor();
    const uint64_t kernel_area =
            layer.conv().kernel_size() * layer.conv().kernel_size();

    const double dsp = double(
            layer.conv().worker_factor()
            * layer.conv().conv_folding_factor()
            * layer.conv().kernel_folding_factor()) / 1.7;

    /* - Assume a constant for sliding window - should be correct for most
     *   of the case. In fact, it is overrly pessismitic.
     * - Calculation for weights BRAM usage assumes 16 bit fixed point.
     */
    const double bram_sliding_window = 100.0;
    const double bram_stream =
            512.0
            * layer.conv().worker_factor()
            * layer.conv().kernel_size() * layer.conv().kernel_size()
            / 1024.0;

    const double weights_vector_bits =
        layer.conv().worker_factor() * layer.conv().conv_folding_factor()
        * layer.conv().kernel_folding_factor()
        * sizeof(fixed_point_t) * 2.0;

    const double bram_kernels_width = std::ceil(weights_vector_bits / 40.0);
    const double bram_kernels_depth =
            std::ceil(calculation::total_iterations(layer) / 1024.0);

    const double bram_kernels = bram_kernels_depth * bram_kernels_width;

    const double bram_accumulator =
        std::ceil(layer.num_outputs() * sizeof(fixed_point_t) * 8.0
                / 20480.0);

    const double bram =
        bram_sliding_window
        + bram_stream + bram_kernels + bram_accumulator;

    const double lut_scheduler =
            (4718.0 * layer.num_inputs()) / wf
            + 621.66 * wf
            - 3575.2;
    const double lut_unit =
            0.26559 * wf * kernel_area
            + 24.505 * wf * cff * kff
            + 3754.3;
    const double lut_acc =
            + 49.874 * wf * cff
            - 456.46 * cff
            + 7595.4;
    const double lut_streams = (2 * wf + 1) * (402 + 100);
    const double lut = lut_scheduler + lut_streams + lut_unit + lut_acc;

    resource_t ret =  {.lut = lut , .bram = bram, .dsp = dsp};
    return ret;
}

static resource_t
pool_resource(const protos::LayerParameter & layer)
{
    const uint64_t dim = layer.pool().dim();
    const uint64_t channel_folding_factor =
        layer.pool().channel_folding_factor();

    const double bram =
        layer.input_width()
        * 20.0
        * layer.num_inputs() / 20480.0 * std::log2(channel_folding_factor);

    const double lut =
        34.9 * dim * dim * channel_folding_factor
        + 1.6152 * layer.input_width() * layer.num_inputs();

    resource_t ret = {.lut = lut, .bram = bram, .dsp = double(0.0)};
    return ret;
}

static resource_t
lrn_resource(const protos::LayerParameter & layer)
{
    const uint64_t dim = layer.lrn().local_size();
    const uint64_t channel_folding_factor =
        layer.lrn().channel_folding_factor();

    const double bram =
        layer.input_width()
        * 20.0
        * layer.num_inputs() / 20480.0 * std::log2(channel_folding_factor);

    const double lut =
        34.9 * dim * dim * channel_folding_factor
        + 1.6152 * layer.input_width() * layer.num_inputs();

    resource_t resource = {.lut = lut, .bram = bram, .dsp = 0.0};

    resource.dsp = channel_folding_factor;
    return resource;
}


static resource_t
project_single_fpga(
        const stream_t input_stream,
        const std::vector<protos::LayerParameter> & layers,
        const stream_t output_stream) 
{
    resource_t resource = {.lut = 0.0, .bram = 0.0, .dsp = 0.0};

    for (int i = 0 ; i < layers.size() ; i++) {
        const protos::LayerParameter layer = layers[i];

        /*
         * Incoming stream resources 
         * TODO(fyq14): The resource usage should be dependent on the
         *              previous stream width as well (DualAspectReg
         *              or something like that)
         */
        double instream_width = 0.0;

        if (layer.has_conv()) {
            instream_width = layer.conv().worker_factor();

        } else if (layer.has_pool() || layer.has_lrn()) {
            instream_width = layer.num_inputs();

        }

        resource.bram += std::ceil(instream_width * 0.5);

        /*
         * Kernel resorurces
         */
        if (layer.has_conv()) {
            resource = resource_add(resource, conv_resource(layer));

        } else if (layer.has_pool()) {
            resource = resource_add(resource, pool_resource(layer));

        } else if (layer.has_lrn()) {
            resource = resource_add(resource, lrn_resource(layer));

        }
    }

    /*
     * Outgoing stream resources
     * TODO(fyq14): The resource usage should be dependent on the
     *              previous stream width as well (DualAspectReg
     *              or something like that)
     */
    double outstream_width = 0.0;

    if (layers.back().has_conv()) {
        outstream_width = layers.back().conv().worker_factor();

    } else if (layers.back().has_pool() || layers.back().has_lrn()) {
        outstream_width = layers.back().num_inputs();

    }

    resource.bram += std::ceil(outstream_width * 0.5);

    return resource;
}


std::vector<resource_t>
project(const protos::Network & network)
{
    std::vector<std::vector<fpgaconvnet::protos::LayerParameter>>
        layers_by_fpga(network.num_fpga_used());
    std::vector<resource_t> resources;

    for (auto & layer : network.layer()) {
        layers_by_fpga[layer.fpga_id()].push_back(layer);
    }

    for (int i = 0; i < layers_by_fpga.size() ; i++) {
        auto & layers = layers_by_fpga[i];
        const stream_t input_stream =
            (i == 0) ? STREAM_PCIE : STREAM_MAX_RING;

        const stream_t output_stream =
            (i == network.num_fpga_used()) ? STREAM_PCIE : STREAM_MAX_RING;

        resources.push_back(project_single_fpga(
                input_stream, layers, output_stream));

    }

    return resources;
}


bool
meets_resource_constraints(const std::vector<resource_t> & resources)
{
    for (const resource_t & resource : resources) {
        if (resource.dsp > MAX_DSP
                || resource.dsp > MAX_BRAM
                || resource.lut >= 0.6 * MAX_LUT) {
            return false;
        }
    }

    return true;
}

}  // resource_model
}  // fpgaconvnet

#ifndef FPGACONVNET_RESOURCE_MODEL_H
#define FPGACONVNET_RESOURCE_MODEL_H

#include <fpgaconvnet/common.h>


namespace fpgaconvnet {
namespace resource_model {

struct resource_t {
    double lut;
    double bram;  /* In M20k units */
    double dsp;   /* In DSP units */
};

enum stream_t {
    STREAM_MAX_RING,
    STREAM_PCIE
};


const double MAX_DSP = 1963;
const double MAX_BRAM = 2567;
const double MAX_LUT = 524800;
const double MAX_FF = 1049600;


bool
meets_resource_constraints(
    const protos::OptimizerOptions & optimiser_options,
    const std::vector<resource_t> & resources);


bool
meets_resource_constraints(
    const protos::OptimizerOptions & optimiser_options,
    const resource_t & resource);


resource_t
project_single_fpga(
        const stream_t input_stream,
        const std::vector<protos::LayerParameter> & layers,
        const stream_t output_stream);

bool
possible_to_fit(
    const protos::OptimizerOptions & optimiser_options,
    const std::vector<protos::LayerParameter> & layers);

/* Assumes that the argumet network is meant for only one bitstream. */
std::vector<resource_t> project_single_bitstream(
        const protos::Network & network);

std::string resource_to_string(const resource_t & res);


}  // resource_model
}  // fpgaconvnet


#endif

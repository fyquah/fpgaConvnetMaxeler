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
meets_resource_constraints(const std::vector<resource_t> & resources);

std::vector<resource_t> project(const protos::Network & network);

std::string
resource_to_string(const std::vector<resource_t> & resources);

std::string resource_to_string(const resource_t & res);


}  // resource_model
}  // fpgaconvnet


#endif

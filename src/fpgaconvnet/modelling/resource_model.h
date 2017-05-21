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
meets_resource_constraints(const fpgaconvnet::protos::Network & network);


}  // resource_model
}  // fpgaconvnet


#endif

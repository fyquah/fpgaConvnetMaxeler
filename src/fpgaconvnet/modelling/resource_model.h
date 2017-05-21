#ifndef FPGACONVNET_RESOURCE_MODEL_H
#define FPGACONVNET_RESOURCE_MODEL_H

#include <fpgaconvnet/common.h>


namespace fpgaconvnet {
namespace resource_model {


const double MAX_DSP = 1963;
const double MAX_BRAM = 2567;
const double MAX_LUT = 524800;
const double MAX_FF = 1049600;

bool meets_resource_constraints(const fpgaconvnet::protos::Network & network);


}  // resource_model
}  // fpgaconvnet


#endif

#ifndef SEARCH_CONFIGURATION_H
#define SEARCH_CONFIGURATION_H

#include <fpgaconvnet/common.h>

namespace fpgaconvnet {
namespace modelling {

fpgaconvnet::protos::Network
search_design_space_for_bitstream_with_fixed_num_fpga(
        const fpgaconvnet::protos::Network & network,
        bool *success,
        const int num_fpga);

}  // modelling
}  // fpgaconvnet


#endif

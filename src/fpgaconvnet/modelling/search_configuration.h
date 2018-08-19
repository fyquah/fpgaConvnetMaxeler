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

/* Reconfigures the network parameters to achieve a target throughput.
 * The target throughput can be either higher or lower than the network's
 * current throughput. The semantics of the seach is carried out using
 * by binary searching N_{ref}
 */
fpgaconvnet::protos::Network
reconfigure_from_layer_id(
        const fpgaconvnet::protos::Network & network,
        const unsigned layer_id,
        const calculation::throughput_t & target_throughput);

}  // modelling
}  // fpgaconvnet


#endif

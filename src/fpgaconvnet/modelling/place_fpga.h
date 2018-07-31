#ifndef PLACE_FPGA_H
#define PLACE_FPGA_H

#include <fpgaconvnet/common.h>

namespace fpgaconvnet {
namespace modelling {

class PositionFpga {
private:
  const fpgaconvnet::protos::Network reference_network;
  const unsigned num_fpga_;

  bool done;
  unsigned considered_solutions;
  unsigned accepted_solutions;
  std::vector<std::vector<int>> solutions;
  void search_recur(std::vector<int>);
public:
  PositionFpga(fpgaconvnet::protos::Network network, const unsigned num_fpga);
  void search();
  unsigned get_num_accepted_solutions();
  unsigned get_num_considered_solutions();
  std::vector<std::vector<int>> get_solutions();  /* TODO: Make this a lazy stream instead? */
};


}  // modelling
}  // fpgaconvnet

#endif

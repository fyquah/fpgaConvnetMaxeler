#ifndef PLACE_BISTREAM_H
#define PLACE_BISTREAM_H

#include <unordered_map>

#include <fpgaconvnet/common.h>

namespace fpgaconvnet {
namespace modelling {

// Cache using this, rather than the entire protobuf, to conserve memory,
// and have better cache performance
struct layer_factors {
  uint32_t wf;
  uint32_t cf;
  uint32_t kff;
};

struct bitstream_solution {
  std::vector<int> fpga_placement;
  std::vector<layer_factors> layer_config;
  double pipeline_throughput;
  uint32_t get_num_fpga_used();
};

class PlaceBitstream {
private:
  const fpgaconvnet::protos::Network reference_network;
  bool done;
  unsigned considered_solutions;
  unsigned accepted_solutions;

  std::vector<std::vector<int>> solutions;
  void search_recur(std::vector<int>);
public:
  PlaceBitstream(fpgaconvnet::protos::Network network);
  void search();
  unsigned get_num_accepted_solutions();
  unsigned get_num_considered_solutions();
  std::vector<std::vector<int>> get_solutions();  /* TODO: Make this a lazy stream instead? */
};


}  // modelling
}  // fpgaconvnet

#endif

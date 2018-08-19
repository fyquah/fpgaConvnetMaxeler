#ifndef PLACE_BISTREAM_H
#define PLACE_BISTREAM_H

#include <map>
#include <memory>
#include <mutex>

#include <fpgaconvnet/common.h>

namespace fpgaconvnet {
namespace modelling {

// Cache using this, rather than the entire protobuf, to conserve memory,
// and have better cache performance
struct layer_config_t {
    uint32_t wf;
    uint32_t cff;
    uint32_t kff;
    uint32_t channel_folding_factor;
    uint32_t fpga_id;
};

struct bitstream_solution_t {
  bool success;
  std::vector<layer_config_t> layer_config;
  double pipeline_throughput;
  double effective_throughput;

  uint32_t get_num_fpga_used();
};

struct placement_solution_t {
  std::vector<int> placement;
  double           real_throughput;
};


class PlaceBitstream {
private:
  const fpgaconvnet::protos::Network reference_network;
  bool done;
  unsigned considered_solutions;
  unsigned accepted_solutions;
  std::vector<placement_solution_t> solutions;
  std::map<std::pair<int, int>, bitstream_solution_t> bitstreams_cache;
  std::mutex m_bitstream_cache_wrt_lock;

  void search_recur(std::vector<int>);
  bitstream_solution_t get_local_solution(
          unsigned start_inclusive, unsigned end_inclusive);
  void add_solution_to_collection(std::vector<int> v, double throughput);

public:
  PlaceBitstream(fpgaconvnet::protos::Network network);
  bool search();
  unsigned get_num_accepted_solutions();
  unsigned get_num_considered_solutions();

  /* [get_solutions] doesn't necessarily return the all the solutions, but
   * is guranteed to return the best solution seen (before receiving a
   * SIGABRT signal)
   */
  std::vector<placement_solution_t> get_solutions();  
  placement_solution_t get_best_solution();

  /* Translates solution into a protobuf that can be readily be parsed
   * by the next stage of the compilation pipeline. The argument vector
   * should have the same lenght as the number of layers of the CNN.
   */
  fpgaconvnet::protos::Network translate_placement_to_protobuf(const std::vector <int> & v);
};


}  // modelling
}  // fpgaconvnet

#endif

#include <algorithm>
#include <iostream>

#include "assert.h"

#include <fpgaconvnet/modelling/resource_model.h>
#include <fpgaconvnet/modelling/place_fpga.h>


namespace fpgaconvnet {
namespace modelling {

static inline fpgaconvnet::protos::Network
reconfigure_if_bottleneck_on_latest_maxring(
    const fpgaconvnet::protos::Network & network,
    std::vector<int> v
)
{
  assert(v.size() > 1);

  /* The maxring connection is between prev_layer_id and next_layer_id
   * by convention, the maxring connection belongs to prev_layer_id.
   * We start reconfiguration from next_layer_id
   */
  const unsigned prev_layer_id = network.layer(v.size() - 2).layer_id();
  const unsigned next_layer_id = network.layer(v.size() - 1).layer_id();

  while (v.size() < network.layer_size()) {
    v.push_back(v.back());
  }
  for (int i = 0; i < network.layer_size() ; i++) {
    assert(!network.layer(i).has_fpga_id());
  }

  const calculation::throughput_t original_throughput =
      fpgaconvnet::calculation::pipeline_throughput(
          insert_fpga_positions(network, v), -1);
  const bool is_maxring_bottleneck =
      original_throughput.bottleneck_type == calculation::BOTTLENECK_MAXRING
      && original_throughput.bottleneck.maxring.layer_id == prev_layer_id;

  if (is_maxring_bottleneck) {
    logging::stdout(logging::INFO)
      << "Reconfiguring to meet maxring "
      << prev_layer_id << " -> "<< next_layer_id
      << " throughput of "
      << original_throughput
      << std::endl;

    logging::Indentation indent;
    auto ret = reconfigure_from_layer_id(
        network, next_layer_id, original_throughput);
    auto new_throughput = fpgaconvnet::calculation::pipeline_throughput(
        insert_fpga_positions(ret, v), -1);
    assert(new_throughput.bottleneck_type == calculation::BOTTLENECK_MAXRING);
    return ret;
  } else {
    return network;
  }
}

void PositionFpga::search_recur(
    const fpgaconvnet::protos::Network & network,
    std::vector<int> v)
{
    assert (network.layer_size() == reference_network.layer_size());
    assert (v.size() <= network.layer_size());
    assert (v.size() > 0);

    if (v.size() == network.layer_size()) {
        // See paper on why we cna't search for multiple FPGA's
        // simultaneously at a point in binary search.
        if (v.back() + 1 != num_fpga_) {
            return;
        }

        auto resource = fpgaconvnet::resource_model::project_single_bitstream(
                fpgaconvnet::insert_fpga_positions(network, v));

        considered_solutions++;
        if (fpgaconvnet::resource_model::meets_resource_constraints(resource)) {
            solutions.push_back(v);
            accepted_solutions++;
        }
        return;
    }

    v.push_back(v.back());
    {
      std::vector<fpgaconvnet::protos::LayerParameter> layers;
      for (int i = v.size() - 1; i >= 0 ; i--) {
        if (v[i] != v.back()) {
          break;
        } else {
          layers.push_back(network.layer(i));
        }
      }
      std::reverse(layers.begin(), layers.end());

      std::vector<fpgaconvnet::resource_model::resource_t> resources;
      resources.push_back(fpgaconvnet::resource_model::project_single_fpga(
          fpgaconvnet::resource_model::STREAM_MAX_RING,
          layers,
          fpgaconvnet::resource_model::STREAM_MAX_RING));
      resources.push_back(fpgaconvnet::resource_model::project_single_fpga(
          fpgaconvnet::resource_model::STREAM_PCIE,
          layers,
          fpgaconvnet::resource_model::STREAM_MAX_RING));
      resources.push_back(fpgaconvnet::resource_model::project_single_fpga(
          fpgaconvnet::resource_model::STREAM_MAX_RING,
          layers,
          fpgaconvnet::resource_model::STREAM_PCIE));
      resources.push_back(fpgaconvnet::resource_model::project_single_fpga(
          fpgaconvnet::resource_model::STREAM_PCIE,
          layers,
          fpgaconvnet::resource_model::STREAM_PCIE));

      for (int i = 0; i < resources.size() ; i++) {
        if (fpgaconvnet::resource_model::meets_resource_constraints(resources[i])) {
          search_recur(network, v);
          break;
        }
      }
    }
    v.pop_back();

    if (v.back() < int(num_fpga_) - 1) {
        v.push_back(v.back() + 1);
        search_recur(
            reconfigure_if_bottleneck_on_latest_maxring(network, v),
            v);
        v.pop_back();
    }
}

PositionFpga::PositionFpga(fpgaconvnet::protos::Network network, unsigned num_fpga)
    : reference_network(network), num_fpga_(num_fpga)
{
    done = false;
    considered_solutions = 0;
    accepted_solutions = 0;
}

void PositionFpga::search()
{
    assert(!done);

    auto v = std::vector<int>();
    v.push_back(0);
    search_recur(reference_network, v);
    done = true;
}

unsigned
PositionFpga::get_num_accepted_solutions()
{
    return accepted_solutions;
}

unsigned
PositionFpga::get_num_considered_solutions()
{
    return considered_solutions;
}

std::vector<std::vector<int>>
PositionFpga::get_solutions()
{
    return solutions;
}

}  // modelling
}  // fpgaconvnet

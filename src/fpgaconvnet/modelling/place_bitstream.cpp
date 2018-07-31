#include <iostream>

#include "assert.h"

#include <fpgaconvnet/modelling/resource_model.h>
#include <fpgaconvnet/modelling/place_bitstream.h>


namespace fpgaconvnet {
namespace modelling {

static unsigned
count_num_bitstreams(std::vector<int> v)
{
  unsigned a = 0;
  for (unsigned i = 0; i < v.size(); i++) {
    a = std::max(a, unsigned(v[i] + 1));
  }
  assert(a != 0);
  return a;
}

uint32_t
bitstream_solution::get_num_fpga_used()
{
  return count_num_bitstreams(fpga_placement);
}

// TODO: Consider using TBB here?
void PlaceBitstream::search_recur(std::vector<int> v)
{
    assert (v.size() <= reference_network.layer_size());
    assert (v.size() > 0);

    if (v.size() == reference_network.layer_size()) {
        std::vector<fpgaconvnet::protos::Network> subnetworks(
              count_num_bitstreams(v));

        auto network = reference_network;
        network.clear_layer();

        for (unsigned i = 0; i < v.size() ; i++) {
            const unsigned b = v[i];
            auto ptr = subnetworks[b].add_layer();
            *ptr = reference_network.layer(i);
        }

       // TODO(fyq14): Invoke code to search here
    }

    v.push_back(v.back());
    if (true) {
        // TODO(fyq14): check resource constraints here to prune
        //              search space here.
        search_recur(v);
    }
    v.pop_back();

    v.push_back(v.back() + 1);
    search_recur(v);
    v.pop_back();
}

PlaceBitstream::PlaceBitstream(fpgaconvnet::protos::Network network)
    : reference_network(network)
{
    done = false;
    considered_solutions = 0;
    accepted_solutions = 0;
}

void PlaceBitstream::search()
{
    assert(!done);

    auto v = std::vector<int>();
    v.push_back(0);
    search_recur(v);
    done = true;
}

unsigned
PlaceBitstream::get_num_accepted_solutions()
{
    return accepted_solutions;
}

unsigned
PlaceBitstream::get_num_considered_solutions()
{
    return considered_solutions;
}

std::vector<std::vector<int>>
PlaceBitstream::get_solutions()
{
    return solutions;
}

}  // modelling
}  // fpgaconvnet

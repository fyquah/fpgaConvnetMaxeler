#include <iostream>
#include <utility>
#include <memory>
#include <sstream>

#include "assert.h"

#include <fpgaconvnet/common.h>
#include <fpgaconvnet/modelling/build_single_bitstream.h>
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
bitstream_solution_t::get_num_fpga_used()
{
  std::vector<int> v;
  for (int i = 0 ; i< layer_config.size() ; i++) {
      v.push_back(layer_config[i].fpga_id);
  }
  return count_num_bitstreams(v);
}

void
PlaceBitstream::add_solution_to_collection(const std::vector<int> v, const double real_throughput)
{
    placement_solution_t new_entry;
    new_entry.placement = v;
    new_entry.real_throughput = real_throughput;

    // poor man's Binary search tree
    for (int i = 0 ; i < solutions.size() ; i++) {
        if (new_entry.real_throughput > solutions[i].real_throughput) {
            solutions.insert(solutions.begin() + i, new_entry);

            if (solutions.size() > 100) {
                solutions.resize(100);
                return;
            }
        }
    }

    if (solutions.size() < 100) {
        solutions.push_back(new_entry);
    }
}

bitstream_solution_t
PlaceBitstream::get_local_solution(unsigned start_inclusive, unsigned end_inclusive)
{
    if (bitstreams_cache.count(std::make_pair(start_inclusive, end_inclusive))) {
        fpgaconvnet::logging::stdout()
            << "Found cached solution for layers "
            << start_inclusive << " to " << end_inclusive
            << "\n";
        return bitstreams_cache[std::make_pair(start_inclusive, end_inclusive)];
    }
    auto subnetwork = reference_network;
    subnetwork.clear_layer();
    for (int i = start_inclusive ; i <= end_inclusive ; i++) {
        *subnetwork.add_layer() = 
            reference_network.layer(i);
    }

    auto b = BuildSingleBitStream(subnetwork);
    bitstream_solution_t ret;

    if (!b.search()) {
        ret.success = false;
        return ret;
    }

    auto optimized_subnetwork = b.get_result();
    ret.success = true;
    ret.pipeline_throughput  = calculation::pipeline_throughput(
            optimized_subnetwork,  -1).throughput;
    ret.effective_throughput = calculation::effective_throughput(
            optimized_subnetwork, -1).throughput;

    for (int i = 0; i < optimized_subnetwork.layer_size(); i++) {
        layer_config_t layer_config;
        const auto layer = optimized_subnetwork.layer(i);

        layer_config.fpga_id = layer.fpga_id();

        if (layer.has_conv()) {
            layer_config.wf  = layer.conv().worker_factor();
            layer_config.cff = layer.conv().conv_folding_factor();
            layer_config.kff = layer.conv().kernel_folding_factor();

        } else if (layer.has_pool()) {
            layer_config.channel_folding_factor = layer.pool().channel_folding_factor();

        } else if (layer.has_lrn()) {
            layer_config.channel_folding_factor = layer.lrn().channel_folding_factor();

        } else {
            assert(false);

        }

        ret.layer_config.push_back(layer_config);
    }

    std::lock_guard<std::mutex> lck(m_bitstream_cache_wrt_lock);
    bitstreams_cache[std::make_pair(start_inclusive, end_inclusive)] = ret;
    return ret;
}

std::vector<std::pair<int, int> >
alloc_list_to_range_list(const std::vector<int> alloc_list)
{
    std::vector<std::pair<int ,int> > ret;
    unsigned begin = 0;

    for (int i = 0 ; i < alloc_list.size() ; i++) {
        if (i == alloc_list.size() - 1 || alloc_list[i] != alloc_list[i+1]) {
            ret.push_back(std::make_pair(begin, i));
            begin = i + 1;
        }
    }

    return ret;
}

static std::string
print_vector(std::vector<int> v)
{
    std::stringstream ss;
    ss << "[";
    for (int i = 0 ; i < v.size() ; i++) {
        ss << v[i];
        if (i != v.size() - 1) {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}

// TODO: Consider using TBB here?
void PlaceBitstream::search_recur(std::vector<int> v)
{
    assert (v.size() <= reference_network.layer_size());
    assert (v.size() > 0);

    if (v.size() == reference_network.layer_size()) {
        fpgaconvnet::logging::stdout()
            << "TRYING BITSTREAM CONFIGURATION: " << print_vector(v) << "\n";
        logging::Indentation indent;

        // TBB Task group should encapsulate this block, all the way to return
        // Solving ...
        auto range_list = alloc_list_to_range_list(v);
        bool this_success = true;
        double inverse_throughput = 0.0;
        std::vector<double> effective_throughputs;

        for (int i = 0 ; i < range_list.size() ; i++) {
            fpgaconvnet::logging::stdout()
                << "SOLVING FOR BISTREAM " << i
                << " (Layers "
                << range_list[i].first
                << " to (inclusive) "
                << range_list[i].second
                << "):\n";
            logging::Indentation indent;

            auto bistream_solution = get_local_solution(
                    range_list[i].first, range_list[i].second);
            if (bistream_solution.success == false) {
                this_success = false;
                break;
            }

            effective_throughputs.push_back(bistream_solution.effective_throughput);
            inverse_throughput += 1.0 / bistream_solution.effective_throughput;
        }

        if (this_success) {
            const double throughput = 1.0 / inverse_throughput;

            fpgaconvnet::logging::stdout()
                << "ACCEPTED CONFIGURATION: " << print_vector(v) << "\n";
            for (int i = 0 ; i < effective_throughputs.size() ; i++) {
                fpgaconvnet::logging::stdout()
                    << "EFFECTIVE THROUGHPUT " << i << " = "
                    << effective_throughputs[i]
                    << "\n";
            }
            fpgaconvnet::logging::stdout() << "REAL THROUGHPUT " << " = " << throughput << "\n";
            add_solution_to_collection(v, throughput);
        }
        fpgaconvnet::logging::stdout() << "\n";

        return;
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

bool PlaceBitstream::search()
{
    assert(!done);

    auto v = std::vector<int>();
    v.push_back(0);
    search_recur(v);
    done = true;
    return solutions.size() != 0;
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

std::vector<placement_solution_t>
PlaceBitstream::get_solutions()
{
    return solutions;
}

placement_solution_t
PlaceBitstream::get_best_solution()
{
    assert(solutions.size() > 0);
    auto best = solutions[0];

    for (unsigned i = 1; i < solutions.size() ; i++) {
        auto candidate = solutions[i];

        if (candidate.real_throughput > best.real_throughput) {
            best = candidate;
        }
    }

    return best;
}


fpgaconvnet::protos::Network
PlaceBitstream::translate_placement_to_protobuf(const std::vector<int> & v)
{
    assert(v.size() == reference_network.layer_size());
    for (int i = 0 ; i < v.size() ; i++) {
        if (i == 0) {
            assert(v[i] == 0);
        } else {
            assert(v[i] == v[i -1] || v[i] == v[i - 1] + 1);
        }
    }

    auto ret = reference_network;
    unsigned begin = 0;
    for (int i = 0; i < v.size() ; i++) {
        assert (v[i] == v[begin]);
        if (i == v.size() - 1 || v[i + 1] != v[i]) {
            auto key = std::make_pair(begin, i);
            assert(bitstreams_cache.count(key));
            auto local_solution = bitstreams_cache[key].layer_config;

            for (int j = 0; j < local_solution.size() ; j++) {
                auto ptr = ret.mutable_layer(begin + j);
                auto layer_soln = local_solution[j];

                ptr->set_bitstream_id(v[i]);
                ptr->set_fpga_id(layer_soln.fpga_id);

                if (ptr->has_conv()) {
                    ptr->mutable_conv()->set_worker_factor(layer_soln.wf);
                    ptr->mutable_conv()->set_conv_folding_factor(layer_soln.cff);
                    ptr->mutable_conv()->set_kernel_folding_factor(layer_soln.kff);
                } else if (ptr->has_lrn()) {
                    ptr->mutable_lrn()->set_channel_folding_factor(layer_soln.channel_folding_factor);
                } else if (ptr->has_pool()) {
                    ptr->mutable_pool()->set_channel_folding_factor(layer_soln.channel_folding_factor);
                } else {
                    assert(false);
                }
            }

            // prepare for the next bitstream
            begin = i + 1;
        }
    }

    return ret;
}

}  // modelling
}  // fpgaconvnet

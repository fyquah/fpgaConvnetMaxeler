#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>

#include <cmath>
#include <cstring>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <functional>

#include <fpgaconvnet/common.h>
#include <fpgaconvnet/modelling/resource_model.h>
#include <fpgaconvnet/modelling/place_fpga.h>
#include <fpgaconvnet/modelling/place_bitstream.h>
#include <fpgaconvnet/modelling/build_single_bitstream.h>
#include <fpgaconvnet/modelling/search_configuration.h>


static fpgaconvnet::protos::Network
search_design_space_for_bitstream(
        const fpgaconvnet::protos::Network & network,
        bool *p_success)
{
    bool success = false;
    fpgaconvnet::protos::Network solution;

    for (int i = 1; i <= std::min(network.num_fpga_available(), unsigned(network.layer_size())) ; i++) {
        fpgaconvnet::logging::stdout()
            << "Starting to search with " << i << " FPGAs\n";
        fpgaconvnet::logging::Indentation indent;

        bool this_success;
        auto this_solution =
          fpgaconvnet::modelling::search_design_space_for_bitstream_with_fixed_num_fpga(
                network, &this_success, i);

        if (this_success)  {
            auto this_throughput = fpgaconvnet::calculation::pipeline_throughput(
                    this_solution, -1);

            fpgaconvnet::logging::stdout() << "Results of searching with "
                << i << " FPGAs : " << "SUCCESS\n";
            fpgaconvnet::calculation::explain_throughput(this_solution);
            fpgaconvnet::logging::stdout() << "\n";

            if (!success) {
                solution = this_solution;
            } else if(this_throughput > fpgaconvnet::calculation::pipeline_throughput(solution, -1)) {
                solution = this_solution;
            }

            success = 1;
        } else {
            fpgaconvnet::logging::stdout() << "Results of searching with "
                << i << " FPGAs : " << "FAILED\n";
            fpgaconvnet::logging::stdout() << "\n";
        }
    }
    *p_success = success;

    return solution;
}


namespace fpgaconvnet {
namespace modelling {

BuildSingleBitStream::BuildSingleBitStream(fpgaconvnet::protos::Network network)
    : reference_(network), success_(false), done_(false)
{
}


bool BuildSingleBitStream::search()
{
    if (!done_) {
        solution_ = search_design_space_for_bitstream(reference_, &success_);
        done_ = true;
    }
    return success_;
}


fpgaconvnet::protos::Network
BuildSingleBitStream::get_result()
{
    assert(done_ && success_);
    return solution_;
}

}  // modelling
}  // fpgaconvnet

static fpgaconvnet::protos::Network
search_design_space(
    fpgaconvnet::protos::Network network,
    bool *success)
{
    if (!network.allow_runtime_reconfiguration()) {
        auto ret = search_design_space_for_bitstream(network, success);
        if (*success) {
            for (auto it = ret.mutable_layer()->begin()
                    ; it != ret.mutable_layer()->end()
                    ; it++) {
                it->set_bitstream_id(0);
            }
        }
        return ret;
    }

    fpgaconvnet::modelling::PlaceBitstream p(network);
    *success = p.search();

    if (!(*success)) {
        return network;
    }

    *success = 1;
    return p.translate_placement_to_protobuf(p.get_best_solution().placement);
}


int main (int argc, char **argv)
{
    if (const char* debug_level = std::getenv("FPGACONVNET_DEBUG_LEVEL")) {
        fpgaconvnet::logging::set_level(debug_level);
    }

    fpgaconvnet::logging::stdout() << "Loading convnet descriptor:" << std::endl;
    fpgaconvnet::protos::Network network =
	    fpgaconvnet::load_network_proto(argv[1]);
    const char *output_filename = argv[2];

    fpgaconvnet::logging::stdout()
        << "Running Design Space Exploration:"
        << std::endl;
    bool success = false;

    timeval t_begin;
    timeval t_end;
    gettimeofday(&t_begin, NULL);
    fpgaconvnet::protos::Network solution = search_design_space(
          network, &success);
    gettimeofday(&t_end, NULL);
    fpgaconvnet::logging::stdout(fpgaconvnet::logging::INFO)
      << "Design Space exploration took "
      << compute_time_difference(t_begin, t_end) / 1e6
      << " seconds"<< std::endl;;

    if (success) {
        double ops = fpgaconvnet::calculation::ops(solution);
        fpgaconvnet::logging::stdout() << "Found an optimal solution!\n";
        fpgaconvnet::logging::stdout()
            << "Network Operations per image = " << ops << '\n';

        fpgaconvnet::calculation::explain_throughput(solution);

        const auto subnetworks = fpgaconvnet::split_by_bitstreams(solution);
        for (unsigned i = 0; i < subnetworks.size(); i++) {
            auto resources = fpgaconvnet::resource_model::project_single_bitstream(
                subnetworks[i]);

            fpgaconvnet::logging::stdout()
                << "Resource usage (Bitstream " << i << ") :\n";

            for (unsigned j = 0; j < resources.size() ; j++ ) {
                fpgaconvnet::logging::Indentation indent;
                fpgaconvnet::logging::stdout()
                    << "fpga " << j
                    << fpgaconvnet::resource_model::resource_to_string(resources[j])
                    << "\n";
            }
        }
        fpgaconvnet::logging::stdout()
            << "Writing optimized protobuf to "
            << output_filename << std::endl;

        std::string s;
        google::protobuf::TextFormat::PrintToString(solution, &s);
        std::ofstream fout(output_filename);
        fout << s;
        fout.close();
        return 0;

    } else {
        fpgaconvnet::logging::stdout()
            << "Failed to find a solution" << std::endl;
        return 1;
    }
}

#include <fcntl.h>
#include <stdio.h>

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


struct conv_layer_factors_t
{
    uint64_t worker_factor;
    uint64_t conv_factor;
    uint64_t kernel_factor;
};


/* Caches expensive calculation in struct. Eg: Listing the possible cff and 
 * kff values for all layers.
 */
struct layer_valid_values_t
{
    /* Used only in conv */
    std::vector<uint64_t> kernel_factors;
    std::vector<uint64_t> conv_factors;
};

struct ring_connection_t
{
    /* The maxring connection here is between layer_index and layer_index + 1 */
    int layer_index;
    double throughput;
};


struct optimizer_t
{
    std::vector<ring_connection_t> maxring_model;
    std::vector<layer_valid_values_t> layer_valid_values;
};


static std::vector<uint64_t>
compute_significant_factors(uint64_t N)
{
    std::vector<uint64_t> factors = {1};

    for (int i = 2; i <= N ; i++)
    {
        if (fpgaconvnet::math::div_ceil(N, factors.back())
                > fpgaconvnet::math::div_ceil(N, i)) {
            factors.push_back(i);
        }
    }

    return factors;
}

static bool
ring_desc_comparator(
        ring_connection_t const &a, ring_connection_t const &b)
{
    return a.throughput > b.throughput;
}

/* The connection is sorted in decreasing order of throughput. */
static std::vector<ring_connection_t>
build_maxring_bottleneck_model(const fpgaconvnet::protos::Network & network)
{

    std::vector<ring_connection_t> ret;

    for (int i = 0 ; i < network.layer_size() - 1 ; i++) {
        uint64_t image_bytes =
            network.layer(i).output_height()
            * network.layer(i).output_width()
            * network.layer(i).num_outputs()
            * sizeof(fpgaconvnet::fixed_point_t);

        const double throughput =
            double(fpgaconvnet::calculation::MAXRING_BANDWIDTH)
            / double(image_bytes);

        ring_connection_t connection;

        connection.layer_index = i;
        connection.throughput = throughput;

        ret.push_back(connection);
    }

    std::sort(ret.begin(), ret.end(), ring_desc_comparator);

    return ret;
}


static optimizer_t
build_initial_optimizer(const fpgaconvnet::protos::Network & network)
{
    /* I assume the C++ compiler is smart enough not to make two copies of this
     * during return values. But oh well, this is cheap to copy anyways.
     */

    std::vector<layer_valid_values_t> layer_valid_values;

    for (auto it = network.layer().begin()
            ; it != network.layer().end()
            ; it++) {
        layer_valid_values_t valid_values;

        uint64_t kernel_area =
            it->conv().kernel_size() * it->conv().kernel_size();

        valid_values.kernel_factors =
            compute_significant_factors(kernel_area);
        valid_values.conv_factors =
            compute_significant_factors(it->num_outputs());

        layer_valid_values.push_back(valid_values);
    }

    optimizer_t optimizer;
    optimizer.layer_valid_values = layer_valid_values;
    optimizer.maxring_model =
            build_maxring_bottleneck_model(network);
    return optimizer;
}



static std::vector<double>
calculate_relative_worker_factors(const fpgaconvnet::protos::Network & network)
{
    std::vector<double> ret;
    const double base_size =
        network.layer(0).input_height() * network.layer(0).input_width();
    const double base_input_channels = network.layer(0).num_inputs();

    for (auto it = network.layer().begin()
            ; it != network.layer().end()
            ; it++) {
        const double size = it->input_height() * it->input_width();
        const double input_channels = it->num_inputs();
        double ratio = input_channels * size
                       / (base_size * base_input_channels);

        ret.push_back(ratio);
    }

    return ret;
}


static void log_vector(const std::vector<double> & v)
{
    fpgaconvnet::logging::stdout() << "[ ";
    for (int i = 0 ; i < v.size() ; i++) {
        std::cout << v[i] << " ; ";
    }
    std::cout << "]" << std::endl;
}


/* Rounds up x such that ceil % x == 0 */
static uint64_t ceil_divisible(double x, uint64_t ceil) {
    uint64_t ret = std::ceil(x);

    if (ret >= ceil) {
        return ceil;
    }

    for (; ret < ceil ; ret++) {
        if (ceil % ret == 0) {
            return ret;
        }
    }

    return ret;
}


static std::vector<double> compute_ideal_worker_factors(
        fpgaconvnet::protos::Network network,
        uint64_t reference_index,
        double reference_target,
        const std::vector<double> & relative_factors)
{
    std::vector<double> ret;

    for (int i = 0 ; i < relative_factors.size() ; i++) {
        double ideal =
            reference_target
            * relative_factors[i]
            / relative_factors[reference_index];
        ret.push_back(ideal);
    }

    return ret;
}

static fpgaconvnet::protos::LayerParameter
solve_minimal_cff_kff(
        const layer_valid_values_t & layer_valid_values,
        const fpgaconvnet::protos::LayerParameter & layer,
        uint64_t worker_factor,
        uint64_t target_iterations)
{
    /* We do not make different LayerParameter object every iteration - C++
     * scoping rules would create and destroy objects - and protobufs
     * are not exactly cheap!
     */
    bool is_best_initialized = false;
    ::fpgaconvnet::protos::LayerParameter best = layer;
    ::fpgaconvnet::protos::LayerParameter tmp_container = layer;
    ::fpgaconvnet::logging::Indentation indent;

    best.mutable_conv()->set_worker_factor(worker_factor);
    tmp_container.mutable_conv()->set_worker_factor(worker_factor);

    ::fpgaconvnet::logging::stdout(fpgaconvnet::logging::DEBUG)
        << "Target iterations = "
        << target_iterations << '\n';

    for (int i = 0 ; i < layer_valid_values.conv_factors.size() ; i++) {
        for (int j = 0 ; j < layer_valid_values.kernel_factors.size(); j++) {
            auto cff = layer_valid_values.conv_factors[i];
            auto kff = layer_valid_values.kernel_factors[j];

            tmp_container.mutable_conv()->set_conv_folding_factor(cff);
            tmp_container.mutable_conv()->set_kernel_folding_factor(kff);

            uint64_t total_iterations =
                ::fpgaconvnet::calculation::total_iterations(tmp_container);

            if (total_iterations <= target_iterations) {
                int best_total_multipliers =
                    best.conv().conv_folding_factor()
                    * best.conv().kernel_folding_factor();

                if (is_best_initialized == false
                        || cff * kff < best_total_multipliers) {
                    is_best_initialized = true;
                    best.mutable_conv()->set_conv_folding_factor(cff);
                    best.mutable_conv()->set_kernel_folding_factor(kff);
                }
                break;
            }

        }
    }

    ::fpgaconvnet::logging::stdout(fpgaconvnet::logging::DEBUG)
        << "Best (wf, cff, kff) = " << "("
        << best.conv().worker_factor() << ","
        << best.conv().conv_folding_factor() << ", "
        << best.conv().kernel_folding_factor() << ")\n";
    return best;
}



static fpgaconvnet::protos::Network
position_fpgas(
        const optimizer_t & optimizer,
        const fpgaconvnet::protos::Network & network,
        const unsigned num_fpga,
        bool *success)
{
    fpgaconvnet::modelling::PositionFpga position_fpga(network, num_fpga);
    position_fpga.search();
    fpgaconvnet::logging::stdout(fpgaconvnet::logging::DEBUG)
        << "FPGA Positiong accepted "
        << position_fpga.get_num_accepted_solutions()
        << "/" << position_fpga.get_num_considered_solutions()
        << " SOLUTIONS\n";
    std::vector<std::vector<int>> solutions = position_fpga.get_solutions();

    if (solutions.size() == 0) {
        *success = 0;
        return network;
    }

    /* All things equal, we want to use as few FPGAs as possible. */
    unsigned best = 0;
    for (unsigned i = 1 ; i < solutions.size() ; i++) {
        if (solutions[i].back() < solutions[best].back()) {
            best = i;
        }
    }

    *success = 1;
    return fpgaconvnet::insert_fpga_positions(network, solutions[best]);
}


static fpgaconvnet::protos::Network
solve_for_ideal_worker_factors(
        const optimizer_t & optimizer,
        const fpgaconvnet::protos::Network & network,
        const std::vector<double> & ideal_worker_factors,
        const unsigned num_fpga,
        bool *success
)
{
    fpgaconvnet::protos::Network optimized_network = network;

    fpgaconvnet::logging::stdout(fpgaconvnet::logging::DEBUG) << "Network configuration:\n";
    ::fpgaconvnet::logging::Indentation indent;

    for (int i = 0 ; i < optimized_network.layer_size() ; i++)
    {
        ::fpgaconvnet::logging::stdout(fpgaconvnet::logging::DEBUG) << "Layer " << i << '\n';
        ::fpgaconvnet::protos::LayerParameter* layer =
            optimized_network.mutable_layer(i);

        if (layer->has_conv()) {
            uint64_t worker_factor = ceil_divisible(
                        ideal_worker_factors[i], layer->num_inputs());
            uint64_t size_out = layer->output_height() * layer->output_width();
            uint64_t size_in  = layer->input_height()  * layer->input_width();
            uint64_t target_total_iterations = std::ceil(
                    double(layer->num_inputs() * size_in)
                    / double(ideal_worker_factors[i] * size_out));

            *layer = solve_minimal_cff_kff(
                        optimizer.layer_valid_values[i],
                        *layer,
                        worker_factor,
                        target_total_iterations);

        } else if (layer->has_pool()) {
            uint64_t channel_folding_factor = 
                ceil_divisible(
                        ideal_worker_factors[i],
                        layer->num_inputs());
            layer->mutable_pool()->set_channel_folding_factor(
                    channel_folding_factor);

            ::fpgaconvnet::logging::Indentation indent;
            ::fpgaconvnet::logging::stdout(fpgaconvnet::logging::DEBUG)
                << "Channel folding factor = "
                << channel_folding_factor << '\n';

        } else if (layer->has_lrn()) {
            uint64_t channel_folding_factor = 
                ceil_divisible(
                        ideal_worker_factors[i],
                        layer->num_inputs());
            layer->mutable_lrn()->set_channel_folding_factor(
                    channel_folding_factor);

            ::fpgaconvnet::logging::Indentation indent;
            ::fpgaconvnet::logging::stdout(fpgaconvnet::logging::DEBUG)
                << "Channel folding factor = "
                << channel_folding_factor << '\n';
        }
    }

    return position_fpgas(optimizer, optimized_network, num_fpga, success);
}


static uint64_t
choose_reference_layer_index(const fpgaconvnet::protos::Network & network)
{
    uint64_t best = 0;

    for (uint64_t i = 1 ; i < network.layer_size() ; i++) {
        if (network.layer(i).num_inputs() > network.layer(best).num_inputs()) {
            best = i;
        }
    }

    return best;
}


static fpgaconvnet::protos::Network
search_design_space_for_bitstream_with_fixed_num_fpga(
        const fpgaconvnet::protos::Network & network,
        bool *success,
        const int num_fpga)
{
    const optimizer_t optimizer = build_initial_optimizer(network);
    const std::vector<double> relative_worker_factors = 
            calculate_relative_worker_factors(network);
    bool is_best_solution_set = false;
    fpgaconvnet::protos::Network best_solution;

    fpgaconvnet::logging::stdout() << "Relative factors:" << std::endl;
    log_vector(relative_worker_factors);

    fpgaconvnet::logging::Indentation indent;

    // Do a binary search for the ideal bottleneck reference working factor.
    const uint64_t reference_layer_index =
            choose_reference_layer_index(network);
    double lo = 0.0;
    double hi = network.layer(reference_layer_index).num_inputs();

    fpgaconvnet::logging::stdout()
        << "Reference layer index = " << reference_layer_index << std::endl;
    fpgaconvnet::logging::stdout() << std::endl;

    while (hi - lo > 0.0001) {
        double reference_wf = (lo + hi) / 2.0;
        bool success = false;

        fpgaconvnet::logging::stdout(fpgaconvnet::logging::DEBUG)
            << "Reference wf = " << reference_wf;

        std::vector<double> ideal_worker_factors =
            compute_ideal_worker_factors(
                    network,
                    reference_layer_index,
                    reference_wf,
                    relative_worker_factors);
        fpgaconvnet::protos::Network local_solution =
            solve_for_ideal_worker_factors(
                    optimizer, network, ideal_worker_factors, num_fpga, 
                    &success);

        std::vector<fpgaconvnet::resource_model::resource_t> resources =
            ::fpgaconvnet::resource_model::project_single_bitstream(local_solution);
        bool meets_resource_constraints =
            ::fpgaconvnet::resource_model::meets_resource_constraints(resources)
            && local_solution.num_fpga_used() <= local_solution.num_fpga_available()
            && success;

        fpgaconvnet::logging::stdout()
            << reference_wf
            << " Meets constraints: "
            << (meets_resource_constraints ?  "YES" : "NO")
            << "\n";

        if (meets_resource_constraints) {
            assert(local_solution.num_fpga_used() == num_fpga);
        };

        if (meets_resource_constraints) {
            fpgaconvnet::logging::Indentation indent;
            fpgaconvnet::logging::stdout() << "Resource usage:\n";
            for (int i = 0 ; i < resources.size() ; i++) {
                fpgaconvnet::logging::stdout()
                    << "fpga " << i
                    << fpgaconvnet::resource_model::resource_to_string(resources[i])
                    << "\n";
            }
        }

        if (meets_resource_constraints) {
            if (!is_best_solution_set ||
                    fpgaconvnet::calculation::pipeline_throughput(local_solution, -1)
                        > fpgaconvnet::calculation::pipeline_throughput(best_solution, -1)) {
                is_best_solution_set = true;
                best_solution = local_solution;
            }

            lo = reference_wf;

        } else {
            hi = reference_wf;
        }
    }

    for (int i = 0 ; i < optimizer.maxring_model.size() ; i++) {
        ring_connection_t conn = optimizer.maxring_model[i];
        ::fpgaconvnet::logging::stdout() << "MAXRING MAXIMUM THROUGHPUT IF AFTER LAYER "
            << conn.layer_index << ": " << conn.throughput << std::endl;
    }

    *success = is_best_solution_set;
    return best_solution;
}


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
        auto this_solution = search_design_space_for_bitstream_with_fixed_num_fpga(
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
    fpgaconvnet::logging::stdout() << "Loading convnet descriptor:" << std::endl;
    fpgaconvnet::protos::Network network =
	    fpgaconvnet::load_network_proto(argv[1]);
    const char *output_filename = argv[2];

    fpgaconvnet::logging::stdout()
        << "Running Design Space Exploration:"
        << std::endl;
    bool success = false;
    fpgaconvnet::protos::Network solution =
            search_design_space(network, &success);

    if (success) {
        double ops = fpgaconvnet::calculation::ops(solution);
        fpgaconvnet::logging::stdout() << "Found an optimal solution!\n";
        fpgaconvnet::logging::stdout()
            << "Network Operations per image = " << ops << '\n';

        fpgaconvnet::logging::stdout() << "Bitstrema throughput break down\n";

        // true makes the calculation verbose
        double throughput = fpgaconvnet::calculation::real_throughput(solution);
        fpgaconvnet::calculation::explain_throughput(solution);

        fpgaconvnet::logging::stdout()
            << "Projected total GOps = " << ops * throughput * 1e-9 << '\n';

        const auto subnetworks = fpgaconvnet::split_by_bitstreams(solution);
        for (unsigned i = 0; i < subnetworks.size(); i++) {
            fpgaconvnet::logging::stdout()
                << "Resource usage (Bitstream " << i << ") :\n"
                << fpgaconvnet::resource_model::resource_to_string(
                        fpgaconvnet::resource_model::project_single_bitstream(subnetworks[i]));
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

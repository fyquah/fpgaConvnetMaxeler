#include <algorithm>
#include <stdint.h>
#include <vector>

#include <fpgaconvnet/modelling/place_fpga.h>
#include <fpgaconvnet/modelling/search_configuration.h>
#include <fpgaconvnet/modelling/resource_model.h>


namespace fpgaconvnet {
namespace modelling {

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
    /* These vectors are indexed by layer index, not layer id */
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

    ::fpgaconvnet::logging::stdout(fpgaconvnet::logging::DDEBUG)
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

    ::fpgaconvnet::logging::stdout(fpgaconvnet::logging::DDEBUG)
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
        const std::vector<double> & ideal_worker_factors
)
{
    fpgaconvnet::protos::Network optimized_network = network;

    fpgaconvnet::logging::stdout(fpgaconvnet::logging::DDEBUG) << "Network configuration:\n";
    ::fpgaconvnet::logging::Indentation indent;

    for (int i = 0 ; i < optimized_network.layer_size() ; i++)
    {
        ::fpgaconvnet::logging::stdout(fpgaconvnet::logging::DDEBUG) << "Layer " << i << '\n';
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
            ::fpgaconvnet::logging::stdout(fpgaconvnet::logging::DDEBUG)
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
            ::fpgaconvnet::logging::stdout(fpgaconvnet::logging::DDEBUG)
                << "Channel folding factor = "
                << channel_folding_factor << '\n';
        }
    }

    return optimized_network;
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

static bool
throughput_less_than(
    const fpgaconvnet::protos::Network & a,
    const fpgaconvnet::protos::Network & b)
{
  return calculation::effective_throughput(a, -1).throughput <
      calculation::effective_throughput(b, -1).throughput;
}


/* Finds the smallest values for N_{in}^{(ref)} that yields a throughput
 * greater than a given throughput.
 *
 * There is no * point looking at values of N_{in}^{(ref)} that is
 * significantly higher than this value, when the given value is a
 * fundamental hardware limit (maxring or I/O).
 */
static double
find_reference_layer_value_upperbound(
    const optimizer_t & optimizer,
    const fpgaconvnet::protos::Network & network,
    const unsigned reference_layer_index,
    const double bandwidth_limit
    )
{
  const std::vector<double> relative_worker_factors = 
          calculate_relative_worker_factors(network);
  double lo = 0.0;
  double hi = 1e6;
  double best = hi;

  while (hi - lo > 1e-6) {
    const double mid = (lo + hi) / 2.0;
    const double reference_wf = mid;

    const std::vector<double> ideal_worker_factors =
        compute_ideal_worker_factors(
                network,
                reference_layer_index,
                reference_wf,
                relative_worker_factors);
    const fpgaconvnet::protos::Network local_solution =
        solve_for_ideal_worker_factors(
                optimizer, network, ideal_worker_factors);
    const fpgaconvnet::calculation::throughput_t throughput =
        fpgaconvnet::calculation::pipeline_throughput(local_solution, -1);

    if (throughput.throughput > bandwidth_limit) {
      best = mid;
      hi = mid;
    } else {
      lo = mid;
    }
  }

  return best;
}


fpgaconvnet::protos::Network
search_design_space_for_bitstream_with_fixed_num_fpga(
        const fpgaconvnet::protos::Network & network,
        bool *success,
        const int num_fpga)
{
    const optimizer_t optimizer = build_initial_optimizer(network);
    const std::vector<double> relative_worker_factors = 
            calculate_relative_worker_factors(network);
    bool is_best_solution_set = false;

    fpgaconvnet::logging::stdout() << "Relative factors:" << std::endl;
    log_vector(relative_worker_factors);

    fpgaconvnet::logging::Indentation indent;

    // Do a binary search for the ideal bottleneck reference working factor.
    const uint64_t reference_layer_index =
            choose_reference_layer_index(network);
    const double bandwidth_limit =
            fpgaconvnet::calculation::bandwidth_throughput_limit(network, -1);
    double lo = 0.0;
    double hi = find_reference_layer_value_upperbound(
        optimizer, network, reference_layer_index, bandwidth_limit);

    fpgaconvnet::logging::stdout()
        << "Reference layer index = " << reference_layer_index << std::endl;
    fpgaconvnet::logging::stdout()
        << "Bandwith performance upper bound = " << hi << std::endl;
    fpgaconvnet::logging::stdout() << std::endl;

    std::vector<fpgaconvnet::protos::Network> valid_solutions;

    while (hi - lo > 0.0001) {
        double reference_wf = (lo + hi) / 2.0;
        bool success = false;

        fpgaconvnet::logging::stdout(fpgaconvnet::logging::DDEBUG)
            << "Reference wf = " << reference_wf;

        std::vector<double> ideal_worker_factors =
            compute_ideal_worker_factors(
                    network,
                    reference_layer_index,
                    reference_wf,
                    relative_worker_factors);
        fpgaconvnet::protos::Network local_solution =
            solve_for_ideal_worker_factors(
                    optimizer, network, ideal_worker_factors);
        local_solution = position_fpgas(
                    optimizer, local_solution, num_fpga, &success);

        std::vector<fpgaconvnet::resource_model::resource_t> resources =
            ::fpgaconvnet::resource_model::project_single_bitstream(local_solution);
        bool meets_resource_constraints =
            ::fpgaconvnet::resource_model::meets_resource_constraints(resources)
            && local_solution.num_fpga_used() <= local_solution.num_fpga_available()
            && success;

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
            auto t = fpgaconvnet::calculation::pipeline_throughput(
                local_solution, -1);

            if (t.bottleneck_type == fpgaconvnet::calculation::BOTTLENECK_COMPUTE) {
              lo = reference_wf;
              fpgaconvnet::logging::stdout()
                  << reference_wf
                  << " Meets constraints: YES (AND ACCEPTED)\n";
            } else {
              hi = reference_wf;
              fpgaconvnet::logging::stdout()
                  << reference_wf
                  << " Meets constraints: YES (BUT REJECTED)\n";
            }

            valid_solutions.push_back(local_solution);

        } else {
            fpgaconvnet::logging::stdout()
                << reference_wf
                << " Meets constraints: NO\n";
            hi = reference_wf;
        }

    }

    for (int i = 0 ; i < optimizer.maxring_model.size() ; i++) {
        ring_connection_t conn = optimizer.maxring_model[i];
        ::fpgaconvnet::logging::stdout() << "MAXRING MAXIMUM THROUGHPUT IF AFTER LAYER "
            << conn.layer_index << ": " << conn.throughput << std::endl;
    }

    if (valid_solutions.size() == 0) {
      *success = false;
      return network;
    } else {
      *success = true;
      return *std::max_element(
          valid_solutions.begin(),
          valid_solutions.end(),
          throughput_less_than);
    }

}

fpgaconvnet::protos::Network
reconfigure_from_layer_id(
        const fpgaconvnet::protos::Network & reference_network,
        const unsigned layer_id,
        const calculation::throughput_t & target_throughput)
{
    unsigned starting_layer_index = -1;
    for (int i = 0; i < reference_network.layer_size() ; i++) {
        if (reference_network.layer(i).layer_id() == layer_id) {
            starting_layer_index = i;
            break;
        }
    }
    assert(starting_layer_index != -1);

    fpgaconvnet::protos::Network subnetwork = reference_network;
    subnetwork.mutable_layer()->Clear();
    for (int i = starting_layer_index; i < reference_network.layer_size() ; i++) {
        *subnetwork.mutable_layer()->Add() = reference_network.layer(i);
    }
    const std::vector<double> relative_worker_factors =
            calculate_relative_worker_factors(subnetwork);
    const unsigned reference_layer_index = choose_reference_layer_index(
            subnetwork);
    double lo = 0.0;
    double hi = subnetwork.layer(reference_layer_index).num_inputs();
    bool is_best_solution_set = false;
    double best_solution = 0.0;

    auto optimizer = build_initial_optimizer(subnetwork);

    while (hi - lo > 1e-6) {
        double reference_wf = (lo + hi) / 2.0;

        fpgaconvnet::logging::stdout(fpgaconvnet::logging::DDEBUG)
            << "Reference wf = " << reference_wf << std::endl;;

        const std::vector<double> ideal_worker_factors =
            compute_ideal_worker_factors(
                    subnetwork,
                    reference_layer_index,
                    reference_wf,
                    relative_worker_factors);
        const fpgaconvnet::protos::Network this_partial_solution =
            solve_for_ideal_worker_factors(
                    optimizer, subnetwork, ideal_worker_factors);
        fpgaconvnet::protos::Network this_full_solultion = reference_network;
        for (int i = 0; i < this_partial_solution.layer_size() ; i++) {
          *this_full_solultion.mutable_layer()->Mutable(i + starting_layer_index)
              = this_partial_solution.layer(i);
        }

        const auto this_throughput = calculation::pipeline_throughput(
              this_full_solultion, -1);

        fpgaconvnet::logging::stdout(fpgaconvnet::logging::DEBUG)
            << "wf = " << reference_wf
            << " | throughput = "
            << this_throughput
            << std::endl;

        if (this_throughput > target_throughput) {
            hi = reference_wf;
            best_solution = reference_wf;
            is_best_solution_set = true;
        } else {
            lo = reference_wf;
        }
    }

    assert(is_best_solution_set);
    const std::vector<double> ideal_worker_factors =
        compute_ideal_worker_factors(
                subnetwork,
                reference_layer_index,
                best_solution,
                relative_worker_factors);

    auto ret = reference_network;
    subnetwork = solve_for_ideal_worker_factors(
            optimizer, subnetwork, ideal_worker_factors);
    for (int i = starting_layer_index ; i < ret.layer_size() ; i++) {
        *ret.mutable_layer()->Mutable(i) =
                subnetwork.layer(i - starting_layer_index);
    }
    return ret;
}

}  // modelling
}  // fpgaconvnet

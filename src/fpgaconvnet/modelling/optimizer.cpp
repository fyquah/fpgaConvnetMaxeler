#include <cmath>

#include <iostream>
#include <vector>
#include <functional>

#include <fpgaconvnet/common.h>
#include <fpgaconvnet/modelling/resource_model.h>


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

struct optimizer_t
{
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

    optimizer_t optimizer = {.layer_valid_values = layer_valid_values};
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
    for (auto x : v) {
        std::cout << x << " ; ";
    }
    std::cout << "]" << std::endl;
}


/* Rounds up x such that ceil % x == 0 */
static uint64_t ceil_divisible(double x, uint64_t ceil) {
    uint64_t ret;

    if (std::fmod(x, 1.0) > 0.0001) {
        ret = ((uint64_t) x) + 1;
    } else {
        ret  = uint64_t(x);
    }

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

    ::fpgaconvnet::logging::stdout()
        << "Target iterations = "
        << target_iterations << '\n';

    for (auto cff : layer_valid_values.conv_factors) {
        for (auto kff : layer_valid_values.kernel_factors) {
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

    ::fpgaconvnet::logging::stdout() << "Best cff = "
        << best.conv().conv_folding_factor() << '\n';
    ::fpgaconvnet::logging::stdout() << "Best kff = "
        << best.conv().kernel_folding_factor() << '\n';

    return best;
}



static fpgaconvnet::protos::Network
solve_for_ideal_worker_factors(
        const optimizer_t & optimizer,
        const fpgaconvnet::protos::Network & network,
        const std::vector<double> & ideal_worker_factors 
)
{
    fpgaconvnet::protos::Network optimized_network = network;
    ::fpgaconvnet::logging::Indentation indent;

    for (int i = 0 ; i < optimized_network.layer_size() ; i++)
    {
        ::fpgaconvnet::logging::stdout() << "Layer " << i << '\n';
        ::fpgaconvnet::protos::LayerParameter* layer =
            optimized_network.mutable_layer(i);

        if (layer->has_conv()) {
            uint64_t worker_factor = ceil_divisible(
                        ideal_worker_factors[i], layer->num_inputs());
            uint64_t size_out = layer->output_height()  * layer->output_width();
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
            ::fpgaconvnet::logging::stdout()
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
            ::fpgaconvnet::logging::stdout()
                << "Channel folding factor = "
                << channel_folding_factor << '\n';
        }
    }

    return optimized_network;
}


static fpgaconvnet::protos::Network
search_design_space(fpgaconvnet::protos::Network network)
{
    const optimizer_t optimizer = build_initial_optimizer(network);
    const std::vector<double> relative_worker_factors = 
            calculate_relative_worker_factors(network);
    fpgaconvnet::protos::Network best_solution;

    fpgaconvnet::logging::stdout() << "Relative factors:" << std::endl;
    log_vector(relative_worker_factors);

    for (uint64_t reference_layer_index = 0
            ; reference_layer_index < network.layer_size()
            ; reference_layer_index++) {

        auto& layer = network.layer(reference_layer_index);

        fpgaconvnet::logging::stdout()
            << "Bottleneck layer = layer " << reference_layer_index << '\n';

        fpgaconvnet::logging::indent();

        // Do a binary search for the ideal bottleneck reference working factor.
        double reference_wf;
        double lo = 0.0;
        double hi = network.layer(reference_layer_index).num_inputs();
        while (hi - lo > 0.001) {
            double reference_wf = (lo + hi) / 2.0;

            std::vector<double> ideal_worker_factors =
                compute_ideal_worker_factors(
                        network,
                        reference_layer_index,
                        reference_wf,
                        relative_worker_factors);
            fpgaconvnet::logging::stdout ()
                << "Bottleneck worker factor = " << reference_wf << '\n';
            fpgaconvnet::protos::Network local_solution =
                solve_for_ideal_worker_factors(
                        optimizer, network, ideal_worker_factors);
            bool meets_resource_constraints =
                ::fpgaconvnet::resource_model::meets_resource_constraints(
                        local_solution);

            if (meets_resource_constraints) {
                if (fpgaconvnet::calculation::throughput(local_solution)
                        > fpgaconvnet::calculation::throughput(best_solution)) {
                    best_solution = local_solution;
                }

                lo = reference_wf;

            } else {
                hi = reference_wf;
            }
        }
        
        fpgaconvnet::logging::dedent();
    }

    return best_solution;
}


int main (int argc, char **argv)
{
    fpgaconvnet::logging::stdout() << "Loading convnet descriptor:" << std::endl;
    fpgaconvnet::protos::Network network =
	    fpgaconvnet::load_network_proto(argv[1]);

    fpgaconvnet::logging::stdout()
        << "Running Design Space Exploration:"
        << std::endl;
    fpgaconvnet::protos::Network optimized_network =
            search_design_space(network);

    // fpgaconvnet::logging::stdout() << "Optimized descriptor:" << std::endl;
    // fpgaconvnet::logging::stdout() << optimized_network.DebugString() << std::endl;

    return 0;
}

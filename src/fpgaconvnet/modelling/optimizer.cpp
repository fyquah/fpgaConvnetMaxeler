#include <fcntl.h>

#include <cmath>
#include <cstring>

#include <algorithm>
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

    ::fpgaconvnet::logging::stdout()
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

    ::fpgaconvnet::logging::stdout() << "Best cff = "
        << best.conv().conv_folding_factor() << '\n';
    ::fpgaconvnet::logging::stdout() << "Best kff = "
        << best.conv().kernel_folding_factor() << '\n';

    return best;
}


static double
calculate_reference_throughput(const fpgaconvnet::protos::Network & network)
{
    fpgaconvnet::protos::Network reference_network = network;

    for (auto it = reference_network.mutable_layer()->begin()
            ; it != reference_network.mutable_layer()->end()
            ; it++) {
        it->set_fpga_id(0);
    }

    return fpgaconvnet::calculation::throughput(reference_network);
}


/* This algorithms runs in O(N^2) - where N is the number of layers.
 *
 * The idea is that we want to choose the maxring connection such that
 * the overall throughput is not bottlenecked by the connection.
 *
 * This may not be possible at times, in those cases, we permit using
 * the connections with throughput just slower than the optimized
 * throughput, more slower, and so on. For the given list of connection
 * we are allowed to use, we then position the connections greedily
 * (This is provably correct, under some weak assumptions).
 */
static fpgaconvnet::protos::Network
position_fpgas(
        const optimizer_t & optimizer,
        const fpgaconvnet::protos::Network & network)
{
    const double reference_throughput =
            calculate_reference_throughput(network);
    bool ring_connection_allowed[network.layer_size() - 1];
    int allowed_conn_count;

    std::memset(
            ring_connection_allowed,
            false,
            sizeof(bool) * (network.layer_size() - 1));

    for (allowed_conn_count = 0 ;
            allowed_conn_count < optimizer.maxring_model.size() ;
            allowed_conn_count++) {
        const int i = allowed_conn_count;
        const auto & model = optimizer.maxring_model[i];

        if ((model.throughput + 0.01) < reference_throughput) {
            break;
        }

        ring_connection_allowed[model.layer_index] = true;
    }

    std::vector<std::vector<fpgaconvnet::protos::LayerParameter> > solution;
    solution.push_back(std::vector<fpgaconvnet::protos::LayerParameter>());

    for (; allowed_conn_count <= optimizer.maxring_model.size()
            ; allowed_conn_count++) {

        bool is_solution_valid = true;
        int prev_splitable_connection = -1;
        int mapped_layers = 0;

        for (int i = 0 ; i < network.layer_size() ; i++) {
            std::vector<fpgaconvnet::protos::LayerParameter> &
                    current_fpga = solution.back();

            fpgaconvnet::resource_model::stream_t input_stream =
                (solution.size() == 1
                    ? fpgaconvnet::resource_model::STREAM_PCIE
                    : fpgaconvnet::resource_model::STREAM_MAX_RING);

            fpgaconvnet::resource_model::stream_t output_stream =
                (i == network.layer_size() - 1
                     ? fpgaconvnet::resource_model::STREAM_PCIE
                     : fpgaconvnet::resource_model::STREAM_MAX_RING);

            current_fpga.push_back(network.layer(i));

            fpgaconvnet::resource_model::resource_t resource =
                    fpgaconvnet::resource_model::project_single_fpga(
                            input_stream, current_fpga, output_stream);

            if (!fpgaconvnet::resource_model::meets_resource_constraints(
                        resource)) {
                const int old_size =
                        prev_splitable_connection - mapped_layers + 1;

                if (old_size == 0 || prev_splitable_connection < 0) {
                    is_solution_valid = false;
                    break;
                }

                std::vector<fpgaconvnet::protos::LayerParameter> new_fpga(
                        current_fpga.begin() + old_size,
                        current_fpga.end());
                current_fpga.resize(old_size);
                solution.push_back(new_fpga);
                mapped_layers = prev_splitable_connection + 1;
                prev_splitable_connection = -1;
            }

            if (ring_connection_allowed[i]) {
                prev_splitable_connection = i;
            }
        }

        if (is_solution_valid
                || solution.size() > network.num_fpga_available()) {
            fpgaconvnet::protos::Network positioned = network;
            auto ptr = positioned.mutable_layer()->begin();

            for (int i = 0; i < solution.size() ; i++) {
                for (int j = 0 ; j < solution[i].size() ; j++) {
                    ptr->set_fpga_id(i);
                    ptr++;
                }
            }

            positioned.set_num_fpga_used(solution.size());

            return positioned;
        }

        if (allowed_conn_count < optimizer.maxring_model.size()) {
            ring_connection_allowed[
                optimizer.maxring_model[allowed_conn_count].layer_index]
                    = true;
        }
    }

    return network;
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

    if (network.num_fpga_available() == 1) {
        return optimized_network;

    }

    return position_fpgas(optimizer, optimized_network);
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
search_design_space(const fpgaconvnet::protos::Network & network, bool *success)
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

    while (hi - lo > 0.0001) {
        double reference_wf = (lo + hi) / 2.0;

        std::vector<double> ideal_worker_factors =
            compute_ideal_worker_factors(
                    network,
                    reference_layer_index,
                    reference_wf,
                    relative_worker_factors);
        fpgaconvnet::protos::Network local_solution =
            solve_for_ideal_worker_factors(
                    optimizer, network, ideal_worker_factors);

        std::vector<fpgaconvnet::resource_model::resource_t> resources =
            ::fpgaconvnet::resource_model::project(local_solution);
        bool meets_resource_constraints =
            ::fpgaconvnet::resource_model::meets_resource_constraints(resources)
            && local_solution.num_fpga_used() <= local_solution.num_fpga_available();

        fpgaconvnet::logging::stdout() << "Resource usage:\n";

        for (int i = 0 ; i < resources.size() ; i++) {
            fpgaconvnet::logging::stdout()
                << "fpga " << i
                << fpgaconvnet::resource_model::resource_to_string(resources[i])
                << "\n";
        }

        fpgaconvnet::logging::stdout()
            << "Meets constraints: "
            << (meets_resource_constraints ?  "YES" : "NO")
            << "\n";

        if (meets_resource_constraints) {
            if (!is_best_solution_set ||
                    fpgaconvnet::calculation::throughput(local_solution)
                        > fpgaconvnet::calculation::throughput(best_solution)) {
                is_best_solution_set = true;
                best_solution = local_solution;
            }

            lo = reference_wf;

        } else {
            hi = reference_wf;
        }
    }

    *success = is_best_solution_set;
    return best_solution;
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
        double throughput = fpgaconvnet::calculation::throughput(solution);
        double ops = fpgaconvnet::calculation::ops(solution);
        fpgaconvnet::logging::stdout() << "Found an optimal solution!\n";
        fpgaconvnet::logging::stdout() << solution.DebugString();
        fpgaconvnet::logging::stdout()
            << "Network ops = " << ops << '\n';
        fpgaconvnet::logging::stdout()
            << "Projected Throughput = " << throughput << '\n';
        fpgaconvnet::logging::stdout()
            << "Projected total GOps = " << ops * throughput * 1e-9 << '\n';
        fpgaconvnet::logging::stdout()
            << "Resource usage:\n"
            << fpgaconvnet::resource_model::resource_to_string(
                    fpgaconvnet::resource_model::project(solution))
            << std::endl;

        int fd = open(output_filename, O_WRONLY);
        google::protobuf::io::FileOutputStream fstream(fd);
        google::protobuf::TextFormat::Print(solution, &fstream);

        return 0;

    } else {
        fpgaconvnet::logging::stdout()
            << "Failed to find a solution" << std::endl;
        return 1;
    }

    return 0;
}

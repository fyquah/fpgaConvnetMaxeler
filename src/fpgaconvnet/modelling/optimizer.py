from __future__ import absolute_import

import argparse
import bisect
import datetime
import logging
import math
import sys
import random

import numpy as np
from scipy import optimize
from sklearn import linear_model
import yaml
import simanneal

from google.protobuf import text_format
from fpgaconvnet.modelling import resource_model
from fpgaconvnet.protos import parameters_pb2

np.random.seed(
        seed=(datetime.datetime.now() - datetime.datetime(1970, 1, 1)).microseconds)


parser = argparse.ArgumentParser(
        description="Takes a CNN design (in fpgaconvnet.protos.Paramter.Network"
                    " protos form) and produces the ideal factor allocations "
                    " subject to resource constraints.")
parser.add_argument("design", type=str,
                    help="Path to the protos with the design. *_folding_factor"
                         " fields in the proto will be ignored.")
parser.add_argument("--resource-bench", dest="resource_bench", type=str,
                    help="Not used - please remove.")
parser.add_argument("--output", dest="output", type=str,
                    help="Path to the protos with the optimized output.",
                    required=True)


def memory_fetch_size(layer):
    return div_ceil(
            layer.conv.conv_folding_factor
            * layer.conv.worker_factor
            * layer.conv.kernel_folding_factor, 96) * 96


def calc_total_iters(layer):
    worker_iters = div_ceil(layer.num_inputs, layer.conv.worker_factor)
    conv_iters = div_ceil(layer.num_outputs, layer.conv.conv_folding_factor)
    kernel_iters = div_ceil(layer.conv.kernel_size * layer.conv.kernel_size,
            layer.conv.kernel_folding_factor)
    return worker_iters * conv_iters * kernel_iters


def is_off_chip_weights(layer):
    worker_iters = div_ceil(layer.num_inputs, layer.conv.worker_factor)
    conv_iters = div_ceil(layer.num_outputs, layer.conv.conv_folding_factor)
    on_chip_bram_factor = (
            layer.conv.worker_factor * worker_iters
            * layer.conv.conv_folding_factor * conv_iters)
    return layer.conv.bram_factor != on_chip_bram_factor


def populate_weight_address(optimized_network):
    address = 0
    for layer in optimized_network.layer:
        if layer.HasField("conv"):
            layer.conv.look_ahead = 1
            layer.conv.should_fit_on_chip = True


def satisfies_resource_constraints(resources):

    return all(r.bram <= 0.7 * resource_model.MAX_BRAM
                and r.lut <= 0.7 * resource_model.MAX_LUT
                and r.flip_flop <= 0.7 * resource_model.MAX_FF
                and r.dsp <= resource_model.MAX_DSP
               for r in resources)


def compute_valid_values(N):
    ret = [1]
    for i in range(2, N + 1):
        if div_ceil(N, i) < div_ceil(N, ret[-1]):
            ret.append(i)
    return ret


def compute_factors(N):
    ret = [1]
    for i in range(2, N + 1):
        if N % i == 0:
            ret.append(i)
    return ret


def compute_multiples(N, cap):
    ret = [N]
    i = 2
    while True:
        if N * i > cap:
            return ret
        i += 1
        ret.append(N * i)


def random_displacement(x, stepsize, values):
    pos = bisect.bisect_left(values, x)
    flag = 1 if np.random.rand() > 0.5 else -1
    new_pos = min(bisect.bisect_right(
            values, x + flag * np.random.normal(stepsize)), len(values) - 1)
    return values[new_pos]


def sample_array(arr):
    idx = random.randint(0, len(arr) - 1)
    return arr[idx]


def make_fpga_alloc_table(network):
    table = [0] * network.num_fpga_used
    for layer in network.layer:
        table[layer.fpga_id] += 1
    return table


def compute_ops_per_image(layer):
    if layer.HasField("conv"):
        return (layer.output_height * layer.output_width
                * (layer.num_outputs * layer.num_inputs)
                * (layer.conv.kernel_size ** 2)
                * 2)
    elif layer.HasField("pool"):
        return (layer.output_height * layer.output_width
                * layer.num_inputs
                * (layer.pool.dim ** 2))

    elif layer.HasField("lrn"):
        return (layer.output_height * layer.output_width
                * layer.num_inputs
                * layer.lrn.local_size)


class FpgaPositioningProblem(simanneal.Annealer):
    """Minimizes standard deviation of ops per image between fpgas."""

    def __init__(self, network):
        self.state = network

    def move(self):
        while True:
            fpga_to_shift = np.random.randint(self.state.num_fpga_used)
            adjacent = (fpga_to_shift - 1
                        if np.random.randint(2) == 0
                        else fpga_to_shift + 1)
            fpga_alloc_table = make_fpga_alloc_table(self.state)

            if (adjacent < 0
                    or adjacent >= self.state.num_fpga_used
                    or fpga_alloc_table[adjacent] == 1):
                continue
            fpga_alloc_table[adjacent] -= 1
            fpga_alloc_table[fpga_to_shift] += 1

            new_state = parameters_pb2.Network()
            new_state.CopyFrom(self.state)

            fpga_index = 0
            for layer in new_state.layer:
                layer.fpga_id = fpga_index
                fpga_alloc_table[fpga_index] -= 1
                if fpga_alloc_table[fpga_index] == 0:
                    fpga_index += 1

            self.state = new_state
            break

    def energy(self):
        ops = [0] * self.state.num_fpga_used
        for layer in self.state.layer:
            ops[layer.fpga_id] += compute_ops_per_image(layer)
        return np.std(ops)


class FoldingFactorOptimizationProblem(simanneal.Annealer):
    """Minimizes GOps with ff values for compute time given a FPGA allocation."""
    def __init__(self, network):
        initial_state = parameters_pb2.Network()
        initial_state.CopyFrom(network)
        super(FoldingFactorOptimizationProblem, self).__init__(initial_state)
        self.valid_values = []

        for layer in network.layer:
            if layer.HasField("conv"):
                self.valid_values.append(("conv", {
                        "worker_factor": compute_valid_values(layer.num_inputs),
                        "conv_folding_factor": compute_valid_values(layer.num_outputs),
                        "kernel_folding_factor": compute_valid_values(
                            layer.conv.kernel_size * layer.conv.kernel_size),
                        # "look_ahead": compute_factors(layer.output_height * layer.output_width),

                        # Special cases... argh.
                        "bram_factor": None,
                }))

            elif layer.HasField("pool"):
                self.valid_values.append(("pool", {
                        "channel_folding_factor": compute_valid_values(layer.num_inputs),
                }))

            elif layer.HasField("lrn"):
                self.valid_values.append(("lrn", {
                        "channel_folding_factor": compute_valid_values(layer.num_inputs),
                }))

            else:
                raise RuntimeError("Unknown layer type.")


    def move(self):
        # TODO(fyq14): Decide if it is time to shift left or shift right.
        self.state = self._pertubate(self.state)

    def _pertubate(self, state):
        while True:
            layer_id = random.randint(0, len(state.layer) - 1)
            layer_type = self.valid_values[layer_id][0]
            field_name = sample_array(self.valid_values[layer_id][1].keys())
            new_network = parameters_pb2.Network()
            new_network.CopyFrom(state)

            if field_name == "bram_factor":
                if new_network.layer[layer_id].conv.should_fit_on_chip:
                    continue
                layer = new_network.layer[layer_id]
                scheduler_iters = div_ceil(
                        layer.num_inputs, layer.conv.worker_factor)
                conv_iters = div_ceil(
                        layer.num_outputs, layer.conv.conv_folding_factor)
                new_value = (layer.conv.conv_folding_factor
                        * layer.conv.worker_factor
                        * sample_array(compute_factors(conv_iters)
                                       + compute_multiples(conv_iters, scheduler_iters * conv_iters)))
            elif (new_network.layer[layer_id].conv.should_fit_on_chip
                        and field_name == "look_ahead"):
                continue
            else:
                new_value = sample_array(self.valid_values[layer_id][1][field_name])

            setattr(getattr(new_network.layer[layer_id], layer_type), field_name, new_value)

            # Special case: synchronize changes to bram_factor
            if field_name == "worker_factor" or field_name == "conv_folding_factor":
                layer = new_network.layer[layer_id]

                layer.conv.bram_factor = (
                        layer.conv.conv_folding_factor
                        * layer.conv.worker_factor
                        * div_ceil(layer.num_outputs, layer.conv.conv_folding_factor)
                        * div_ceil(layer.num_inputs, layer.conv.worker_factor))

                # Select the largest layer.layer_id that satisfies the new
                # constraints imposed.
                # while True:
                #     total_convolvers = (layer.conv.conv_folding_factor
                #                         * layer.conv.worker_factor)

                #     if layer.conv.bram_factor % total_convolvers == 0:
                #         size = layer.conv.bram_factor / total
                #         conv_iters = div_ceil(layer.num_outputs, layer.conv.conv_folding_factor)
                #         if (size < conv_iters and conv_iters % size == 0
                #                 or size >= conv_iters and size % conv_iters):
                #             break
                #         layer.conv.bram_factor -= 1

            resources = resource_model.project(new_network)

            if satisfies_resource_constraints(resources):
                return new_network

        return None

    def energy(self):
        return -estimate_gops(self.state)

    def copy_state(self, state):
        new_network = parameters_pb2.Network()
        new_network.CopyFrom(state)
        return new_network


def curry(func, *args):
    def fn(a):
        arr = args[:] + (a,)
        return func(*arr)
    return fn


def nearest_value(x, vs):
    return vs[np.argmin(abs(np.array(vs) - x))]


def make_presence_constraint(idx, values):
    def fn(x):
        return max(0 if abs(x[idx] - v) < 0.01 else -1 for v in values)

    return {"type": "eq", "fun": fn}


def randomize_compute_factors(network, valid_values):
    """Randomizes the compute-bound parameters of the net.
    
    Specifically randomizes worker_factor, conv_folding_factor,
    kernel_folding_factor, channel_folding_factor.
    """
    for layer, values in zip(network.layer, valid_values):
        if layer.HasField("conv"):
            layer.conv.worker_factor = sample_array(values[0])
            layer.conv.conv_folding_factor = sample_array(values[1])
            layer.conv.kernel_folding_factor = sample_array(values[2])

        elif layer.HasField("pool"):
            layer.pool.channel_folding_factor = sample_array(values[0])

        elif layer.HasField("lrn"):
            layer.lrn.channel_folding_factor = sample_array(values[0])


def search_initial_state(network, num_fpgas):

    for i, layer in enumerate(network.layer):
        layer.fpga_id = min(i, num_fpgas - 1)
    network.num_fpga_used = num_fpgas

    if num_fpgas != 1:
        problem = FpgaPositioningProblem(network)
        network, e = problem.anneal()

    # Randomly search the folding factor search until we get a feasible
    # configuration, then we can annael from there.
    valid_values = []
    for layer in network.layer:
        if layer.HasField("conv"):
            valid_values.append((
                    compute_valid_values(layer.num_inputs),
                    compute_valid_values(layer.num_outputs),
                    compute_valid_values(layer.conv.kernel_size
                                         * layer.conv.kernel_size)))

        elif layer.HasField("pool") or layer.HasField("lrn"):
            valid_values.append((
                    compute_valid_values(layer.num_outputs),))

    for i in range(100000):
        randomize_compute_factors(network, valid_values)
        resource_projection = resource_model.project(network)
        if satisfies_resource_constraints(resource_projection):
            print "Folding factor randomization succeeded after %d stages" % i
            return network


def optimize_with_fixed_fpga_count(network, num_fpgas):

    initial_state = search_initial_state(network, num_fpgas)
    if initial_state is None:
        print "Failed to find an initial state for", num_fpgas
        return None

    print "====> Found a suitable initial state for ", num_fpgas
    print initial_state
    print ""
    problem = FoldingFactorOptimizationProblem(initial_state)
    state, e = problem.anneal()
    resource = resource_model.project(state)

    print "====> Optimized for", num_fpgas, "FPGAs."
    for i, r in enumerate(resource):
        total_lut_used = r.lut
        total_ff_used = r.flip_flop
        total_dsp_used = r.dsp
        total_m20k_used = r.bram

        print "fpga %d Estimated total LUT used: %d (%.3f) " % \
                (i, total_lut_used,
                 float(total_lut_used) / resource_model.MAX_LUT)
        print "fpga %d Estimated total flip flops used: %d (%.3f)" % \
                (i, total_ff_used,
                 float(total_ff_used) / resource_model.MAX_FF)
        print "fpga %d Estimated total DSP used: %d (%.3f)" % \
                (i, total_dsp_used,
                 float(total_dsp_used) / resource_model.MAX_DSP)
        print "fpga %d Estimaed M20k used: %d (%.3f)" % \
                (i, total_m20k_used,
                 float(total_m20k_used) / resource_model.MAX_BRAM)
        print ""
    print "Estimated GOps:", estimate_gops(state), "\n"
    print ""

    return state


def run_optimizer(network):
    minimized_states = []
    original_network = network

    for i in range(network.num_fpga_available, network.num_fpga_available + 1):
        network = parameters_pb2.Network()
        network.CopyFrom(original_network)
        network.num_fpga_used = i
        minimized_states.append(optimize_with_fixed_fpga_count(network , i))

    # Choose the best state, that uses the lowest number of FPGAs.
    best = None
    best_index = None
    for i, network in enumerate(minimized_states):
        if (network is not None
                and (best is None or estimate_gops(network) > best + 1.0)):
            best = estimate_gops(network)
            best_index = i
    return minimized_states[best_index]


def div_ceil(a, b):
    a = int(a)
    b = int(b)
    if a % b == 0:
        return a / b
    else:
        return a / b + 1


def plot_figs(fig_num, elev, azim, X, Y, clf, labels=["X_1", "X_2", "Y"]):
    fig = plt.figure(fig_num, figsize=(4, 3))
    plt.clf()
    ax = mplot3d.Axes3D(fig, elev=elev, azim=azim)

    ax.scatter(X[:, 0], X[:, 1], Y, c='k', marker='+')
    X_0_min = np.min(X[:, 0]) - 0.1
    X_0_max = np.max(X[:, 0]) + 0.1
    X_1_min = np.min(X[:, 1]) - 0.1
    X_1_max = np.max(X[:, 1]) + 0.1
    ax.plot_surface(np.array([[X_0_min, X_0_min],
                              [X_0_max, X_0_max]]),
                    np.array([[X_1_min, X_1_max],
                              [X_1_min, X_1_max]]),
                    clf.predict(
                        np.array([[X_0_min, X_1_min],
                                  [X_0_min, X_1_max],
                                  [X_0_max, X_1_min],
                                  [X_0_max, X_1_max]])
                                ).reshape((2, 2)),
                    alpha=.5)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])


def calc_total_iterations(layer, factors):
    wf, cff, kff = factors
    return (float(div_ceil(layer.conv.kernel_size * layer.conv.kernel_size, kff))
            * float(div_ceil((layer.num_outputs), cff))
            * float(div_ceil((layer.num_inputs), wf)))


def calc_total_ops(layer):
    return float(layer.num_inputs) * layer.num_outputs \
            * layer.conv.kernel_size * layer.conv.kernel_size * 2.


def estimate_gops(network):
    clock_rate = network.frequency * 1e6

    # At our clock rate, it is safe to assume that LMem can produce
    # pixel a cycle. The maximum bandwidth of PCI is around 0.5GBps (
    # according to a post in mdx) - which translates to:
    #   - 0.25 GBps in a direction
    #   - 2.0 Gbps in a direction
    #   - 0.1111 giga words/s in a direction (a number is 18 bits)
    #   - maximum of (1.111 * 1e8 / num_inputs) inputs per second in the
    #     first layer.
    #   - This is the bottle neck imposed by PCIe transfers transfers.

    # Similarly:
    # If all the memory are connected to the same

    # In the case of reading / writing from the PC, that would be 0.5 Gbps
    # and similar maths follows.
    prev_cycles = float(clock_rate
                        / (0.5 * (8. / 18.) * 1e9 * network.layer[0].num_inputs))
    minimum_cycles = prev_cycles

    # *_cycles denotes the number of cycles (possibly float) a particular
    # kernel / unit takes to generate a new output. This number is possibly
    # a float, which means it is generating output at a higher rate then
    # consuming input (as in scheduler and conv_unit)
    ops_per_cycle = 0

    kernel_input_cycles = []

    for layer in network.layer:
        if layer.HasField("conv"):

            wf = layer.conv.worker_factor
            cff = layer.conv.conv_folding_factor
            kff = layer.conv.kernel_folding_factor

            # ConvolutionScheduler produces a set of new output at
            # every non-border cycles.
            output_height = (
                    (layer.input_height + 2 * layer.conv.pad - layer.conv.kernel_size) / layer.conv.stride + 1)
            output_width = (
                    (layer.input_width + 2 * layer.conv.pad - layer.conv.kernel_size) / layer.conv.stride + 1)

            scheduler_cycles = (float(prev_cycles)
                    * (layer.input_height * layer.input_width)
                    / output_height
                    / output_width
                    / div_ceil(layer.num_inputs, wf))

            # ConvolutionUnit produces a new output as fast as it receives
            conv_unit_cycles = (scheduler_cycles
                    / float(div_ceil(layer.num_outputs, cff))
                    / float(div_ceil(
                        layer.conv.kernel_size * layer.conv.kernel_size,
                        kff)))

            # - Accumulator produces a new output every
            #   calc_total_iterations(..) of inputs.
            # - It accepts an input every conv_unit_cycles.
            acc_cycles = (calc_total_iterations(layer, (wf, cff, kff))
                    * conv_unit_cycles)

            kernel_input_cycles.append(
                (prev_cycles, scheduler_cycles, conv_unit_cycles))

            prev_cycles = acc_cycles
            ops_per_cycle += calc_total_ops(layer) / acc_cycles

            minimum_cycles = min([
                    acc_cycles,
                    conv_unit_cycles,
                    scheduler_cycles,
                    minimum_cycles])

        elif layer.HasField("pool"):
            kernel_input_cycles.append((prev_cycles,))
            pool_iters = div_ceil(layer.num_outputs, layer.pool.channel_folding_factor)
            prev_cycles = (
                prev_cycles
                * layer.pool.stride
                * layer.pool.stride
                * pool_iters)

        elif layer.HasField("lrn"):
            kernel_input_cycles.append((prev_cycles,))
            lrn_iters = div_ceil(layer.num_outputs, layer.lrn.channel_folding_factor)
            prev_cycles = (prev_cycles * lrn_iters)

        else:
            raise RuntimeError("Unknown layer %d." % (layer.layer_id))

    # Resolving the actual number of cycles is a linear programming cycles
    # if we account for the number of cycles which is relatively slow
    # to compute and might disrupt simulated annaeling optimization.
    #
    # The heriustic that we will use to compute the post-memory frequency
    # is to take the sum of memory accesses and scale down by the amount
    # it has surpass the memory bandwidth.
    memory_access_per_cycle = 0.0
    for i, layer in enumerate(network.layer):
        if layer.HasField("conv") and not resource_model.is_cpu_init(layer):
            width = div_ceil(
                    layer.conv.worker_factor
                    * layer.conv.conv_folding_factor
                    * layer.conv.kernel_folding_factor, 96) * 96 * 32.
            memory_access_per_cycle += (
                width * clock_rate * minimum_cycles
                / (kernel_input_cycles[i][1] * layer.conv.look_ahead))

    memory_access_per_cycle = clock_rate * minimum_cycles
    memory_access_scale_factor = min(1, 38.4 * 1e9 / memory_access_per_cycle)

    # Multiply by minimum_cycles at the end so that none of the kernels
    # in the pipeline is running faster than the given clock rate.
    return 1e-9 * ops_per_cycle * clock_rate * minimum_cycles \
            * memory_access_scale_factor


def main():
    with open(FLAGS.design, "r") as f:
        network = text_format.Parse(f.read(), parameters_pb2.Network())

    for i, layer in enumerate(network.layer):
        if i != 0:
            layer.input_height = network.layer[i - 1].output_height
            layer.input_width = network.layer[i - 1].output_width
            layer.num_inputs = network.layer[i - 1].num_outputs

        layer.layer_id = i

        if layer.HasField("conv"):
            layer.output_height = (
                    (layer.input_height + 2 * layer.conv.pad - layer.conv.kernel_size)
                     / layer.conv.stride + 1)
            layer.output_width = (
                    (layer.input_width + 2 * layer.conv.pad - layer.conv.kernel_size)
                     / layer.conv.stride + 1)

            # Default parameters
            # Roughly half the worker factor, prevents heavy multiplexing at the
            # conv scheduler.
            layer.conv.worker_factor = int(math.ceil(layer.num_inputs / 3.))
            layer.conv.kernel_folding_factor = 1
            layer.conv.conv_folding_factor = 1
            layer.conv.look_ahead = 1
            layer.conv.bram_factor = layer.num_inputs * layer.num_outputs
            layer_bram_required = (
                    layer.num_inputs * layer.num_outputs * (layer.conv.kernel_size ** 2)
                        * resource_model.NUM_BITS)
            layer.conv.should_fit_on_chip = True

        elif layer.HasField("pool"):
            layer.num_outputs = layer.num_inputs
            stride = layer.pool.stride or layer.pool.dim
            layer.pool.stride = stride
            layer.output_height = div_ceil(layer.input_height, stride)
            layer.output_width = div_ceil(layer.input_width, stride)

            # Default parameters
            layer.pool.channel_folding_factor = 1

        elif layer.HasField("lrn"):
            layer.num_outputs = layer.num_inputs
            layer.output_height = layer.input_height
            layer.output_width =  layer.input_width

            # Default parameters
            layer.lrn.channel_folding_factor = 1

        else:
            raise RuntimeError("Unknown layer!")

    network.layer[0].is_first_layer = True
    network.layer[-1].is_last_layer = True
    print network
    logging.getLogger().setLevel(logging.DEBUG)
    optimized_network = run_optimizer(network)
    populate_weight_address(optimized_network)

    resource_model.project(optimized_network)

    with open(FLAGS.output, "w") as f:
        f.write(text_format.MessageToString(optimized_network))


if __name__ == "__main__":
    FLAGS = parser.parse_args()
    main()

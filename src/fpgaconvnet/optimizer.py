from __future__ import absolute_import

import argparse
import bisect
import logging
import sys
import random

import numpy as np
from scipy import optimize
from sklearn import linear_model
import yaml
import simanneal

from google.protobuf import text_format
from fpgaconvnet import resource_model
from fpgaconvnet.protos import parameters_pb2


parser = argparse.ArgumentParser(
        description="Takes a CNN design (in fpgaconvnet.protos.Paramter.Network"
                    " protos form) and produces the ideal factor allocations "
                    " subject to resource constraints.")
parser.add_argument("design", type=str,
                    help="Path to the protos with the design. *_folding_factor"
                         " fields in the proto will be ignored.")
parser.add_argument("--resource-bench", dest="resource_bench", type=str,
                    help=("Path to yaml file with the results from running "
                          " resource benches."),
                    required=True)
parser.add_argument("--output", dest="output", type=str,
                    help="Path to the protos with the optimized output.",
                    required=True)

# The amount of M20k used per 18-bit fixed point word.
BASE_M20K_FACTOR = 18.0 / 20480.0


def make_model_from_lm(lm):
    def fn(data):
        return lm.predict(np.reshape(data, (1, -1)))[0]
    return fn


def make_negate_fn(fn):
    def new_fn(*args, **kwargs):
        return -fn(*args, **kwargs)
    return new_fn


def compute_valid_values(N):
    ret = [1]
    for i in range(2, N + 1):
        if div_ceil(N, i) < div_ceil(N, ret[-1]):
            ret.append(i)
    return ret


def random_displacement(x, stepsize, values):
    pos = bisect.bisect_left(values, x)
    flag = 1 if np.random.rand() > 0.5 else -1
    new_pos = min(bisect.bisect_right(
            values, x + flag * np.random.normal(stepsize)), len(values) - 1)
    return values[new_pos]


def sample_array(arr):
    idx = random.randint(0, len(arr) - 1)
    return arr[idx]


class OptimizationProblem(simanneal.Annealer):

    def __init__(self, network):
        initial_state = parameters_pb2.Network()
        initial_state.CopyFrom(network)
        super(OptimizationProblem, self).__init__(initial_state)
        self.valid_values = []

        for layer in network.layer:
            if layer.HasField("conv"):
                self.valid_values.append(("conv", {
                        "worker_factor": compute_valid_values(layer.num_inputs),
                        "conv_folding_factor": compute_valid_values(layer.num_outputs),
                        "kernel_folding_factor": compute_valid_values(
                            layer.conv.kernel_size * layer.conv.kernel_size)
                }))

            elif layer.HasField("pool"):
                self.valid_values.append(("pool", {
                        "channel_folding_factor": compute_valid_values(layer.num_inputs)
                }))

            elif layer.HasField("lrn"):
                self.valid_values.append(("lrn", {
                        "channel_folding_factor": compute_valid_values(layer.num_inputs)
                }))

            else:
                raise RuntimeError("Unknown layer type.")


    def move(self):
        while True:
            layer_id = random.randint(0, len(self.state.layer) - 1)
            layer_type = self.valid_values[layer_id][0]
            field_name = sample_array(self.valid_values[layer_id][1].keys())
            new_value = sample_array(self.valid_values[layer_id][1][field_name])
            original = getattr(getattr(self.state.layer[layer_id], layer_type), field_name)
            setattr(getattr(self.state.layer[layer_id], layer_type), field_name, new_value)

            resources = resource_model.project(self.state)

            if (resources.bram <= resource_model.MAX_BRAM
                    and resources.lut <= resource_model.MAX_LUT
                    and resources.flip_flop <= resource_model.MAX_FF
                    and resources.dsp <= resource_model.MAX_DSP):
                break
            else:
                setattr(getattr(self.state.layer[layer_id], layer_type), field_name, original)

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


def run_optimizer(network):
    num_conv_layers = 0
    for layer in network.layer:
        if layer.HasField("conv"):
            num_conv_layers += 1

    minimized_states = []
    for i in range(1):
        problem = OptimizationProblem(network)
        state, e = problem.anneal()
        resource = resource_model.project(state)
        total_lut_used = resource.lut
        total_ff_used = resource.flip_flop
        total_dsp_used = resource.dsp
        total_m20k_used = resource.bram

        print "=> Attempt", i
        print "Estimated total LUT used: %d (%.3f) " % \
                (total_lut_used,
                 float(total_lut_used) / resource_model.MAX_LUT)
        print "Estimated total flip flops used: %d (%.3f)" % \
                (total_ff_used,
                 float(total_ff_used) / resource_model.MAX_FF)
        print "Estimated total DSP used: %d (%.3f)" % \
                (total_dsp_used,
                 float(total_dsp_used) / resource_model.MAX_DSP)
        print "Estimaed M20k used: %d (%.3f)" % \
                (total_m20k_used,
                 float(total_m20k_used) / resource_model.MAX_BRAM)
        print "Estimated GOps:", estimate_gops(state), "\n"
        minimized_states.append(state)

    # TODO(fyq14): Choose the best state, rather than returning the first one..
    return minimized_states[0]


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
            prev_cycles = (
                prev_cycles
                * layer.pool.stride
                * layer.pool.stride
                * layer.pool.channel_folding_factor)

        elif layer.HasField("lrn"):
            kernel_input_cycles.append((prev_cycles,)) 
            prev_cycles = (
                prev_cycles
                * layer.lrn.stride
                * layer.lrn.stride
                * layer.lrn.channel_folding_factor)

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

            if not layer.conv.HasField("bram_factor"):
                wf = layer.conv.worker_factor
                cff = layer.conv.conv_folding_factor
                layer.conv.bram_factor = (
                    div_ceil(layer.num_inputs, wf) * wf
                    * div_ceil(layer.num_outputs, cff) * cff)

        elif layer.HasField("pool"):
            layer.num_outputs = layer.num_inputs
            stride = layer.pool.stride or layer.pool.dim
            layer.pool.stride = stride
            layer.output_height = div_ceil(layer.input_height, stride)
            layer.output_width =  div_ceil(layer.input_width, stride)

        else:
            raise RuntimeError("Unknown layer!")

    network.layer[0].is_first_layer = True
    network.layer[-1].is_last_layer = True
    print network
    # logging.getLogger().setLevel(logging.DEBUG)
    # print resource_model.project(network)
    optimized_network = run_optimizer(network)

    with open(FLAGS.output, "w") as f:
        f.write(text_format.MessageToString(optimized_network))


if __name__ == "__main__":
    FLAGS = parser.parse_args()
    main()

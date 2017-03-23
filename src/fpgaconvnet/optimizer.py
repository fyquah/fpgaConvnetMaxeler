from __future__ import absolute_import

import argparse
import bisect
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


def scale_x(x_new):
    return [(x_new[i], x_new[i+1], x_new[i+2])
            for i in range(0, len(x_new), 3)]

def sample_array(arr):
    idx = random.randint(0, len(arr) - 1)
    return arr[idx]


class OptimizationProblem(simanneal.Annealer):

    def __init__(self, network, constraints):
        initial_state = parameters_pb2.Network()
        initial_state.CopyFrom(network)
        super(OptimizationProblem, self).__init__(initial_state)
        self.gops_fn = make_gops_fn(network)
        self.valid_values = []
        self.constraints = constraints

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
            new_value = sample_array(self.valid_values[layer_id][field_name])
            original = getattr(getattr(self.state.layer[layer_id], layer_type), field_name)
            setattr(getattr(self.state.layer[layer_id], layer_type), new_value)

            resources = resource_model.project(self.state)

            if (resources.bram <= MAX_DSP
                    and resources.lut <= MAX_LUT
                    and resources.flip_flop <= MAX_FF
                    and resources.dsp <= MAX_DSP):
                break
            else:
                setattr(getattr(self.state.layer[layer_id], layer_type), original)

    def energy(self):
        return -self.gops_fn(self.state)


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


def run_optimizer(network, constraints):
    num_conv_layers = 0
    for layer in network.layer:
        if layer.HasField("conv"):
            num_conv_layers += 1

    minimized_states = []
    for i in range(1):
        problem = OptimizationProblem(network, constraints)
        problem.copy_strategy = "slice"  
        state, e = problem.anneal()
        total_logic_used = constraints["logic_utilization"] \
                           - problem.logic_utilization_constraint(state)
        total_multipliers = 0.7 * constraints["multiplier"] \
                            - problem.multiplier_constraint(state)
        total_m20k = constraints["block_memory"] \
                            - problem.bram_constraint(state)

        print "=> Attempt", i
        print "Estimated total logic utilization: %d (%.3f)" % \
                (total_logic_used,
                 float(total_logic_used) / constraints["logic_utilization"])
        print "Estimated total multipliers: %d (%.3f)" % \
                (total_multipliers,
                 float(total_multipliers) / constraints["multiplier"])
        print "Optimal params:", scale_x(state)
        print "Estimaed M20k used: %d (%.3f)" % \
                (total_m20k,
                 float(total_m20k) / constraints["block_memory"])
        print "Estimated GOps:", problem.gops_fn(state), "\n"
        minimized_states.append(state)

    # TODO(fyq14): Choose the best state, rather than returning the first one..
    return scale_x(minimized_states[0])


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


def make_gops_fn(network):
    """Returns a function that computes the estimated GOps of a given configuration.

    Function assumes a saturated system.
    """

    valid_values = []
    for layer in network.layer:
        if layer.HasField("conv"):
            valid_values.append(compute_valid_values(layer.num_inputs))
            valid_values.append(compute_valid_values(layer.num_outputs))
            valid_values.append(compute_valid_values(
                    layer.conv.kernel_size * layer.conv.kernel_size))
    valid_values = tuple(valid_values)

    def fn(factors):
        """
        Arguments:
            args: List of (wf, cff, kff) in integer form.
        """
        factors = factors[:]
        clock_rate = network.frequency * 1e6
        acc_pipeline_length = 1

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

        factors_ptr = 0
        kernel_input_cycles = []

        for layer in network.layer:
            if layer.HasField("conv"):
                (wf, cff, kff) = factors[factors_ptr:factors_ptr+3]
                if any(x <= 0 for x in (wf, cff, kff)):
                    return 0

                wf = nearest_value(wf, valid_values[factors_ptr])
                cff = nearest_value(cff, valid_values[factors_ptr + 1])
                kff = nearest_value(kff, valid_values[factors_ptr + 2])
                factors_ptr += 3

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
                #   calc_total_iterations(..) * acc_pipeline_length of inputs.
                # - It accepts an input every conv_unit_cycles.
                acc_cycles = (calc_total_iterations(layer, (wf, cff, kff))
                        * acc_pipeline_length
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
                    / (kernel_input_cycles[i][1] * layer.conv.look_ahaed)

        memory_access_per_cycle = clock_rate * minimum_cycles
        memory_access_scale_factor = min(1, 38.4 * 1e9 / memory_access_per_cycle)

        # Multiply by minimum_cycles at the end so that none of the kernels
        # in the pipeline is running faster than the given clock rate.
        return 1e-9 * ops_per_cycle * clock_rate * minimum_cycles \
                * memory_access_scale_factor

    return fn


def main():
    with open(FLAGS.resource_bench, "r") as f:
        resource_bench = yaml.load(f.read())

    max_logic_utilization = resource_bench["resources"]["logic_utilization"]
    max_multiplier = resource_bench["resources"]["multiplier"]
    max_block_memory = resource_bench["resources"]["block_memory"]
    # Each M20k block contains 20k bits

    optimized_params = run_optimizer(
            network=network,
            constraints=resource_bench["resources"])

    # Write the results to a new protobuf and flush it to FLAGS.output
    optimized_network = parameters_pb2.Network()
    optimized_network.CopyFrom(network)
    for layer in optimized_network.layer:
        if layer.HasField("conv"):
            (wf, cff, kff) = optimized_params.pop(0)
            layer.conv.worker_factor = wf
            layer.conv.conv_folding_factor = cff
            layer.conv.kernel_folding_factor = kff

            if not layer.HasField("bram_factor"):
                layer.conv.bram_factor = (
                    div_ceil(layer.conv.num_inputs, wf) * wf
                    * div_ceil(layer.conv.num_outputs, cff) * cff)

        elif layer.HasField("pool"):
            if not layer.pool.HasField("stride"):
                layer.pool.stride = layer.pool.dim

    with open(FLAGS.output, "w") as f:
        f.write(text_format.MessageToString(optimized_network))


if __name__ == "__main__":
    FLAGS = parser.parse_args()
    main()

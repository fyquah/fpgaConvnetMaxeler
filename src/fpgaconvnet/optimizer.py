from __future__ import absolute_import

import argparse
import bisect
import sys
import random

# import GPy
# from mpl_toolkits import mplot3d
# from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize
from sklearn import linear_model
import yaml
import simanneal

from google.protobuf import text_format
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


class OptimizationProblem(simanneal.Annealer):

    def __init__(self, network, models, constraints, x0):
        super(OptimizationProblem, self).__init__(x0)
        self.gops_fn = make_gops_fn(network)
        self.valid_values = []
        self.constraints = constraints
        self.models = models
        self.base_bram_usage = 0.0

        for layer in network.layer:
            if layer.HasField("conv"):
                self.valid_values.append(compute_valid_values(layer.num_inputs))
                self.valid_values.append(compute_valid_values(layer.num_outputs))
                self.valid_values.append(compute_valid_values(
                    layer.conv.kernel_size * layer.conv.kernel_size))
                total_weights = (layer.num_inputs * layer.num_outputs
                                 * layer.conv.kernel_size * layer.conv.kernel_size
                                 + layer.num_outputs)
                total_fifo = layer.conv.kernel_size * layer.input_width * layer.num_inputs
                self.base_bram_usage += (total_weights + total_fifo) * BASE_M20K_FACTOR

            elif layer.HasField("pool"):
                total_fifo = layer.pool.dim * layer.input_width * layer.num_inputs
                self.base_bram_usage += total_fifo * BASE_M20K_FACTOR

    def logic_utilization_constraint(self, x_new):
        logic_utilization = sum(
                self.models["conv"]["residual_logic"](x) for x in scale_x(x_new)) \
                + self.models["conv"]["base_logic"](x_new)
        return self.constraints["logic_utilization"] - logic_utilization

    def multiplier_constraint(self, x_new):
        multiplier = sum(
            self.models["conv"]["multiplier"](x) for x in scale_x(x_new))
        return self.constraints["multiplier"] - multiplier

    def bram_constraint(self, x_new):
        block_memory = sum(self.models["conv"]["residual_block_memory"](x)
                           for x in scale_x(x_new)) \
                       + self.base_bram_usage
        return self.constraints["block_memory"] - block_memory

    def move(self):
        while True:
            a = random.randint(0, len(self.state) - 1)
            value_index = random.randint(0, len(self.valid_values[a]) - 1)
            original = self.state[a]
            self.state[a] = self.valid_values[a][value_index]

            if (self.multiplier_constraint(self.state) >= 0
                    and self.logic_utilization_constraint(self.state) >= 0
                    and self.bram_constraint(self.state) >= 0):
                break
            else:
                self.state[a] = original

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


def run_optimizer(network, models, constraints):
    num_conv_layers = 0
    for layer in network.layer:
        if layer.HasField("conv"):
            num_conv_layers += 1

    for i in range(5):
        problem = OptimizationProblem(network, models, constraints,
                                      num_conv_layers * 3 * [1])
        problem.copy_strategy = "slice"  
        state, e = problem.anneal()
        total_logic_used = constraints["logic_utilization"] \
                           - problem.logic_utilization_constraint(state)
        total_multipliers = constraints["multiplier"] \
                            - problem.multiplier_constraint(state)
        total_m20k = constraints["block_memory"] \
                            - problem.bram_constraint(state)

        print "Attempt", i
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

    return scale_x(state)


    # print "Iter", iter
    # print "Estimated total logic utilization: %d (%.3f)" % \
    #         (total_logic_used,
    #          float(total_logic_used) / constraints["logic_utilization"])
    # print "Estimated total multipliers: %d (%.3f)" % \
    #         (total_multipliers,
    #          float(total_multipliers) / constraints["multiplier"])
    # print "Violation:", multiplier_constraint(results.x)
    # print "Optimal params:", parameters
    # print "Estimated GOps:", gops_fn(rounded_xs), "\n"

    return parameters


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
        # pixel a cycle. The maximum bandwidth of LMem is around 38.4GBps (
        # according to a post in mdx) - which translates to:
        #   - 19.2 GBps in a direction
        #   - 153.6 Gbps in a direction
        #   - 8.53 giga words/s in a direction (a number is 18 bits)
        #   - maximum of (8.53 * 1e9 / num_inputs) inputs per second in the
        #     first layer.
        #   - This is the bottle neck imposed by LMem transfers.
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

                prev_cycles = acc_cycles
                ops_per_cycle += calc_total_ops(layer) / acc_cycles

                minimum_cycles = min([
                        acc_cycles,
                        conv_unit_cycles,
                        scheduler_cycles,
                        minimum_cycles])

            elif layer.HasField("pool"):
                prev_cycles = prev_cycles * layer.pool.dim * layer.pool.dim

            else:
                raise RuntimeError("Unknown layer %d." % (layer.layer_id))

        # Multiply by minimum_cycles at the end so that none of the kernels
        # in the pipeline is running faster than the given clock rate.
        return 1e-9 * ops_per_cycle * clock_rate * minimum_cycles

    return fn


def main():
    with open(FLAGS.resource_bench, "r") as f:
        resource_bench = yaml.load(f.read())

    max_logic_utilization = resource_bench["resources"]["logic_utilization"]
    max_multiplier = resource_bench["resources"]["multiplier"]
    max_block_memory = resource_bench["resources"]["block_memory"]
    # Each M20k block contains 20k bits

    X = []
    Y = []


    for datum in resource_bench["data"]:
        # We do not attempt to model everything automatically, somethings are
        # simple to model:
        #
        #   1. number_of_multipliers - upper bounded by `kernel_size *
        #      kernel_size`. In some cases the multiplication tree will be
        #      optimized by Maxeler and uses less resources, but this is
        #      a good general rule of thumb.
        #   2. base_number_of_block_memory - At the bare minimum, the layer
        #      requires `B = (total_weights + total_bias) * bits_per_weight`
        #      bits of FMem, which translates to at least `B / (20,480)`
        #      M20 blocks.
        #
        # That said, what we try to model with our training data:
        #   1. logic_utilization
        #   2. residual_block_memory

        params = datum["params"]
        layer = datum["layer"]
        usage = datum["usage"]
        total_weights = ((
                layer["num_inputs"] * layer["num_outputs"]
                    * layer["kernel_size"] * layer["kernel_size"])
                + layer["num_outputs"])
        total_fifo = ((layer["kernel_size"] - 1) * layer["input_width"] * layer["num_inputs"]
                      + layer["kernel_size"])
        base_m20k = (total_weights + total_fifo) * BASE_M20K_FACTOR

        wf = float(params["worker_factor"])
        cff = float(params["conv_folding_factor"])
        kff = float(params["kernel_folding_factor"])
        X.append([wf, cff, kff])
        Y.append([usage["logic_utilization"],
                  float(usage["block_memory"] - base_m20k)])

    X = np.array(X)
    Y = np.array(Y)

    logic_utilization_model = linear_model.LinearRegression(
            normalize=True, fit_intercept=True)
    logic_utilization_model.fit(X, Y[:, 0])
    residual_block_memory_model = linear_model.LinearRegression(
            normalize=True, fit_intercept=True)
    residual_block_memory_model.fit(X, Y[:, 1])
    base_logic = logic_utilization_model.predict([0, 0, 0])

    logic_utilization_model = make_model_from_lm(logic_utilization_model)
    base_logic_model = lambda x: base_logic
    residual_logic_model = lambda x: logic_utilization_model(x) - base_logic[0]
    residual_block_memory_model = make_model_from_lm(residual_block_memory_model)

    with open(FLAGS.design, "r") as f:
        network = text_format.Parse(f.read(), parameters_pb2.Network())

    for i, layer in enumerate(network.layer):
        if i != 0:
            layer.input_height = network.layer[i - 1].output_height
            layer.input_width = network.layer[i - 1].output_width
            layer.num_inputs = network.layer[i - 1].num_outputs

        if layer.HasField("conv"):
            layer.output_height = (
                    (layer.input_height + 2 * layer.conv.pad - layer.conv.kernel_size)
                     / layer.conv.stride + 1)
            layer.output_width = (
                    (layer.input_width + 2 * layer.conv.pad - layer.conv.kernel_size)
                     / layer.conv.stride + 1)

        elif layer.HasField("pool"):
            layer.num_outputs = layer.num_inputs
            layer.output_height = layer.input_height / layer.pool.dim
            layer.output_width =  layer.input_width / layer.pool.dim

        else:
            raise RuntimeError("Unknown layer!")

    print network

    optimized_params = run_optimizer(
            network=network,
            models={
                "conv": {
                    "base_logic": base_logic_model,
                    "residual_logic": residual_logic_model,
                    "residual_block_memory": residual_block_memory_model,
                    "multiplier": lambda a: a[0] * a[1] * a[2] * 2
                }
            },
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

    with open(FLAGS.output, "w") as f:
        f.write(text_format.MessageToString(optimized_network))

    # GP Model - not very useful unless we have a lot of data.
    # ker = GPy.kern.Poly(input_dim=X.shape[1])
    # model = GPy.models.GPRegression(X, Y[:, 0:1], ker)
    # model.optimize(messages=1)
    # print(model)
    # print(model.param_array)
    # _  = model.plot()
    # plt.show()


if __name__ == "__main__":
    FLAGS = parser.parse_args()
    main()

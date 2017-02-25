from __future__ import absolute_import

import argparse
import sys

import GPy
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize
from sklearn import linear_model
import yaml

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


def constrainted_rng(lo, hi, x, stepsize):
    x = int(x + np.random.normal(stepsize / 2.3 * (hi - lo)))
    if x < lo:
        x = lo
    if x > hi:
        x = hi
    return x


class TakeStep(object):

    def __init__(self, conv_layers):
        self._conv_layers = tuple(conv_layers)
        self.stepsize = 1

    def __call__(self, x):
        ret = []
        for i in range(0, len(x), 3):
            kernel_size = (self._conv_layers[i / 3].conv.kernel_size
                    * self._conv_layers[i / 3].conv.kernel_size)
            ret.append(constrainted_rng(
                    1, self._conv_layers[i / 3].num_inputs, x[i], self.stepsize))
            ret.append(constrainted_rng(
                    1, self._conv_layers[i / 3].num_outputs, x[i + 1], self.stepsize))
            ret.append(constrainted_rng(
                    1, kernel_size, x[i + 2], self.stepsize))
        return np.array(ret)


def curry(func, *args):
    def fn(a):
        arr = args[:] + (a,)
        return func(*arr)
    return fn


def run_optimizer(network, models, constraints):

    def scale_x(x_new):
        """Scales x from integer factors to floating point factors."""
        return [(x_new[i], x_new[i+1], x_new[i+2])
                for i in range(0, len(x_new), 3)]

    def logic_utilization_constraint(x_new):
        logic_utilization = sum(
                models["conv"]["residual_logic"](x) for x in scale_x(x_new)) \
                + models["conv"]["base_logic"](x_new)
        return constraints["logic_utilization"] - logic_utilization

    def multiplier_constraint(x_new):
        multiplier = sum(
            models["conv"]["multiplier"](x) for x in scale_x(x_new))
        return constraints["multiplier"] - multiplier

    def bram_constraint(x_new):
        block_memory = sum(models["conv"]["residual_block_memory"](x)
                           for x in scale_x(x_new)) \
                       + base_bram_usage
        return constraints["block_memory"] - block_memory

    base_bram_usage = 0
    conv_layers = []
    x0 = []
    cobyla_constraints = []
    i = 0
    for layer in network.layer:
        if layer.HasField("conv"):
            conv_layers.append(layer)
            base_bram_usage += float(
                layer.num_inputs * layer.num_outputs
                * layer.conv.kernel_size * layer.conv.kernel_size)
            total_kernel_size = (layer.conv.kernel_size
                                 * layer.conv.kernel_size)
            cobyla_constraints.extend([
                {"type": "ineq",                        # wf >= 1
                 "fun": curry(lambda j, x: x[j] - 1,
                              i)},
                {"type": "ineq",                        # wf <= num_inputs
                 "fun": curry(lambda j, layer, x: layer.num_inputs - x[j],
                              i, layer)},
                {"type": "ineq",                        # cff >= 1
                 "fun": curry(lambda j, x: x[j] - 1,
                              i + 1)},
                {"type": "ineq",                        # cff <= num_outputs
                 "fun": curry(lambda j, layer, x: layer.num_outputs - x[j],
                              i + 1, layer)},
                {"type": "ineq",                        # kf >= 1
                 "fun": curry(lambda j, x: x[j] - 1,
                              i + 2)},
                {"type": "ineq",                        # kf <= kernel_size^2
                 "fun": curry(lambda j, total_kernel_size, x: total_kernel_size - x[j],
                              i + 2, total_kernel_size)}
            ])
            x0.extend([div_ceil(layer.num_inputs, 2),
                       div_ceil(layer.num_outputs, 2),
                       div_ceil(layer.conv.kernel_size * layer.conv.kernel_size,
                                2)])
            i += 3
    base_bram_usage = base_bram_usage * BASE_M20K_FACTOR

    cobyla_constraints.extend([
        {"type": "ineq", "fun": logic_utilization_constraint},
        # {"type": "ineq", "fun": bram_constraint},
        {"type": "ineq", "fun": multiplier_constraint}])
    gops_fn = make_gops_fn(network)
    print optimize.basinhopping(
            make_negate_fn(gops_fn),
            minimizer_kwargs={
                    "method": "COBYLA", "constraints": cobyla_constraints},
            x0=np.array(x0),
            niter=200,
            disp=1)


def div_ceil(a, b):
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

    def fn(factors):
        """
        Arguments:
            args: List of (wf, cff, kff) in integer form.
        """
        factors = factors[:]
        total_ops = 0
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
        prev_cycles = float(clock_rate
                            / (8.5333 * 1e9 * network.layer[0].num_inputs))
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
                factors_ptr += 3
                ops = wf * cff * kff * 2

                # ConvolutionScheduler produces a set of new output at
                # every non-border cycles.
                scheduler_cycles = (float(prev_cycles)
                        * (layer.input_height * layer.input_width)
                        / (layer.input_height - layer.conv.kernel_size + 1)
                        / (layer.input_width - layer.conv.kernel_size + 1)
                        / div_ceil(float(layer.num_inputs), wf))

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
    redisual_logic_model = lambda x: logic_utilization_model(x) - base_logic
    residual_block_memory_model = make_model_from_lm(residual_block_memory_model)

    with open(FLAGS.design, "r") as f:
        network = text_format.Parse(f.read(), parameters_pb2.Network())
    optimized_network = run_optimizer(
            network=network,
            models={
                "conv": {
                    "base_logic": base_logic_model,
                    "residual_logic": logic_utilization_model,
                    "residual_block_memory": residual_block_memory_model,
                    "multiplier": lambda a: a[0] * a[1] * a[2] * 2
                }
            },
            constraints=resource_bench["resources"])


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

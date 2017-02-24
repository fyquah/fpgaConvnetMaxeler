import argparse
import sys


import GPy
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize
from sklearn import linear_model
import yaml


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


def main(argv):
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
        total_weights = (layer["num_inputs"] * layer["num_outputs"]
                * layer["kernel_size"] * layer["kernel_size"] + layer["num_outputs"])
        base_m20k = (total_weights * 18.0) / 20480.0

        wf = float(params["worker_factor"]) \
                / layer["num_inputs"]
        cff = float(params["conv_folding_factor"]) \
                / layer["num_outputs"]
        kff = float(params["kernel_folding_factor"]) \
                / (layer["kernel_size"] * layer["kernel_size"])
        X.append([wf, cff])
        Y.append([usage["logic_utilization"],
                  float(usage["block_memory"] - base_m20k)])

    X = np.array(X)
    Y = np.array(Y)
    logic_utilization_model = linear_model.LinearRegression(
            normalize=True, fit_intercept=True)
    logic_utilization_model.fit(X, Y[:, 0])
    residual_block_memory_model = linear_model.LinearRegression(
            normalize=True, fit_intercept=True)
    residual_block_memory_model.fit(X, Y[:, 0])

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
    main(sys.argv)

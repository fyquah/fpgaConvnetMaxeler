import argparse
import sys

from fpgaconvnet.protos import parameters_pb2

import numpy as np
from scipy import optimize
from sklearn import linear_model


parser = argparse.ArgumentParser(
        description="Takes a CNN design (in fpgaconvnet.protos.Paramter.Network"
                    " protos form) and produces the ideal factor allocations "
                    " subject to resource constraints.")
parser.add_argument("design", type=str, nargs="",
                    help="Path to the protos with the design. *_folding_factor"
                         " fields in the proto will be ignored.",
                    required=True)
parser.add_argument("--resource-bench", dest="resource_bench", type=str,
                    help=("Path to yaml file with the results from running "
                          " resource benches."),
                    required=True)
parser.add_argument("--output", dest="output", type=str,
                    help="Path to the protos with the optimized output.",
                    required=True)
                     

def make_constraint(func):
    """Returns a constraint dict where func(*args) <= 0."""
    def new_func(v):
        # v is a numpy array.
        return -(func(*v))
    return {"type": "ineq", "fun": new_func}


def make_trivial_constraints(n_args):
    for arg in n_args:
        yield make_constraint()


# ===== Simple Constraint, for testing purposes ======

def simple_target(x):
    return x **2


simple_constraints = [
        make_constraint(lambda x: x + 1)]


def make_composite_factors(factors):
    return np.array([(a, b, a * b) for a,b in factors])

# ===== LeNet ====

CONV_FACTORS = np.array([
        (50, 25), (50, 5), (50, 1), (20, 25),
        (20, 1), (10, 25), (10, 5)])
COMPOSITE_CONV_FACTORS = make_composite_factors(CONV_FACTORS)


def _build_linear_models():
    # These are results obtained from running resource bench - DO NOT CHANGE THEM!
    flip_flops = np.array(
            [141265, 141305, 136848, 83962, 79929, 46909, 46441])
    dsp_blocks = np.array(
            [750, 150, 30, 500, 20, 250, 50])
    block_memory = np.array(
            [1351, 1351, 1351, 920, 920, 989, 989])
    throughput = np.array(
            [2034.0, 736.068, 312.48, 959.693, 128.705, 491.359, 158.381])
    # End of resource_bench results.

    for y in (flip_flops, dsp_blocks, block_memory, throughput):
        lm = linear_model.LinearRegression()
        lm.fit(COMPOSITE_CONV_FACTORS, y)
        yield lm


FLIP_FLOP_MODEL, DSP_BLOCK_MODEL, BLOCK_MEMORY_MODEL, THROUGHPUT_MODEL = \
        tuple(_build_linear_models())

# Use only 80% of LUT, give some buffer for pooling units.
# We do not need such conservative checks for DSP and Block Memory, as pooling unit
# do not use those.
TOTAL_FLIP_FLOPS = 262400 * 0.8
TOTAL_DSP_BLOCKS = 3926
TOTAL_BLOCK_MEMORY = 2567


def lenet_target(x):
    ff0, kff0, ff1, kff1 = x
    # It makes more sense to minimize throughput rather than
    # multiply them.
    t0 = THROUGHPUT_MODEL.predict(np.array(
            [[ff0, kff0, ff0 * kff0]]))[0]
    t1 = THROUGHPUT_MODEL.predict(np.array(
            [[ff1, kff1, ff1 * kff1]]))[0]
    # The ratio of weightage between t0 and t1  should be similar
    # to the ratio between number of convolutions in the two kernels
    # Negatate the result, because we are maimizing the function
    return -t0-50*t1


def _make_lenet_constraint_func(model, total):
    def f(ff0, kff0, ff1, kff1):
        p0 = model.predict(np.array(
                [[ff0, kff0, ff0 * kff0]]))[0]
        p1 = model.predict(np.array(
                [[ff1, kff1, ff1 * kff1]]))[0]
        return (p0 + p1 - total)
    return f


block_memory_constraint = _make_lenet_constraint_func(
        BLOCK_MEMORY_MODEL, TOTAL_BLOCK_MEMORY)
flip_flop_constraint = _make_lenet_constraint_func(
        FLIP_FLOP_MODEL, TOTAL_FLIP_FLOPS)
dsp_block_constraint = _make_lenet_constraint_func(
        DSP_BLOCK_MODEL, TOTAL_DSP_BLOCKS)


lenet_constraints = [make_constraint(func) for func in [
        block_memory_constraint, flip_flop_constraint,
        dsp_block_constraint]]
lenet_bounds = [
    (1, 20),    # ff0
    (1, 25),    # kff0
    (1, 1000),  # ff1
    (1, 25),    # kff1
]


MAPPINGS = {
        "lenet": (4, lenet_target, lenet_constraints, lenet_bounds),
        "simple": (1, simple_target, simple_constraints, [])}


def run_optimizer(params_size, target_function, constraints, bounds):
    initial_vector = [10 for _ in range(params_size)]
    options = {"disp": True, "maxfev": 100000, "maxiter": 100000}
    res = optimize.minimize(target_function,
            x0=initial_vector,
            constraints=constraints,
            bounds=bounds,
            options=options)
    print(res)


def main(argv):
    if len(argv) >= 2:
        name = argv[1]
    else:
        name = "simple"
        print("No mappings provided, using %s!" % name)
    params_size, target_function, constraints, bounds = MAPPINGS[name]
    run_optimizer(params_size, target_function, constraints, bounds)


if __name__ == "__main__":
    main(sys.argv)

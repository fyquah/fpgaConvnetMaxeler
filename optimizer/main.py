import sys

from scipy import optimize


def make_constraint(func):
    """Returns a constraint dict where func(*args) <= 0."""
    def new_func(*args):
        return -(func(*args))
    return {
            "type": "ineq",
            "fun": new_func}

# ===== Simple Constraint, for testing purposes ======

def simple_target(x):
    return x **2


simple_constraints = [
        make_constraint(lambda x: x + 1)]

# ===== LeNet ====

def lenet_target(ff0, kff0, ff1, kff1):
    return 1 * ff0


def _block_memory_constraint(ff0, kff0, ff1, kff1):
    pass


def _flip_flop_constraint(ff0, kff0, ff1, kff1):
    pass


lenet_constraints = [
        make_constraint(_block_memory_constraint),
        make_constraint(_flip_flop_constraint)]

# TODO(fyq14): Complete this
# ==== Scene Analysis ====


MAPPINGS = {
        "lenet": (lenet_target, lenet_constraints),
        "simple": (simple_target, simple_constraints)}


def run_optimizer(target_function, constraints):
    initial_vector = [-5.0]
    options = {
            "disp": True}
    res = optimize.minimize(target_function,
            x0=initial_vector, constraints=constraints, options=options)
    print res


def main(argv):
    if len(argv) >= 2:
        name = argv[1]
    else:
        name = "simple"
        print "No mappings provided, using %s!" % name
    target_function, constraints = MAPPINGS[name]
    run_optimizer(target_function, constraints)


if __name__ == "__main__":
    main(sys.argv)

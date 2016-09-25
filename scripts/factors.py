
# [(int, int)], fst is foldingFactor, snd is kernelFoldingFactor
factors = [
    (50, 1),
    (20, 1),
    (10, 1),
    (50, 5),
    (20, 5),
    (10, 5),
    (50, 25),
    (20, 25),
    (10, 25),
]

sim_factors = [
    (20, 5)
]

def make_max_name(a, b):
    return "resource_bench_%d_%d" % (a,b)

def make_sim_path(max_file_name):
    return "%s_$(DFEModel)_DFE_SIM/results/%s.max" % (max_file_name, max_file_name)

def make_sim_header(max_file_name):
    return "%s_$(DFEModel)_DFE_SIM/results/%s.h" % (max_file_name, max_file_name)

def make_sim_include_flag(max_file_name):
    return "-I%s_$(DFEModel)_DFE_SIM/results" % (max_file_name)

def make_path(max_file_name):
    return "%s_$(DFEModel)_DFE/results/%s.max" % (max_file_name, max_file_name)

def make_header(max_file_name):
    return "%s_$(DFEModel)_DFE/results/%s.h" % (max_file_name, max_file_name)

def make_include_flag(max_file_name):
    return "-I%s_$(DFEModel)_DFE/results" % (max_file_name)



# [(int, int)], fst is foldingFactor, snd is kernelFoldingFactor
factors = [
    (100, 10),
    (90, 10),
    (80, 10),
    (70, 10),
    (60, 15),
    (80, 15),
    (70, 15),
    (60, 15)
    (70, 20),
    (50, 20),
    (40, 20),
]

sim_factors = [
    (1,13)
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


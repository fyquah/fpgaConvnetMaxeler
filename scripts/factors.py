
# [(int, int)], fst is foldingFactor, snd is kernelFoldingFactor
factors = [
    (7,2),
    (1,3),
    (5,6),
    (1,13),
    (10,1),
    (4,9),
    (1,6),
    (2,6),
    (1,20),
    (4,6)
]

sim_factors = [
    (1,3),
    (2,3),
    (3,3)
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


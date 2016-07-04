#!/usr/bin/env python

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

# [(int, int)], fst is foldingFactor, snd is kernelFoldingFactor
# factors = [
#     (1,3),
#     (5,6),
#     (1,13),
#     (10,1),
#     (4,9),
#     (1,6),
#     (7,2),
#     (5,7),
#     (2,6),
#     (1,20),
#     (2,9),
#     (4,8)
# ]

makefile = open("Makefile.resource.benchmark", "w")
factors = [(1,3), (2,4)]
max_names = map(lambda (a, b): make_max_name(a, b), factors)

print >>makefile, "# === resource benchmark simulation ===\n"

for (a, b) in factors:
    max_name = make_max_name(a, b)
    s = """%s: $(ENGINEFILES)
\t$(MAXJC) $(JFLAGS) $(ENGINEFILES)
\tMAXAPPJCP='.:$(CP)' MAXSOURCEDIRS='../src' $(MAXJAVARUN) -v -m 8192 $(MANAGER) DFEModel=$(DFEModel) maxFileName=%s target='DFE_SIM' enableMPCX=$(MPCX) convFoldingFactor=%d kernelFoldingFactor=%d
""" % (make_sim_path(max_name), max_name, a, b)
    print >>makefile, s


print >>makefile, "RESOURCE_BENCH_SIM_MAXFILES=" + " ".join(map(make_sim_path, max_names))
print >>makefile, "\n"
print >>makefile, "RESOURCE_BENCH_SIM_INCLUDE_FLAGS=" + " ".join(map(make_sim_include_flag, max_names))
print >>makefile, "\n"
print >>makefile, " === END === "
print >>makefile, "\n"


print >>makefile, "# === resource benchmark DFE ===\n"

for (a, b) in factors:
    max_name = make_max_name(a, b)
    s = """%s: $(ENGINEFILES)
\t$(MAXJC) $(JFLAGS) $(ENGINEFILES)
\tMAXAPPJCP='.:$(CP)' MAXSOURCEDIRS='../src' $(MAXJAVARUN) -v -m 8192 $(MANAGER) DFEModel=$(DFEModel) maxFileName=%s target='DFE' enableMPCX=$(MPCX) convFoldingFactor=%d kernelFoldingFactor=%d
""" % (make_path(max_name), max_name, a, b)
    print >>makefile, s

print >>makefile, "RESOURCE_BENCH_MAXFILES=" + " ".join(map(make_path, max_names))
print >>makefile, "\n"
print >>makefile, "RESOURCE_BENCH_INCLUDE_FLAGS=" + " ".join(map(make_include_flag, max_names))
print >>makefile, "\n"
print >>makefile, " === END === "
print >>makefile, "\n"

makefile.close()

header = open("build/resource_benchmark.h", "w")
for name in max_names:
    print >>header, "#include \"%s.h\"" % name
header.close()


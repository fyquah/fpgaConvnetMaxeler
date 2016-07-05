#!/usr/bin/env python
import factors
import os

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

def make_slic_object_file_path(max_name):
    return max_name + "_sim.o"

my_dir = os.path.dirname(os.path.realpath(__file__))
makefile = open(os.path.join(my_dir, "../Makefile.resource.benchmark"), "w")
max_names = map(lambda (a, b): make_max_name(a, b), factors.factors)
sim_max_names = map (lambda (a, b): make_max_name(a, b), factors.sim_factors)

print >>makefile, "# === resource benchmark simulation ===\n"

for (a, b) in factors.sim_factors:
    max_name = make_max_name(a, b)
    s = """%s: $(ENGINEFILES)
\t$(MAXJC) $(JFLAGS) $(ENGINEFILES)
\tMAXAPPJCP='.:$(CP)' MAXSOURCEDIRS='../src' $(MAXJAVARUN) -v -m 8192 $(MANAGER) DFEModel=$(DFEModel) maxFileName=%s target='DFE_SIM' enableMPCX=$(MPCX) convFoldingFactor=%d kernelFoldingFactor=%d

%s_sim.o: %s
	$(SLICCOMPILE) $< $@
""" % (make_sim_path(max_name), max_name, a, b, max_name, make_sim_path(max_name))
    print >>makefile, s


print >>makefile, "RESOURCE_BENCH_SIM_MAXFILES=" + " ".join(map(make_sim_path, sim_max_names))
print >>makefile, "\n"
print >>makefile, "RESOURCE_BENCH_SIM_INCLUDE_FLAGS=" + " ".join(map(make_sim_include_flag, sim_max_names))
print >>makefile, "\n"
print >>makefile, "RESOURCE_BENCH_SIM_SLIC_OBJECT_FILES=" + " ".join(map(make_slic_object_file_path, sim_max_names))
print >>makefile, "\n"
print >>makefile, "# === END === "
print >>makefile, "\n"


print >>makefile, "# === resource benchmark DFE ===\n"

for (a, b) in factors.factors:
    max_name = make_max_name(a, b)
    s = """%s: $(ENGINEFILES)
\t$(MAXJC) $(JFLAGS) $(ENGINEFILES)
\tMAXAPPJCP='.:$(CP)' MAXSOURCEDIRS='../src' $(MAXJAVARUN) -v -m 8192 $(MANAGER) DFEModel=$(DFEModel) maxFileName=%s target='DFE' enableMPCX=$(MPCX) convFoldingFactor=%d kernelFoldingFactor=%d

%s_dfe.o: %s
	$(SLICCOMPILE) $< $@
""" % (make_path(max_name), max_name, a, b, max_name, make_path(max_name))
    print >>makefile, s

print >>makefile, "RESOURCE_BENCH_MAXFILES=" + " ".join(map(make_path, max_names))
print >>makefile, "\n"
print >>makefile, "RESOURCE_BENCH_INCLUDE_FLAGS=" + " ".join(map(make_include_flag, max_names))
print >>makefile, "\n"
print >>makefile, "RESOURCE_BENCH_SLIC_OBJECT_FILES=" + " ".join(map(make_slic_object_file_path, max_names))
print >>makefile, "\n"
print >>makefile, "# === END === "
print >>makefile, "\n"

makefile.close()


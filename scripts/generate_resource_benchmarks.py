#!/usr/bin/env python
import os
import factors

my_dir = os.path.dirname(os.path.realpath(__file__))

max_names = map(lambda (a, b) : factors.make_max_name(a, b), factors.factors)
sim_max_names = map(lambda (a, b) : factors.make_max_name(a, b), factors.sim_factors)

header = open(os.path.join(my_dir, "../build/resource_bench_helper.h"), "w")

print >>header, "#ifndef RESOURCE_BENCH_HELPER_H"
print >>header, "#define RESOURCE_BENCH_HELPER_H"
print >>header, "#ifdef __SIM__"
for name in sim_max_names:
    print >>header, "#include \"%s.h\"" % name
print >>header, "#else"
for name in max_names:
    print >>header, "#include \"%s.h\"" % name
print >>header, "#endif"

print >>header, "\n"
print >>header, "#include \"../src/resource_bench.h\"\n"
print >>header, "void run_resource_bench();"
print >>header, "template <typename action_t>"
print >>header, "void resource_benchmark(max_file_t* max_file, void (*run_fnc)(max_engine_t*, action_t*), std::string out_file_name);"
print >>header, "#endif"
header.close()

src = open(os.path.join(my_dir, "../build/resource_bench_helper.cpp"), "w")

print >>src, "#include \"resource_bench_helper.h\""
print >>src, "\n\n"
print >>src, "#ifndef __SIM__"
print >>src, "void run_resource_bench () {"

for (a, b) in factors.factors:
    print >>src, """
    max_file_t* max_file_%d_%d = resource_bench_%d_%d_init();
    resource_benchmark<resource_bench_%d_%d_actions_t>(max_file_%d_%d, resource_bench_%d_%d_run, "%d_%d.out");
    resource_bench_%d_%d_free();
    """ % (a,b,a,b,a,b,a,b,a,b,a,b,a,b)

print >>src, "}\n"
print >>src, "#else\n"

print >>src, "void run_resource_bench () {"


for (a, b) in factors.sim_factors:
    print >>src, """
    max_file_t* max_file_%d_%d = resource_bench_%d_%d_init();
    resource_benchmark<resource_bench_%d_%d_actions_t>(max_file_%d_%d, resource_bench_%d_%d_run, "%d_%d.out");
    resource_bench_%d_%d_free();
    """ % (a,b,a,b,a,b,a,b,a,b,a,b,a,b)

print >>src, "}"
print >>src, "#endif"


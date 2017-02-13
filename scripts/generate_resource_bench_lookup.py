import sys

args = sys.argv[2:]
function_names = ["%s_init" % argv for argv in args]

structs = ",\n    ".join(
        "{\"%s\", &%s_init}" % (arg, arg) for arg in args)
includes = "\n".join("#include \"%s.h\"" % arg for arg in args)

src = """
#ifndef RESOURCE_BENCH_LOOKUP_H
#define RESOURCE_BENCH_LOOKUP_H

#include "MaxSLiCInterface.h"

%s

namespace {

struct lookup_t {
    const char *name;
    max_file_t*(*fnc)();
};


const lookup_t resource_bench_lookup_table[%d] = {
    %s 
};

} // namespace

#endif
""" % (includes, len(args), structs)

with open(sys.argv[1], "w") as f:
    f.write(src)

import sys
import collections

def main():
    target_names = sys.argv[1:]
    d = collections.defaultdict(list)
    for arg in target_names:
        assert(arg.startswith("target_"))
        _, bitstream, fpga = arg.split("_")
        d[int(bitstream)].append(int(fpga))


    print("#ifndef FPGACONVNET_TARGETS_H")
    print("#define FPGACONVNET_TARGETS_H")
    print("")

    for target in target_names:
        print("#include \"%s.h\"" % target)
    print("")
    print("static std::vector<std::vector<max_file_t*>> targets_init() {")
    print("  std::vector<std::vector<max_file_t*>> ret;")

    for b in range(max(d.keys()) + 1):
        print("  ret.push_back(std::vector<max_file_t*>());")
        for f in d[b]:
            print("  ret.back().push_back(target_%d_%d_init());" % (b, f))

    print("  return ret;")
    print("}")
    print("#endif")


if __name__ == "__main__":
    main()

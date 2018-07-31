#!/usr/bin/env python

import argparse
import sys

from google.protobuf import text_format
from fpgaconvnet.protos import parameters_pb2


parser = argparse.ArgumentParser(
        description=("Generate the makefile for a multi-FPGA project."))
parser.add_argument("--descriptor", type=str, help="Name of network descriptor.")
parser.add_argument("--output", type=str, help="Name of the generated Makefile.")


def main():
    with open(FLAGS.descriptor, "r") as f:
        network = text_format.Parse(f.read(), parameters_pb2.Network())
    arr = []
    for i, layer in enumerate(network.layer):
        if layer.HasField("bitstream_id"):
            bitstream_id = layer.bitstream_id
        else:
            bitstream_id = 0
        fpga_id = layer.fpga_id

        if (bitstream_id, fpga_id) not in arr:
            arr.append((bitstream_id, fpga_id))

    with open(FLAGS.output, "w") as f:
        f.write("TARGET_NAMES= " + " ".join(["target_%d_%d" % (i, j) for (i, j) in arr]))


if __name__ == "__main__":
    FLAGS = parser.parse_args()
    main()

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
        if len(arr) == 0 or arr[-1] != layer.fpga_id:
            arr.append(layer.fpga_id)

        if not layer.HasField("fpga_id"):
            raise RuntimeError("Missing fpga_id in layer %d!" % i)

    with open(FLAGS.output, "w") as f:
        f.write("TARGET_NAMES= " + " ".join(["target_%d" % i for i in arr]))


if __name__ == "__main__":
    FLAGS = parser.parse_args()
    main()

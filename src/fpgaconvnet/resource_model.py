"""Resource modelling on the FPGAs.

Convention:
    *_layer_* models the output to the unit. I.e: It models the net io.output()
    (including FIFOs and stream statuses) but not the net io.input().
    Modeling that is the responsibility of the subsequent kernel. The only
    exceptions to this rule is weights loaded from CPU and LMem.

"""
from __future__ import absolute_import

import math
import sys

from google.protobuf import text_format
from fpgaconvnet.protos import parameters_pb2

NUM_BITS = 18
DEFAULT_FIFO_SIZE = 512
M20K_SIZE = 20480

Resource = collections.namedtuple("Resource", ["flip_flop", "lut", "bram"])


def div_ceil(a, b):
    if a % b == 0:
        return a / b
    else:
        return a / b + 1


def gcd(a, b):
    if a % b == 0:
        return a
    else:
        return gcd(b, a % b)


def lcm(a, b):
    return a / gcd(a, b) * b


def log2(a):
    return math.log(a) / math.log(2)


def is_cpu_init(layer):
    total_iters = (
            div_ceil(layer.num_inputs, layer.conv.worker_factor)
            * div_ceil(layer.num_outputs, layer.conv.conv_folding_factor));
    return total_iters == layer.conv.bram_factor


def conv_layer_dsp(layer):
    return (layer.conv.worker_factor
            * layer.conv.conv_folding_factor
            * layer.conv.kernel_folding_factor)


def conv_layer_lut(layer):
    wf = layer.conv.worker_facrtor
    cff = layer.conv.conv_folding_factor
    kff = layer.conv.kernel_folding_factor
    scheduler = (
            4718 * float(layer.num_inputs) / layer.conv.wokrer_factor
            + 621.66 * layer.conv.worker_factor
            - 3575.2)
    unit = (
            0.26559 * layer.conv.worker_factor * (layer.conv.kernel_size ** 2)
            + 24.505 * layer.conv.worker_factor
                * layer.conv.conv_folding_factor
                * layer.conv.kernel_folding_factor
            + 3754.3)
    acc = (
        + 49.874 * wf * cff
        - 456.46 * cff
        + 7595.4)
    streams = (2 * wf + 1) * (402 + 100)  # StreamStatus + fifo

    if is_cpu_init(layer):
        ctr = 2
        width = kff * 32
        if ((width >= 128 and width % 128 != 0)
                or width < 128 and 128 % width != 0):
            ctr += 1
        streams += ctr * (402 + 100)

    else:
        streams +=  2 * (402 + 100)

    return scheduler + unit + acc + streams


def conv_layer_flip_flop(layer):
    wf = layer.conv.worker_facrtor
    cff = layer.conv.conv_folding_factor
    kff = layer.conv.kernel_folding_factor
    scheduler = (
            layer.num_inputs * (layer.conv.kernel_size ** 2)
            + math.abs(layer.num_inputs - wf))
    unit = (
            -38.779 * wf * cff
            + 39.529 * wf * cff * kff
            + 6272.7)
    acc = (
            layer.num_outputs * layer.conv.look_ahaed * NUM_BITS
            + 49.874 * wf * cff
            - 456.46 * cff
            + 7595.4)
    streams = (2 * wf + 1) * (100 + 300)  # StreamStatus + fifo

    if is_cpu_init(layer):
        ctr = 2
        width = kff * 32
        if ((width >= 128 and width % 128 != 0)
                or width < 128 and 128 % width != 0):
            ctr += 1
        streams += ctr * (402 + 100)

    else:
        streams +=  2 * (402 + 100)

    return scheduler + unit + acc + streams


def conv_layer_bram(layer):
    wf = layer.conv.worker_factor
    cff = layer.conv.conv_folding_factor
    kff = layer.conv.kernel_folding_factor
    kernel_size = layer.conv.kernel_size
    num_outputs = layer.num_outputs
    scheduler = -238.3 * log2(layer.conv.worker_factor) \
                + 70 + 960 / 30 * layer.num_inputs
    unit = (layer.conv.bram_factor
            + max(0, 0.0.9187 * kff - 5.8784)
            + layer.conv.kernel_size ** 2)
    accumulator = 2
    streams = (
            # from scheduler -> unit
            wf * math.ceil(DEFAULT_FIFO_DEPTH * (kernel_size ** 2) * NUM_BITS
                           / M20K_SIZE)

            # from unit -> acc
            + wf * math.ceil(DEFAULT_FIFO_DEPTH * cff * NUM_BITS / M20K_SIZE)

            # from acc -> <next>
            + wf * math.ceil(DEFAULT_FIFO_DEPTH * num_outputs * NUM_BITS
                              / M20K_SIZE))
    if is_cpu_init(layer):
        streams += math.ceil(DEFAULT_FIFO_DEPTH * kff * 32 / M20K_SIZE)
        streams += math.ceil(DEFAULT_FIFO_DEPTH * 128 / M20K_SIZE)
        width = kff * 32

        if is_compatible_streams(width, 128):
            # Fifo between DualAspectReg and DualAspectMux
            dual_aspect_width = lcm(width, 128)
            streams += math.ceil(
                DEFAULT_FIFO_DEPTH * dual_aspect_width / M20K_SIZE)

    else:
        # The LMem Stream Size for our target FPGA is 384 bytes
        # = 96 floating point numbers
        lmem_stream_size = div_ceil(kff * cff * wf, 96) * 96 * 32

        # TODO(fyq14): Model FIFO right after LMEM.
        streams += math.ceil(DEFAULT_FIFO_DEPTH * lmem_stream_size / M20K_SIZE)

    return scheduler + unit + accumulator + streams


# TODO(fyq14): This Model is broken. The BRAM usage goes down after some point,
#              but I have no idea how to model that point.
def pool_layer_bram(layer):
    channel_folding_factor = layer.pool.channel_folding_factor
    return (
            layer.input_width * 20.0 * layer.num_inputs / 20480.0
                * math.log2(channel_folding_factor)
            - 11.2)


def pool_layer_lut(layer):
    return (
            34.9 * layer.pool.dim * layer.pool.dim * layer.pool.channel_folding_factor
            + 1.6152 * layer.input_width * layer.num_inputs)


def pool_layer_flip_flop(layer):
    return (
            136.26 * layer.pool.dim * layer.pool.dim * layer.pool.channel_folding_factor
            + 399.39 * layer.num_inputs)


def project(network):
    lut = 0.0
    flip_flop = 0.0
    bram = 0.0
    dsp = 0.0

    per_fpga_constants = [
        # CheckSumMappedDRP
        (40, 37, 1)

        # Max4Pcie
        (833, 1000, 2),

        # PCIEBase
        (1647, 941, 4),

        # PCIESreaming
        (5848, 8889, 54),

        # SignalForwardingAdapter
        (90, 84, 1)
    ]

    for a, b, c in per_fpga_constants:
        lut += a
        flip_flop += b
        bram += c

    # If any of the layer uses off-chip weight transfer, then only we need this.
    for layer in network.layer:
        if not is_cpu_init(layer):
            off_chip_constants = [
                # MemoryControllerPro
                (6797, 2830, 95),

                # MemoryController
                (11203, 11805, 30)
            ]

            for a, b, c in off_chip_constants:
                lut += a
                flip_flop += b
                bram += c
            break

    # fromcpu
    width = network.layers[0].num_inputs * 32
    lut += 400
    flip_flop += 502
    bram += wf * math.ceil(DEFAULT_FIFO_DEPTH * width / M20K_SIZE)
    if not is_compatible_streams(128, network.layers[0].num_inputs * 32):
        lut += 400
        flip_flop += 502
        bram += math.ceil(
            DEFAULT_FIFO_DEPTH * lcm(width, 128) / M20K_SIZE)

    # Kernel streaming and actual computation work.
    for layer in network.layers:
        if layer.HasField("conv"):
            lut += conv_layer_lut(layer)
            flip_flop += conv_layer_flip_flop(layer)
            bram += conv_layer_bram(layer)
            dsp += conv_layer_dsp(layer)

        elif layer.HasField("pool"):
            lut += pool_layer_lut(layer)
            flip_flop += pool_layer_flip_flop(layer)
            bram += pool_layer_bram(layer)

        elif layer.HasField("lrn"):
            lut += lrn_layer_lut(layer)
            flip_flop += lrn_layer_flip_flop(layer)
            bram += lrn_layer_bram(layer)

    # tocpu
    width = network.layers[-1].num_outputs * 32
    lut += 400
    flip_flop += 502
    bram += math.ceil(DEFAULT_FIFO_DEPTH * width / M20K_SIZE)
    if not is_compatible_streams(128, width):
        lut += 400
        flip_flop += 502
        bram += math.ceil(
            DEFAULT_FIFO_DEPTH * lcm(width, 128) / M20K_SIZE)

    return Resource(bram=bram, flip_flop=flip_flop, lut=lut)


def is_compatible_streams(a, b):
    return (a < b and b % a == 0) or (a >= b and a % b == 0)



def main():
    pass


if __name__ == "__main__":
    main()

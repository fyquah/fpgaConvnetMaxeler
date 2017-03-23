"""Resource modelling on the FPGAs.

Convention:
    *_layer_* models the output to the unit. I.e: It models the net io.output()
    (including FIFOs and stream statuses) but not the net io.input().
    Modeling that is the responsibility of the subsequent kernel. The only
    exceptions to this rule is weights loaded from CPU and LMem.

"""
from __future__ import absolute_import

import collections
import logging
import math
import sys

from google.protobuf import text_format
from fpgaconvnet.protos import parameters_pb2

ACTUAL_BITS = 18.
NUM_BITS = 18.
DEFAULT_FIFO_DEPTH = 512.
M20K_SIZE = 20480.

# Resource constraints
MAX_DSP = 1963.
MAX_BRAM = 2567.
MAX_LUT = 524800.
MAX_FF = 1049600.


Resource = collections.namedtuple(
    "Resource", ["flip_flop", "lut", "bram", "dsp"])


def div_ceil(a, b):
    if a % b == 0:
        return a / b
    else:
        return a / b + 1


def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a % b)


def lcm(a, b):
    return a / gcd(a, b) * b


def log2(a):
    return math.log(a) / math.log(2)


def is_cpu_init(layer):
    total_iters = (
            layer.conv.worker_factor * div_ceil(layer.num_inputs, layer.conv.worker_factor)
            * layer.conv.conv_folding_factor * div_ceil(layer.num_outputs, layer.conv.conv_folding_factor));
    return total_iters == layer.conv.bram_factor


def is_compatible_streams(a, b):
    return (a < b and b % a == 0) or (a >= b and a % b == 0)


def conv_layer_dsp(layer):
    return (layer.conv.worker_factor
            * layer.conv.conv_folding_factor
            * layer.conv.kernel_folding_factor)


def calc_total_iters(layer):
    return (div_ceil(layer.num_inputs, layer.conv.worker_factor)
            * div_ceil(layer.num_outputs, layer.conv.conv_folding_factor)
            * div_ceil(layer.conv.kernel_size * layer.conv.kernel_size,
                       layer.conv.kernel_folding_factor))


def conv_layer_lut(layer):
    wf = layer.conv.worker_factor
    cff = layer.conv.conv_folding_factor
    kff = layer.conv.kernel_folding_factor
    scheduler = (
            4718 * float(layer.num_inputs) / wf
            + 621.66 * wf
            - 3575.2)
    unit = (
            0.26559 * wf * (layer.conv.kernel_size ** 2)
            + 24.505 * wf * cff * kff
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
    wf = layer.conv.worker_factor
    cff = layer.conv.conv_folding_factor
    kff = layer.conv.kernel_folding_factor
    scheduler = (
            95.65 * layer.num_inputs * (layer.conv.kernel_size ** 2)
            + 40.61 * abs(layer.num_inputs - wf))
    unit = (
            -38.779 * wf * cff
            + 39.529 * wf * cff * kff
            + 6272.7)
    acc = (
            layer.num_outputs * layer.conv.look_ahead * NUM_BITS
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
    weight_bits_per_multiplier = (
            div_ceil(layer.conv.kernel_size * layer.conv.kernel_size,
                     layer.conv.kernel_folding_factor)
            * layer.conv.bram_factor / (wf * cff))
    unit = (layer.conv.bram_factor
            + max(0, 0.09187 * kff - 5.8784)
            + (7.0248 * (layer.conv.kernel_size ** 2))
            + wf * cff * kff * math.ceil(weight_bits_per_multiplier / 20480))
    accumulator = 2 + layer.num_outputs
    scheduler_unit_fifo_bram = math.ceil(
            DEFAULT_FIFO_DEPTH * (kernel_size ** 2) * NUM_BITS / M20K_SIZE)
    unit_acc_fifo_bram = math.ceil(
            DEFAULT_FIFO_DEPTH * cff * NUM_BITS / M20K_SIZE)
    acc_next = math.ceil(DEFAULT_FIFO_DEPTH * num_outputs * NUM_BITS / M20K_SIZE)
    streams = (
            # from scheduler -> unit
            wf * scheduler_unit_fifo_bram

            # from unit -> acc
            + wf * unit_acc_fifo_bram

            # from acc -> <next>
            + acc_next)

    if is_cpu_init(layer):
        streams += math.ceil(DEFAULT_FIFO_DEPTH * kff * 32 / M20K_SIZE)
        streams += math.ceil(DEFAULT_FIFO_DEPTH * 128 / M20K_SIZE)
        width = kff * 32

        if not is_compatible_streams(width, 128):
            # Fifo between DualAspectReg and DualAspectMux
            dual_aspect_width = lcm(width, 128)
            logging.debug("layer %d dual aspect width = %.3f" %
                          (layer.layer_id, dual_aspect_width))
            fifo_dual_aspect_width = math.ceil(
                    DEFAULT_FIFO_DEPTH * dual_aspect_width / M20K_SIZE)
            logging.debug("Layer %d weights dual aspect fifo = %.3f"
                          % (layer.layer_id, fifo_dual_aspect_width))
            streams += fifo_dual_aspect_width

    else:
        # The LMem Stream Size for our target FPGA is 384 bytes
        # = 96 floating point numbers
        lmem_stream_size = div_ceil(kff * cff * wf, 96) * 96 * 32

        # TODO(fyq14): Model FIFO right after LMEM.
        streams += math.ceil(DEFAULT_FIFO_DEPTH * lmem_stream_size / M20K_SIZE)

    logging.debug("Layer %d BRAM scheduler: %.3f" % (layer.layer_id, scheduler))
    logging.debug("Layer %d BRAM convolution unit: %.3f" % (layer.layer_id, unit))
    logging.debug("Layer %d BRAM accumulator: %.3f" % (layer.layer_id, accumulator))
    logging.debug("Layer %d BRAM used by kernels: %.3f"
                  % (layer.layer_id, scheduler + unit + accumulator))
    logging.debug("Layer %d BRAM streams: %.3f" % (layer.layer_id, streams))
    logging.debug("Layer %d BRAM scheduler -> convUnit: %.3f"
                  % (layer.layer_id, scheduler_unit_fifo_bram))
    logging.debug("Layer %d BRAM convUnit -> accumulator: %.3f"
                  % (layer.layer_id, unit_acc_fifo_bram))
    logging.debug("Layer %d BRAM accumulator -> <next>: %.3f"
                  % (layer.layer_id, acc_next))

    return scheduler + unit + accumulator + streams


# TODO(fyq14): This Model is broken. The BRAM usage goes down after some point,
#              but I have no idea how to model that point.
def pool_layer_bram(layer):
    channel_folding_factor = layer.pool.channel_folding_factor
    kernel_bram = (
        layer.input_width * 20.0
            * layer.num_inputs / 20480.0 * log2(channel_folding_factor)
        - 11.2)
    if layer.is_last_layer:
        bits = 32
    else:
        bits = NUM_BITS

    stream_bram = math.ceil(
            DEFAULT_FIFO_DEPTH * layer.num_inputs * bits / M20K_SIZE)
    logging.debug("Layer %d pooling stream_bram = %.3f" %
                  (layer.layer_id, stream_bram))
    return kernel_bram + stream_bram


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
        (40, 37, 1),

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
        if layer.HasField("conv"):
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
    width = network.layer[0].num_inputs * 32
    stream_input_fifo = math.ceil(DEFAULT_FIFO_DEPTH * width / M20K_SIZE)

    lut += 400
    flip_flop += 502
    bram += math.ceil(DEFAULT_FIFO_DEPTH * 128 / M20K_SIZE)
    bram += stream_input_fifo

    logging.debug("Stream input fifo BRAM = %.3f" % stream_input_fifo)

    if not is_compatible_streams(128, width):
        lut += 400
        flip_flop += 502
        non_multiple_transition_fifo = math.ceil(
            DEFAULT_FIFO_DEPTH * lcm(width, 128) / M20K_SIZE)
        bram += non_multiple_transition_fifo
        logging.debug("Input non multiple transition fifo BRAM = %.3f"
                      % non_multiple_transition_fifo)

    # Kernel streaming and actual computation work.
    for layer in network.layer:

        logging.debug("-----------------")

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

        logging.debug("-----------------")

    # tocpu
    width = network.layer[-1].num_outputs * 32
    lut += 400
    flip_flop += 502
    stream_output_fifo = math.ceil(DEFAULT_FIFO_DEPTH * width / M20K_SIZE)
    bram += stream_output_fifo
    bram += math.ceil(DEFAULT_FIFO_DEPTH * 128 / M20K_SIZE)  # Final connection via PCIe
    logging.debug("Stream output width = %.3f" % width)
    logging.debug("Stream output fifo BRAM = %.3f" % stream_output_fifo)

    if not is_compatible_streams(128, width):
        lut += 400
        flip_flop += 502
        dual_aspect_width = lcm(width, 128)
        non_multiple_transition_fifo = math.ceil(
            DEFAULT_FIFO_DEPTH * dual_aspect_width / M20K_SIZE)
        logging.debug("Stream output dual aspect width = %.3f"
                       % dual_aspect_width);
        logging.debug("Stream output non multiple transition fifo BRAM = %.3f"
                      % non_multiple_transition_fifo)
        bram += non_multiple_transition_fifo


    return Resource(bram=bram, flip_flop=flip_flop, lut=lut, dsp=dsp)


if __name__ == "__main__":
    main()
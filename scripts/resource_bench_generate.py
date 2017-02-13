#!/usr/bin/env python
import numpy as np
from scipy import signal

inputHeight = 30
inputWidth = 30
inputChannels = 32
outputChannels = 96
kernelDim = 5
testCases = 100
outputHeight = inputHeight - kernelDim + 1
outputWidth = inputWidth - kernelDim + 1


def flip_matrix(mat):
    m = np.zeros(mat.shape)
    height = mat.shape[0]
    width = mat.shape[1]

    for r in range(height):
        for c in range(width):
            m[r, c] = mat[height - 1 - r][width - 1 - c]

    return m


def dump_matrix(filename, data):
    with open(filename, "w") as f:
        for row in data:
            f.write(" ".join(str(x) for x in row) + "\n")


def dump_kernel(o, i, mat):
    filename = "convolution_kernels/mat_" + str(o) + "_" + str(i) + ".txt"
    f = open(filename, "w")
    dump_matrix(f, [mat])
    f.close()

def dump_bias(bias):
    filename = "convolution_kernels/bias.txt"
    f = open(filename, "w")
    assert len(bias) == outputChannels
    for b in bias:
        f.write("%.5f " % b)
    f.close()


def cnn_layer(inputData, kernels, bias):
    output = np.zeros((outputHeight, outputWidth, outputChannels))
    for o in range(outputChannels):
        out = 0
        for i in range(inputChannels):
            out = out + signal.convolve2d(inputData[:, :, i], flip_matrix(kernels[o][i]), mode="valid")
        output[:, :, o] = out + bias[o]
    return np.array(np.maximum(output, 0))


def main():
    kernels = np.random.randn(outputChannels, inputChannels, kernelDim, kernelDim) * 0.05
    bias = 0.05 * np.random.randn(outputChannels)
    inputs = np.random.randn(testCases, inputHeight, inputWidth, inputChannels)
    outputs = np.array(
            [cnn_layer(input_vector, kernels, bias) for input_vector in inputs])
    print(outputs.shape)

    dump_matrix("test_data/resource_bench/kernels.txt",
                kernels.reshape((outputChannels * inputChannels,
                                 kernelDim * kernelDim)))
    dump_matrix("test_data/resource_bench/bias.txt",
                bias.reshape((outputChannels, 1)))
    dump_matrix("test_data/resource_bench/inputs.txt",
                inputs.reshape((testCases, -1)))
    dump_matrix("test_data/resource_bench/outputs.txt",
                outputs.reshape((testCases, -1)))


if __name__ == "__main__":
    main()

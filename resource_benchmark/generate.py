#!/usr/bin/env python
import numpy as np
from scipy import signal

inputHeight = 28
inputWidth = 28
inputChannels = 10
outputChannels = 20
kernelDim = 5
testCases = 100

def flip_matrix(mat):
    m = np.zeros(mat.shape)
    height = mat.shape[0]
    width = mat.shape[1]

    for r in range(height):
        for c in range(width):
            m[r, c] = mat[height - 1 - r][width - 1 - c]

    return m

def dump_matrix(f, mat):
    n_channels = len(mat)
    height = mat[0].shape[0]
    width = mat[0].shape[1]

    for r in range(height):
        for c in range(width):
            for x in range(n_channels):
                f.write('{:0.5f} '.format(mat[x][r, c]))
        f.write("\n")

def dump_kernel(o, i, mat):
    filename = "convolution_kernels/mat_" + str(o) + "_" + str(i) + ".txt"
    f = open(filename, "w")
    dump_matrix(f, [mat])
    f.close()

def dump_test_data(inputData, mat, i):
    input_filename = "inputs/input_" + str(i) + ".txt"
    output_filename = "outputs/output_" + str(i) + ".txt"

    output = [None for o in range(outputChannels)]
    for o in range(outputChannels):
        out = 0
        for i in range(inputChannels):
            out = out + signal.convolve2d(inputData[i], flip_matrix(mat[o][i]), mode="valid")
        output[o] = out


    with open(input_filename, "w") as f: dump_matrix(f, inputData)
    with open(output_filename, "w") as f: dump_matrix(f, output)
    

kernels = [[None for i in range(inputChannels)] for x in range(outputChannels)]

# inputs
for o in range(outputChannels):
    for i in range(inputChannels):
        kernels[o][i] = np.random.randn(kernelDim, kernelDim) * 0.1
        dump_kernel(o, i, kernels[o][i])

# outputs
for i in range(testCases):
    inputData = [ 0.1 * np.random.randn(inputHeight, inputWidth) for c in range(inputChannels)]
    dump_test_data(inputData, kernels, i)

# properties
with open("config.properties", "w") as f:
    f.write("inputHeight=" + str(inputHeight) + "\n")
    f.write("inputWidth=" + str(inputWidth) + "\n")
    f.write("inputChannels=" + str(inputChannels) + "\n")
    f.write("outputChannels=" + str(outputChannels) + "\n")
    f.write("kernelDim=" + str(kernelDim) + "\n")
    f.write("testCases=" + str(testCases) + "\n")


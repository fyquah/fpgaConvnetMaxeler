#!/usr/bin/env python

import struct
import sys

import numpy as np
from scipy import signal


def convolve(input_images, filters, bias):
    """
    Arguments:
        input_images: [N * height * width * channels]
        filters: [output_channels * input_channels * filter_height
                  * filter_width]
        bias: [ouput_channels]
    """
    output_height = input_images.shape[1] - filters.shape[2] + 1
    output_width = input_images.shape[2] - filters.shape[3] + 1
    output_images = np.zeros((input_images.shape[0], output_height,
                              output_width, filters.shape[0]))
    for image_index in range(len(input_images)):
        for output_channel in range(filters.shape[0]):
            output_slice = np.zeros((output_height, output_width))
            for input_channel in range(filters.shape[1]):
                t = signal.convolve2d(
                        input_images[image_index, :, :, input_channel],
                        np.fliplr(np.flipud(filters[output_channel, input_channel])),
                        mode="valid")
                output_slice = output_slice + t
            output_images[image_index, :, :, output_channel] = \
                    output_slice + bias[output_channel]
    return output_images


def max_pool_slice(x):
    """Return maximum in groups of 2x2 for a N,h,w slice"""
    N,h,w = x.shape
    x = x.reshape(N,h/2,2,w/2,2).swapaxes(2,3).reshape(N,h/2,w/2,4)
    return np.amax(x,axis=3)


def max_pool(images):
    shape = (images.shape[0], images.shape[1] / 2, images.shape[2] / 2,
             images.shape[3])
    output = np.zeros(shape)
    for c in range(images.shape[3]):
        output[:, :, :, c] = max_pool_slice(images[:, :, :, c])
    return output


def load_filters(filename, shape):
    with open(filename, "r") as f:
        s = f.read().strip()
    data = [[float(x) for x in line.strip().split(" ")]
            for line in s.split("\n")]
    return np.reshape(data, shape)


def load_bias(filename):
    with open(filename, "r") as f:
        s = f.read().strip()
    return np.array([float(x) for x in s.split("\n")])


def load_mnist(image_filename, label_filename):
    with open(label_filename, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(image_filename, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8)
        img = img.reshape(len(lbl), rows, cols, 1) / 255.0
    return img, lbl


def main():
    images, labels = load_mnist("build/mnist/t10k-images-idx3-ubyte",
                                "build/mnist/t10k-labels-idx1-ubyte")
    filters = [load_filters("test_data/lenet/weights/conv0_kernels.txt",
                            (20, 1, 5, 5)),
               load_filters("test_data/lenet/weights/conv2_kernels.txt",
                            (50, 20, 5, 5))]
    biases = [load_bias("test_data/lenet/weights/conv0_bias.txt"),
              load_bias("test_data/lenet/weights/conv2_bias.txt")]
    x = images[0:10, :, :, :]
    x = np.maximum(convolve(x, filters[0], biases[0]), 0)
    x = max_pool(x)
    x = np.maximum(convolve(x, filters[1], biases[1]), 0)
    x = max_pool(x)
    assert x.shape == (10, 4, 4, 50)
    data = x.reshape(10, -1)
    with open("test_data/lenet/output.txt", "w") as f:
        for row in data:
            f.write(" ".join(str(x) for x in row) + "\n")


if __name__ == "__main__":
    main()

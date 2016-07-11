#include <iostream>
#include "mnist.h"
#include "lenet.h"

int main() {
    float *x, *y;
    int N;
    std::vector<std::vector<double> > images;
    std::vector<int> labels;

    read_mnist_images(images, "./mnist/t10k-images-idx3-ubyte");
    read_mnist_labels(labels, "./mnist/t10k-labels-idx1-ubyte");

    N = 1;
    x = new float[N * 784];

    for (int t = 0 ; t < N ; t++) {
        for (int i = 0 ; i < 784 ; i++) {
            x[i + t * 784] = images[0][i];
        }
    }

    lenet(N, x, y);

    return 0;
}

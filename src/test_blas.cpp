#include "feedforward.h"
#include <iostream>

void print_matrix(int m, int n, double *mat) {
    for (int i = 0 ; i < m ; i++) {
        for (int j = 0 ; j < n ; j++) {
            std::cout << mat[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    int m = 3;
    int n = 2;
    int k = 4;

    layer_t layer;
    layer.out = k;
    layer.in = n;

    // alloc memory
    double *mat = new double[m * n];
    layer.weights = new double[n * k];
    layer.bias = new double[k];

    for (int i = 0 ; i < m ; i++) {
        for (int j = 0 ; j < n ; j++) {
            mat[i * n + j] = rand_double();
        }
    }

    // initialize weights
    for (int i = 0 ; i < n ; i++) {
        for (int j = 0 ; j < k ; j++) {
            layer.weights[i * k + j] = rand_double();
        }
    }

    for (int i = 0 ; i < k ; i++) {
        layer.bias[i] = rand_double();
    }

    double *ret = feed_forward(m, mat, layer);
    std::cout << "mat = " << std::endl;
    print_matrix(m, n, mat);
    std::cout << "weights = " << std::endl;;
    print_matrix(n, k, layer.weights);
    std::cout << "bias = " << std::endl;
    print_matrix(1, k, layer.bias);
    std::cout << "ret = " << std::endl;
    print_matrix(m, k, ret);
    return 0;
}


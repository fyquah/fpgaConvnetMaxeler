#include "feedforward.h"
#include <fstream>

layer_t layers[N_LAYERS];

float rand_float() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX); 
}

void fully_connected_layers_init (layer_t *layers) {
    layers[0].in = 800;
    layers[0].out = 500;
    layers[0].weights = new float[layers[0].in * layers[0].out];
    layers[0].bias = new float[layers[0].out];

    std::ifstream fin("../lenet_params/ip1_weights.txt");
    for (int i = 0 ; i < layers[0].in ; i++) {
        for (int j = 0 ; j < layers[0].out ; j++) {
            fin >> layers[0].weights[i * layers[0].out + j];
        }
    }
    fin.close();

    fin.open("../lenet_params/ip1_bias.txt");
    for (int i = 0 ; i < layers[0].out ; i++) {
        fin >> layers[0].bias[i];
    }
    fin.close();

    layers[1].in = 500;
    layers[1].out = 10;
    layers[1].weights = new float[layers[1].in * layers[1].out];
    layers[1].bias = new float[layers[1].out];

    fin.open("../lenet_params/ip2_weights.txt");
    for (int i = 0 ; i < layers[1].in ; i++) {
        for (int j = 0 ; j < layers[1].out ; j++) {
            fin >> layers[1].weights[i * layers[1].out + j];
        }
    }
    fin.close();

    fin.open("../lenet_params/ip2_bias.txt");
    for (int i = 0 ; i < layers[1].out ; i++) {
        fin >> layers[1].bias[i];
    }
    fin.close();
}


float* feed_forward (const int m, float * mat, layer_t layer) {
    /*
     * mat is m * layer.in
     * return value is m * layer.out
     * this performs a matrix multiplication of mat * layer
     */
    float *ret = new float[m * layer.out];
    for (int i = 0 ; i < m ; i++) {
        for (int j = 0 ; j < layer.out ; j++) {
            float x = 0;
            for (int k = 0 ; k < layer.in ; k++) {
                x += mat[i * layer.in + k] * layer.weights[k * layer.out + j];
            }
            ret[i * layer.out + j] = x + layer.bias[j];
        }
    }

    return ret;
}

/*
 * Matrix multiplication in openBlas
 * Don't care for now
 */

/*
float* feed_forward(const int m, float *mat, layer_t layer) {
    float *A = mat;
    float *B = layer.weights;
    float *C = new float[m * layer.out];

    for (int i = 0 ; i < m ; i++) {
        for (int j = 0 ; j < layer.out ; j++) {
            C[i * layer.out + j] = layer.bias[j];
        }
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            m, layer.out, layer.in, 1, A, m, B, layer.in, 1, C, m);

    return C;
}
*/


int* get_row_max_index(const int m, const int n, float *mat) {
    int* ret = new int[m];

    for (int i = 0 ; i < m ; i++) {
        float best = mat[i * n];
        int best_index = 0;

        for (int j = 1; j < n ; j++) {
            if (mat[i * n + j] > best) {
                best_index = j;
                best = mat[i * n + j];
            }
        }

        ret[i] = best_index;
    }

    return ret;
}

float* softmax(const int m, const int n, float *mat) {
    float *ret = new float[m * n];

    for (int i = 0 ; i < m ; i++) {
        for (int j = 0 ; j < n ; j++) {
            ret[i * n + j] = exp(mat[i * n + j]);
        }
    }

    for (int i = 0 ; i < m ; i++) {
        float total = 0;
        for (int j = 0 ; j < n ; j++) {
            total += ret[i * n + j];
        }

        for (int j = 0 ; j < n ; j++) {
            ret[i * n + j] /= total;
        }
    }

    return ret;
}

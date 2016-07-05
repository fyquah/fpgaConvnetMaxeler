#include "feedforward.h"

layer_t layers[N_LAYERS];

double rand_double() {
    return static_cast <double> (rand()) / static_cast <double> (RAND_MAX); 
}

void fully_connected_layers_init () {
    for (int i = 0 ; i < N_LAYERS ; i++) {
        layers[i].in = 10;
        layers[i].out = 30;
        int total = layers[i].in * layers[i].out;

        for (int j = 0 ; j < total ; j++) {
            layers[i].weights[j] = rand_double();
        }
    }
}

#ifdef __SIM__

double* feed_forward (const int m, double * mat, layer_t layer) {
    /*
     * mat is m * layer.in
     * return value is m * layer.out
     * this performs a matrix multiplication of mat * layer
     */
    double *ret = new double[m * layer.out];
    for (int i = 0 ; i < m ; i++) {
        for (int j = 0 ; j < layer.out ; j++) {
            double x = 0;
            for (int k = 0 ; k < layer.in ; k++) {
                x += mat[i * layer.in + k] * layer.weights[k * layer.out + j];
            }
            ret[i * layer.out + j] = x + layer.bias[j];

        }
    }

    return ret;
}

#else

double* feed_forward(const int m, double *mat, layer_t layer) {
    double *A = mat;
    double *B = layer.mat;
    double *C = new double[m * layer.out];

    for (int i = 0 ; i < m ; i++) {
        for (int j = 0 ; j < layer.out ; j++) {
            C[i * layer.out + j] = layer.bias[j];
        }
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            m, layer.out, layer.in, 1, A, m, B, layer.in, 1, C, m);

    return C;
}

#endif


int* get_row_max_index(const int m, const int n, double *mat) {
    int* ret = new int[m];

    for (int i = 0 ; i < m ; i++) {
        double best = mat[i * n];
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

double* softmax(const int m, const int n, double *mat) {
    double *ret = new double[m * n];

    for (int i = 0 ; i < m ; i++) {
        for (int j = 0 ; j < n ; j++) {
            ret[i * n + j] = exp(mat[i * n + j]);
        }
    }

    for (int i = 0 ; i < m ; i++) {
        double total = 0;
        for (int j = 0 ; j < n ; j++) {
            total += ret[i * n + j];
        }

        for (int j = 0 ; j < n ; j++) {
            ret[i * n + j] /= total;
        }
    }

    return ret;
}

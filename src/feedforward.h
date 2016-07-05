#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include <cstdlib>
#include <cmath>

struct layer_t {
    int out;
    int in;
    double *weights;
    double *bias;
};

const int N_LAYERS = 3;
extern layer_t layers[N_LAYERS];

double rand_double();
void fully_connected_layers_init();
double* feed_forward(const int m, double *mat, layer_t layer);
double* softmax(const int m, const int k, double *mat);
int* get_row_max_index(const int m, const int n, double *mat);

#endif


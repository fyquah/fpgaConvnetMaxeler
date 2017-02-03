#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include <cstdlib>
#include <cmath>
#include <iostream>

struct layer_t {
    int out;
    int in;
    float *weights;
    float *bias;
};

const int N_LAYERS = 2;

float rand_float();
void fully_connected_layers_init(layer_t*);
float* feed_forward(const int m, float *mat, layer_t layer);
float* softmax(const int m, const int k, float *mat);
int* get_row_max_index(const int m, const int n, float *mat);

#endif


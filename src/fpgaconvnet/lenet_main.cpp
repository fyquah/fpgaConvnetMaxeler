#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstring>

#include "mnist.h"
#include "lenet.h"
#include "feedforward.h"
#include "convnet.h"


#ifdef __SIM__
    static const unsigned N = 3;
#else
    static const unsigned N = 10000;
#endif
static const int CONV_IN_SIZE = 784;
static const int CONV_OUT_SIZE = 800;


template<typename T>
static T max(T a, T b) {
    return a > b ? a : b ;
}

void inplace_relu(int N, float *x) {
    for (int i = 0 ; i < N ; i++) {
        x[i] = max(0.0f, x[i]);
    }
}

float *reshape_lenet_conv_out(int N, float *a) {
    int height = 4;
    int width = 4;
    int channels = 50;
    float* ret = new float[N * height * width * channels];
    // a is now  N * height * width * channels
    // we want to make it N * channels * height * width

    for (int t = 0 ; t < N ; t++) {
        for (int c = 0 ; c < channels ; c++) {
            for (int i = 0 ; i < height ; i++) {
                for (int j = 0 ; j < width ; j++) {
                    int x = t * (height * width * channels) + c * (height * width) + i * width + j;
                    int y = t * (height * width * channels) + i * (width * channels) + j * channels + c;
                    ret[x] = a[y];
                }
            }
        }
    }

    return ret;
}


void report_conv_performance(timeval t_begin, timeval t_end)
{
    double begin = double(t_begin.tv_sec) * 1000000 + double(t_begin.tv_usec);
    double end = double(t_end.tv_sec) * 1000000 + double(t_end.tv_usec);
    double delta = end - begin;
    double throughput = double(N) / delta * 1000000;

    std::cout << "Time taken for " << N << " feature extractions  = ";
    std::cout << delta << std::endl;
    std::cout << "Throughput (images per second) = "
              << throughput << std::endl;
    std::cout << "GOps = " << throughput * 0.0038 << std::endl;
}

void verify_output(float *conv_out, std::string filename)
{
    std::ifstream fin(filename.c_str());
    uint32_t total_pixels = 0;
    float total_error = 0.0;

    for (uint32_t i = 0 ; i < std::min(N, 10u) ; i++) {
        for (uint32_t j = 0 ; j < CONV_OUT_SIZE; j++) {
            float expected;
            float obtained = conv_out[CONV_OUT_SIZE * i + j];
            fin >> expected;
            std::cout << "Expected = " << expected
                    << " obtained = " << obtained << std::endl;
            total_error += std::abs(obtained  - expected);
            total_pixels += 1;

            if (std::abs(obtained - expected) > 0.01) {
                std::cout << "Error > 0.01 while verifying output!" << std::endl;
            }
        }
    }
    std::cout << "Average pixel error = " << float(total_error) / float(total_pixels) << std::endl;
    fin.close();
}


unsigned ceil_div(unsigned a, unsigned b)
{
    if (a % b == 0) {
        return a / b;
    } else {
        return a / b + 1;
    }
}


void weights_copy(
        double *dest_base,
        double *src_base,
        const convnet::conv_layer_t & layer
)
{
    const int conv_ff = layer.conv_folding_factor;
    const int kernel_dim = layer.kernel_size;
    const int worker_factor = layer.worker_factor;

    for (int i = 0 ; i < worker_factor ; i++) {
        int total = ((layer.num_inputs / worker_factor)
                * layer.kernel_size * layer.kernel_size
                * layer.num_outputs);
        double *dest = dest_base + (i * total);
        double *src = src_base + (i * layer.kernel_size * layer.kernel_size);

        for (int w = 0; w < layer.num_inputs / worker_factor; w++) {
            const int worker_iter = w;  // the w-th channel that the worker's handling.
            std::cout << "/**/ worker_iter = " << worker_iter << std::endl;

            for (int channel = 0 ; channel < layer.num_outputs ; channel++) {
                const int src_offset =
                        (channel * layer.num_inputs + worker_iter * worker_factor)
                        * kernel_dim * kernel_dim;
                const int dest_offset =
                        ((worker_iter * (layer.num_outputs / conv_ff))
                         + (channel % conv_ff) * (layer.num_outputs / conv_ff) * (layer.num_inputs / worker_factor)
                         + (channel / conv_ff))
                        * kernel_dim * kernel_dim;
                std::cout << "src_offset = " 
                        << src_offset / (kernel_dim * kernel_dim)
                        << " -> "
                        << dest_offset / (kernel_dim * kernel_dim)
                        << std::endl;
                std::memcpy(
                        dest + dest_offset,
                        src + src_offset,
                        sizeof(double) * kernel_dim * kernel_dim);
            }
        }
    }
}


static void set_layer_weights(
        max_actions_t *action,
        convnet::conv_layer_t layer,
        double *weights,
        double *bias)
{
    char buffer[100];

    for (int worker = 0 ; worker < layer.worker_factor ; worker++) {
        const uint64_t worker_size =
                convnet::total_kernel_weights(layer) / layer.worker_factor;

        sprintf(buffer, "kernels_%d_worker_%d", layer.id, worker);

        for (int i = 0 ; i < worker_size ; i++) {
            max_set_param_array_double(
                    action,
                    buffer,
                    weights[worker_size * worker + i],
                    i);
        }
    }

    for (int i = 0 ; i < layer.num_outputs ; i++) {
        sprintf(buffer, "bias_%d", layer.id);
        max_set_param_array_double(action, buffer, bias[i], i);
    }
}


void run_feature_extraction(const float *images, float *conv_out)
{
    max_file_t *max_file = lenet_init();
    max_engine_t *dfe = max_load(max_file, "*");
    timeval t_begin, t_end;
    convnet::conv_layer_t conv0_layer =
            {.id = 0, .kernel_size = 5, .num_inputs = 1, .num_outputs = 20,
             .conv_folding_factor = 5, .worker_factor = 1};
    convnet::conv_layer_t conv2_layer =
            {.id = 2, .kernel_size = 5, .num_inputs = 20, .num_outputs = 50,
             .conv_folding_factor = 5, .worker_factor = 5};
    double *conv0_kernels = new double[convnet::total_kernel_weights(conv0_layer)];
    double *conv0_bias = new double[conv0_layer.num_outputs];
    double *conv2_kernels = new double[convnet::total_kernel_weights(conv2_layer)];
    double *conv2_bias = new double[conv2_layer.num_outputs];
    double *layer_0_worker_weights = new double[convnet::total_kernel_weights(conv0_layer)];
    double *layer_2_worker_weights = new double[convnet::total_kernel_weights(conv2_layer)];
    const uint64_t address_images = 0;
    const uint64_t address_features= N * 784 * sizeof(float);

    convnet::load_kernels_from_file(
            std::string("../test_data/lenet/weights/conv0_kernels.txt"),
            conv0_layer, conv0_kernels);
    convnet::load_bias_from_file(
            std::string("../test_data/lenet/weights/conv0_bias.txt"),
            conv0_layer, conv0_bias);
    convnet::load_kernels_from_file(
            std::string("../test_data/lenet/weights/conv2_kernels.txt"),
            conv2_layer, conv2_kernels);
    convnet::load_bias_from_file(
            std::string("../test_data/lenet/weights/conv2_bias.txt"),
            conv2_layer, conv2_bias);

    /*
     * Reallign the kernels such that we have:
     * [kernel[c][0], kernel[c][convFactor], kernel[c][2 * convFactor] ...,
     *  kernel[c][1], kernel[c][convFactor + 1], kernel[c][2 * convFactor + 2], ...,
     *  ]
     *  akin reshape(kernels, (convFactors, -1)).T
     */
    weights_copy(layer_0_worker_weights, conv0_kernels, conv0_layer);
    weights_copy(layer_2_worker_weights, conv2_kernels, conv2_layer);
    __sync_synchronize();

    /* Calling the initializtion */
    std::cout << "Initializing net weights in DFE." << std::endl;
    max_actions_t *memory_action = max_actions_init(max_file, "init_convnet");
    set_layer_weights(
            memory_action, conv0_layer, layer_0_worker_weights, conv0_bias);
    set_layer_weights(
            memory_action, conv2_layer, layer_2_worker_weights, conv2_bias);
    max_run(dfe, memory_action);
    max_actions_free(memory_action);

    std::cout << "Copying sample data to off-chip memory." << std::endl;
    lenet_load_data(address_images, N * 784 * sizeof(float), images);

    std::cout << "Running Feature Extraction ... " << std::endl;
    lenet_run_convnet_actions_t run_action = {
            .param_N = N,
            .param_address_features = address_features,
            .param_address_images = address_images};
    gettimeofday(&t_begin, NULL);
    lenet_run_convnet_run(dfe, &run_action);
    gettimeofday(&t_end, NULL);
    max_unload(dfe);
    delete[] conv0_kernels;
    delete[] conv0_bias;
    delete[] conv2_kernels;
    delete[] conv2_bias;
    delete[] layer_0_worker_weights;
    delete[] layer_2_worker_weights;

    std::cout << "Completed feature extraction!" << std::endl;
    report_conv_performance(t_begin, t_end);


    std::cout << "Copying features from off-chip memory." << std::endl;
    lenet_load_data(address_features, N * 800 * sizeof(float), conv_out);

    verify_output(conv_out, "../test_data/lenet/output.txt");
}

int main()
{
    float *x = new float[N * CONV_IN_SIZE];
    float *conv_out = new float[N * CONV_OUT_SIZE];
    std::vector<std::vector<double> > images;
    std::vector<int> labels;
    layer_t layers[N_LAYERS];

    try {
        std::cout << "Reading images ..." << std::endl;
        read_mnist_images(images, "mnist/t10k-images-idx3-ubyte");
        read_mnist_labels(labels, "mnist/t10k-labels-idx1-ubyte");
        for (unsigned i = 0 ; i < N ; i++) {
            for (unsigned j = 0 ; j < 784 ; j++) {
                x[i * 784 + j] = (float) images[i][j];
            }
        } 
        run_feature_extraction(x, conv_out);
        return 0;

        std::cout << "Initializing fully connected weights ..." << std::endl;
        fully_connected_layers_init(layers);
	std::cout << "Computing feedforward layers ..." << std::endl;
        float *a = feed_forward(N, reshape_lenet_conv_out(N, conv_out), layers[0]);
        inplace_relu(N * layers[0].out, a);
        float *b = feed_forward(N, a, layers[1]);
        float *c = softmax(N, 10, b);

        int *klasses = get_row_max_index(N, 10, c);
        int total_correct = 0;
        for (unsigned i = 0 ; i < N ; i++) {
            if (klasses[i] == labels[i]) {
                total_correct++;
            }
        }
        double accuracy = double(total_correct) / double(N);

        std::cout << "Prediction accuracy = " << accuracy << std::endl;

    } catch (const std::string & s) {
        std::cerr << s << std::endl;
    }

    return 0;
}

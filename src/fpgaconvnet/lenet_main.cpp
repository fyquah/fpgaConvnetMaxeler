#include <iostream>
#include <fstream>

#include "mnist.h"
#include "lenet.h"
#include "feedforward.h"
#include "convnet.h"


#ifdef __SIM__
    const unsigned N = 10;
#else
    const unsigned N = 10000;
#endif


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


void run_feature_extraction(const double *images, double *conv_out)
{
    max_file_t *max_file = lenet_init();
    max_engine_t *dfe = max_load(max_file, "*");
    lenet_actions_t action;
    timeval t_begin, t_end;
    const convnet::conv_layer_t conv0_layer =
            {.kernel_size = 5, .num_inputs = 1, .num_outputs = 20};
    const convnet::conv_layer_t conv2_layer =
            {.kernel_size = 5, .num_inputs = 20, .num_outputs = 50};
    double *conv0_kernels = new double[total_kernel_weights(conv_layer0)];
    double *conv0_bias = new double[conv_layer0.num_outputs];
    double *conv2_kernels = new double[total_kernel_weights(conv_layer2)];
    double *conv2_bias = new double[conv_layer2.num_outputs];

    convnet::load_conv_kernels(conv0_layer, conv0_kernels);
    convnet::load_conv_bias(conv0_layer, conv0_bias);
    convnet::load_conv_kernels(conv2_layer, conv2_kernels);
    convnet::load_conv_bias(conv2_layer, conv2_bias);
    action.inmem_ConvolutionScheduler_0_mappedRom = conv0_kernels;
    action.inmem_ConvolutionAccumulator_0_bias_layer_0 = conv0_bias;
    action.inmem_ConvolutionScheduler_1_mappedRom = conv2_kernels;
    action.inmem_ConvolutionAccumulator_1_bias_layer_2 = conv2_bias;
    action.param_N = N;
    action.instream_x = x;
    action.outstream_y = conv_out;

    std::cout << "Running Feature Extraction ... " << std::endl;
    gettimeofday(&t_begin, NULL);
    lenet_run(dfe, &action);
    gettimeofday(&t_end, NULL);
    max_unload(dfe);
    free(conv0_kernels);
    free(conv0_bias);
    free(conv2_kernels);
    free(conv2_bias);

    std::cout << "Completed feature extraction!" << std::endl;
    report_conv_performance(t_begin, t_end);
}

int main() {
    float *x = new float[N * 784];
    float *conv_out = new float[N * 800];
    std::vector<std::vector<double> > images;
    std::vector<int> labels;
    layer_t layers[N_LAYERS];

    try {
        std::cout << "Reading images ..." << std::endl;
        read_mnist_images(images, "./mnist/t10k-images-idx3-ubyte");
        read_mnist_labels(labels, "./mnist/t10k-labels-idx1-ubyte");
        for (int i = 0 ; i < N ; i++) {
            for (int j = 0 ; j < 784 ; j++) {
                x[i * 784 + j] = (float) images[i][j];
            }
        } 
        run_feature_extraction(x, conv_out);

        std::cout << "Initializing fully connected weights ..." << std::endl;
        fully_connected_layers_init(layers);
	std::cout << "Computing feedforward layers ..." << std::endl;
        float *a = feed_forward(N, reshape_lenet_conv_out(N, conv_out), layers[0]);
        inplace_relu(N * layers[0].out, a);
        float *b = feed_forward(N, a, layers[1]);
        float *c = softmax(N, 10, b);

        int *klasses = get_row_max_index(N, 10, c);
        int total_correct = 0;
        for (int i = 0 ; i < N ; i++) {
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

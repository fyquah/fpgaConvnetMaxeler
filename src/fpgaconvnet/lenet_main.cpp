#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstring>

#include "fpgaconvnet/protos/parameters.pb.h"
#include "fpgaconvnet/mnist.h"
#include "fpgaconvnet/convnet.h"
#include "fpgaconvnet/feedforward.h"

#include "lenet.h"


#ifdef __SIM__
    static const uint64_t N = 6;
#else
    static const uint64_t N = 10000;
#endif
static const uint64_t CONV_IN_SIZE = 784;
static const uint64_t CONV_OUT_SIZE = 800;


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


unsigned ceil_div(unsigned a, unsigned b)
{
    if (a % b == 0) {
        return a / b;
    } else {
        return a / b + 1;
    }
}


std::vector<float> run_feature_extraction(
        const fpgaconvnet::protos::Network & network_parameters,
        const std::vector<float> & images
)
{
    std::vector<std::string> filenames = {
            "../test_data/lenet/weights/conv0_kernels.txt",
            "../test_data/lenet/weights/conv0_bias.txt",
            "../test_data/lenet/weights/conv2_kernels.txt",
            "../test_data/lenet/weights/conv2_bias.txt"};
    max_file_t *max_file = lenet_init();
    fpgaconvnet::Convnet convnet(network_parameters, max_file, "");

    convnet.load_weights_from_files(filenames);
    convnet.max_init_weights();
    convnet.max_load_input_data(images, N);
    convnet.max_run_inference(N);
    std::vector<float> extracted_features = convnet.max_retrieve_features(N);

    fpgaconvnet::verify_conv_output(
            network_parameters,
            N,
            &extracted_features[0],
            "../test_data/lenet/output.txt");

    return extracted_features;
}


int main(int argc, char **argv)
{
    std::vector<std::vector<double> > images;
    std::vector<int> labels;
    std::vector<float> pixel_stream;
    layer_t layers[N_LAYERS];

    try {
        if (argc < 2) {
            std::cout << "Missing network descriptor" << std::endl;
            return 1;
        }

        std::cout << "Loading netowork parameters from " << argv[1] << std::endl;
        fpgaconvnet::protos::Network network_parameters =
                fpgaconvnet::load_network_proto(argv[1]);

        std::cout << "Reading images ..." << std::endl;
        read_mnist_images(images, "mnist/t10k-images-idx3-ubyte");
        read_mnist_labels(labels, "mnist/t10k-labels-idx1-ubyte");
        for (unsigned i = 0 ; i < N ; i++) {
            for (unsigned j = 0 ; j < CONV_IN_SIZE ; j++) {
                pixel_stream.push_back(images[i][j]);
            }
        } 

        std::vector<float> conv_out = run_feature_extraction(
                network_parameters, pixel_stream);
        return 0;

        std::cout << "Initializing fully connected weights ..." << std::endl;
        fully_connected_layers_init(layers);
	std::cout << "Computing feedforward layers ..." << std::endl;
        float *a = feed_forward(
                N, reshape_lenet_conv_out(N, &conv_out[0]), layers[0]);
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

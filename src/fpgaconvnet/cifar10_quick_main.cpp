#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstring>

#include "fpgaconvnet/protos/parameters.pb.h"
#include "fpgaconvnet/mnist.h"
#include "fpgaconvnet/convnet.h"
#include "fpgaconvnet/feedforward.h"

#include "cifar10_quick.h"


#ifdef __SIM__
    static const uint64_t N = 6;
#else
    static const uint64_t N = 10000;
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


unsigned ceil_div(unsigned a, unsigned b)
{
    if (a % b == 0) {
        return a / b;
    } else {
        return a / b + 1;
    }
}

void load_cifar10(const char *filename, std::vector<float> &images, std::vector<int> &labels)
{
    labels = std::vector<int>(10000);
    images = std::vector<float>(10000 * 1024 * 3);
    std::ifstream fin(filename);

    if (!fin.is_open()) {
        throw fpgaconvnet::Exception("File not found");
    }

    for (unsigned iter = 0 ; iter < N ; iter++) {
        char byte;
        std::vector<float> red(1024);
        std::vector<float> blue(1024);
        std::vector<float> green(1024);

        fin.read(static_cast<char*>(&byte), 1);
        labels.push_back(int(byte));

        for (int i = 0 ; i< 1024 ; i++) {
            fin.read(static_cast<char*>(&byte), 1);
            red[i] = float((unsigned char) byte) / 255.0;
        }

        for (int i = 0 ; i< 1024 ; i++) {
            fin.read(&byte, 1);
            green[i] = float((unsigned char) byte) / 255.0;
        }

        for (int i = 0 ; i< 1024 ; i++) {
            fin.read(&byte, 1);
            blue[i] = float((unsigned char) byte) / 255.0;
        }

        for (int i = 0 ; i < 1024 ; i++) {
            images[(iter * 3 * 1024) + 3 * i] = red[i];
            images[(iter * 3 * 1024) + 3 * i + 1] = green[i];
            images[(iter * 3 * 1024) + 3 * i + 2] = blue[i];
        }
    }

    fin.close();
}


std::vector<float> run_feature_extraction(
        const fpgaconvnet::protos::Network & network_parameters,
        const std::vector<float> & images
)
{
    std::vector<std::string> filenames = {
            "../test_data/cifar10_quick/weights/conv0_weights",
            "../test_data/cifar10_quick/weights/conv0_bias",
            "../test_data/cifar10_quick/weights/conv2_weights",
            "../test_data/cifar10_quick/weights/conv2_bias",
            "../test_data/cifar10_quick/weights/conv4_weights",
            "../test_data/cifar10_quick/weights/conv4_bias" };
    max_file_t *max_file = cifar10_quick_init();
    std::vector<float> extracted_features;
    fpgaconvnet::Convnet convnet(network_parameters, max_file, "");

    std::cout << "Loading weights from file." << std::endl;
    convnet.load_weights_from_files(filenames, fpgaconvnet::FORMAT_BINARY);

    // warm up the DFE with the weights.
    std::cout << "Runnin inference" << std::endl;
    extracted_features = convnet.max_run_inference(N, images, false);
    extracted_features = convnet.max_run_inference(N, images, true);
    fpgaconvnet::verify_conv_output(
            network_parameters,
            N,
            &extracted_features[0],
            "../test_data/cifar10_quick/pool3.bin",
            fpgaconvnet::FORMAT_BINARY);

    return extracted_features;
}


int main(int argc, char **argv)
{
    std::vector<float> images;
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
        fpgaconvnet::protos::LayerParameter first_layer = network_parameters.layer(0);
        const uint32_t conv_in_size = 
            first_layer.input_height() * first_layer.input_width()
            * first_layer.num_inputs();

        std::cout << "Reading images ..." << std::endl;
        load_cifar10("cifar-10-batches-bin/data_batch_1.bin", images, labels);

        for (unsigned i = 0 ; i < N ; i++) {
            for (unsigned j = 0 ; j < conv_in_size ; j++) {
                pixel_stream.push_back(images[i * conv_in_size + j]);
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

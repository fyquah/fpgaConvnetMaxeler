#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstring>

#include "fpgaconvnet/protos/parameters.pb.h"
#include "fpgaconvnet/mnist.h"
#include "fpgaconvnet/convnet.h"
#include "fpgaconvnet/feedforward.h"

#include "target_0.h"
#include "target_1.h"


#ifdef __SIM__
    static const uint64_t N = 4;
#else
    static const uint64_t N = 10000;
#endif


template<typename T>
static T max(T a, T b) {
    return a > b ? a : b ;
}

std::vector<float> run_feature_extraction(
        const fpgaconvnet::protos::Network & network_parameters,
        const std::vector<float> & images
)
{
    std::vector<max_file_t*> max_files;

    max_files.push_back(target_0_init());
    max_files.push_back(target_1_init());

    std::vector<std::string> filenames = {
            "../weights/conv0_kernels.txt",
            "../weights/conv0_bias.txt",
            "../weights/conv2_kernels.txt",
            "../weights/conv2_bias.txt"};
    std::vector<float> extracted_features;
    fpgaconvnet::Convnet convnet(network_parameters, max_files, "");

    convnet.load_weights_from_files(filenames, fpgaconvnet::FORMAT_TXT);
    convnet.max_init_weights();

    // warm up the DFE with the weights.
    extracted_features = convnet.max_run_inference(N, images, false);
#ifndef __SIM__
    extracted_features = convnet.max_run_inference(N, images, true);
#endif

    fpgaconvnet::verify_conv_output(
            network_parameters,
            N,
            &extracted_features[0],
            "../test_data/output.txt");

    return extracted_features;
}


void load_cifar10(const char *filename, std::vector<float> &images, std::vector<int> &labels)
{
    labels = std::vector<int>(10000);
    images = std::vector<float>(10000 * 1024 * 3);
    std::ifstream fin(filename);

    if (!fin.is_open()) {
        throw fpgaconvnet::Exception("File not found");
    }

    for (unsigned iter = 0 ; iter < 10000 ; iter++) {
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

std::vector<float> duplicate_vector(const std::vector<float> &v, int count)
{
    std::vector<float> ret(v.size() * count);
    for (int i = 0 ; i < count ; i++) {
        std::memcpy(&ret[i * v.size()], &v[0], sizeof(float) * v.size());
    }
    return ret;
}


int main(int argc, char **argv)
{
    std::vector<int> labels;
    std::vector<std::vector<double> > images;
    std::vector<float> pixel_stream;

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

    std::cout << network_parameters.DebugString() << std::endl;


    std::cout << "Reading images ..." << std::endl;
    read_mnist_images(images, "mnist/t10k-images-idx3-ubyte");
    read_mnist_labels(labels, "mnist/t10k-labels-idx1-ubyte");
    for (unsigned i = 0 ; i < N ; i++) {
        for (unsigned j = 0 ; j < conv_in_size ; j++) {
            pixel_stream.push_back(images[i][j]);
        }
    } 

    std::vector<float> conv_out = run_feature_extraction(network_parameters, pixel_stream);
    return 0;
}

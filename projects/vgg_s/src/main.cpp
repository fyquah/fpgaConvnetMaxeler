#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstring>

#include "fpgaconvnet/protos/parameters.pb.h"
#include "fpgaconvnet/mnist.h"
#include "fpgaconvnet/convnet.h"
#include "fpgaconvnet/feedforward.h"

#include "vgg_s.h"


#ifdef __SIM__
    static const uint64_t N = 2;
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
    std::vector<std::string> filenames = {
            "../weights/conv0_weights.bin",
            "../weights/conv0_bias.bin",
            "../weights/conv3_weights.bin",
            "../weights/conv3_bias.bin",
            "../weights/conv5_weights.bin",
            "../weights/conv5_bias.bin"
            "../weights/conv6_weights.bin",
            "../weights/conv6_bias.bin"
            "../weights/conv7_weights.bin",
            "../weights/conv7_bias.bin"
    };
    max_file_t *max_file = vgg_s_init();
    std::vector<float> extracted_features;
    fpgaconvnet::Convnet convnet(network_parameters, max_file, "");

    convnet.load_weights_from_files(filenames, fpgaconvnet::FORMAT_BINARY);
    convnet.max_init_weights();

    /* warm up the DFE with the weights. */
    extracted_features = convnet.max_run_inference(N, images, false);
    extracted_features = convnet.max_run_inference(N, images, false);

    fpgaconvnet::verify_conv_output(
            network_parameters,
            N,
            &extracted_features[0],
            "../test_data/output.bin",
            fpgaconvnet::FORMAT_BINARY);

    return extracted_features;
}


static void load_data(std::string filename, std::vector<float> & images)
{
    const int image_size = 224 * 224;

    images = std::vector<float>(100 * image_size * 3);
    std::ifstream fin(filename);

    if (!fin.is_open()) {
        throw fpgaconvnet::Exception("File not found");
    }

    for (unsigned iter = 0 ; iter < N ; iter++) {
        char byte;
        std::vector<float> red(image_size);
        std::vector<float> blue(image_size);
        std::vector<float> green(image_size);

        fin.read(static_cast<char*>(&byte), 1);

        for (int i = 0 ; i< image_size ; i++) {
            fin.read(static_cast<char*>(&byte), 1);
            red[i] = float((unsigned char) byte) / 255.0;
        }

        for (int i = 0 ; i< image_size ; i++) {
            fin.read(&byte, 1);
            green[i] = float((unsigned char) byte) / 255.0;
        }

        for (int i = 0 ; i< image_size ; i++) {
            fin.read(&byte, 1);
            blue[i] = float((unsigned char) byte) / 255.0;
        }

        for (int i = 0 ; i < image_size ; i++) {
            images[(iter * 3 * image_size) + 3 * i] = red[i];
            images[(iter * 3 * image_size) + 3 * i + 1] = green[i];
            images[(iter * 3 * image_size) + 3 * i + 2] = blue[i];
        }
    }

    fin.close();
}


int main(int argc, char **argv)
{
    std::vector<int> labels;
    std::vector<float> pixel_stream;

    if (argc < 2) {
	std::cout << "Missing network descriptor" << std::endl;
	return 1;
    }

    std::cout << "Loading netowork parameters from " << argv[1] << std::endl;
    fpgaconvnet::protos::Network network_parameters =
	    fpgaconvnet::load_network_proto(argv[1]);

    std::cout << "Reading images ..." << std::endl;
    load_data("../test_data/input.bin", pixel_stream);

    std::vector<float> conv_out = run_feature_extraction(
	    network_parameters, pixel_stream);
    return 0;
}

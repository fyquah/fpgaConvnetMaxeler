#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstring>

#include "fpgaconvnet/protos/parameters.pb.h"
#include "fpgaconvnet/mnist.h"
#include "fpgaconvnet/convnet.h"
#include "fpgaconvnet/feedforward.h"

#include "large_layer_test.h"


#ifdef __SIM__
    static const uint64_t N = 6;
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
    max_file_t *max_file = large_layer_test_init();
    std::vector<float> extracted_features;
    fpgaconvnet::Convnet convnet(network_parameters, max_file, "");

    std::vector<std::string> filenames = {
        "../test_data/weights.bin",
        "../test_data/bias.bin",
    };
    convnet.load_weights_from_files(filenames, fpgaconvnet::FORMAT_BINARY);
    convnet.randomize_weights();
    convnet.max_init_weights();

    /* warm up the DFE with the weights. */
    extracted_features = convnet.max_run_inference(N, images, false);
    extracted_features = convnet.max_run_inference(N, images, false);

    fpgaconvnet::verify_conv_output(
            network_parameters,
            N,
            &extracted_features[0],
            "../test_data/data_output.bin",
            fpgaconvnet::FORMAT_BINARY);
    return extracted_features;
}


int main(int argc, char **argv)
{
    if (argc < 2) {
	std::cout << "Missing network descriptor" << std::endl;
	return 1;
    }

    std::cout << "Loading netowork parameters from " << argv[1] << std::endl;
    fpgaconvnet::protos::Network network_parameters =
	    fpgaconvnet::load_network_proto(argv[1]);

    std::cout << "Reading images ..." << std::endl;
    const int input_array_size = 100 * 16 * 16 * 512;
    std::vector<float> pixel_stream(input_array_size, 0);
    fpgaconvnet::load_float_array_from_binary_file(
            "../test_data/data_input.bin",
            input_array_size,
            (float*) &pixel_stream[0]);


    if (N < 100) {
        pixel_stream.resize(N * 16 * 16 * 512);

    } else if (N > 100) {
        std::vector<float> ret;
        for (int i = 0 ; i < (N / 100) ; i++) {
            std::memcpy(
                    &ret[i * pixel_stream.size()],
                    &pixel_stream[0],
                    sizeof(float) * pixel_stream.size());
        }
        pixel_stream = ret;

    }


    std::vector<float> conv_out = run_feature_extraction(
	    network_parameters, pixel_stream);
    return 0;
}

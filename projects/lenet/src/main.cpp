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


template<typename T>
static T max(T a, T b) {
    return a > b ? a : b ;
}

std::vector<float> run_feature_extraction(
        const fpgaconvnet::protos::Network & network_parameters,
        const std::vector<float> & images
)
{
    max_file_t *max_file = lenet_init();
    std::vector<float> extracted_features;
    fpgaconvnet::Convnet convnet(network_parameters, max_file, "");

    /* TODO: load the weights from somewhere. */
    convnet.randomize_weights();

    /* warm up the DFE with the weights. */
    extracted_features = convnet.max_run_inference(N, images, false);
    extracted_features = convnet.max_run_inference(N, images, false);

    /* TODO: Verify the output is correct. You can use the
     * fpgaconvnet::verify_conv_out function for this.
     */

    return extracted_features;
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

    /* TODO: Code to load images here. Code below is a stub to load random
     *       data as image input.
     */
    std::cout << "Reading images ..." << std::endl;
    for (unsigned i = 0 ; i < N ; i++) {
	for (unsigned j = 0;
		j < fpgaconvnet::calc_conv_in_size(network_parameters);
		j++) {
	    pixel_stream.push_back((float(rand()) / float(RAND_MAX)));
	}
    } 

    std::vector<float> conv_out = run_feature_extraction(
	    network_parameters, pixel_stream);
    return 0;
}

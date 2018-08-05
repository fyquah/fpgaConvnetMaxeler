#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstring>
#include <sstream>

#include "fpgaconvnet/protos/parameters.pb.h"
#include "fpgaconvnet/mnist.h"
#include "fpgaconvnet/convnet.h"
#include "fpgaconvnet/feedforward.h"

#include "targets.h"  // includes all the header file for maxfiles generated.


#ifdef __SIM__
    static const uint64_t N = 3;
#else
    static const uint64_t N = 1000;
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
    std::vector<max_file_t*> max_files = targets_init();

    std::vector<float> extracted_features;
    fpgaconvnet::Convnet convnet(network_parameters, max_files, "");

    /* TODO: load the weights from somewhere. */
    convnet.randomize_weights();
    convnet.max_init_weights();

    /* warm up the DFE with the weights. */
    extracted_features = convnet.max_run_inference(N, images, false);
    extracted_features = convnet.max_run_inference(N, images, true);

    /* TODO: Verify the output is correct. You can use the
     * fpgaconvnet::verify_conv_out function for this.
     */
    {
        const unsigned N = 1;
        std::vector<double> times;
        fpgaconvnet::logging::stdout(fpgaconvnet::logging::INFO)
            << "Measuring latency using " << N << " images" << std::endl;

        for (int i = 0 ; i < 1000 ; i++) {
            double p;
            convnet.max_run_inference(N, images, true, &p);
            times.push_back(p);
        }

        std::stringstream ss;
        ss << "../results/latency_" << N << ".txt";
        fpgaconvnet::dump_latencies(ss.str().c_str(), times);
    }

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

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
    static const uint64_t N = 6;
#else
    static const uint64_t N = 2 * 96 * 100;
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
    std::vector<std::vector<max_file_t*>> max_files = targets_init();

    std::vector<float> extracted_features;
    fpgaconvnet::Convnet convnet(network_parameters, max_files, "");

    std::vector<unsigned> conv_layer_ids = {0, 3, 6, 7, 8};
    std::vector<std::string> filenames;
    for (unsigned i = 0; i < conv_layer_ids.size() ; i++) {
        std::stringstream ss0;
        std::stringstream ss1;

        ss0 << "./testdata/weights/conv" << conv_layer_ids[i] << "_weights";
        ss1 << "./testdata/weights/conv" << conv_layer_ids[i] << "_bias";
        filenames.push_back(ss0.str());
        filenames.push_back(ss1.str());
    }

    convnet.load_weights_from_files(filenames, fpgaconvnet::FORMAT_BINARY);
    convnet.max_init_weights();

    /* warm up the DFE with the weights. */
    extracted_features = convnet.max_run_inference(N, images, true);
    extracted_features = convnet.max_run_inference(N, images, true);

    /* TODO: Verify the output is correct. You can use the
     * fpgaconvnet::verify_conv_out function for this.
     */
    fpgaconvnet::verify_conv_output(
            network_parameters,
            N,
            &extracted_features[0],
            "./testdata/data/output.bin",
            fpgaconvnet::FORMAT_BINARY);

    // this is to measure latency
    std::vector<unsigned> counts;
    counts.push_back(4);
    counts.push_back(8);
    counts.push_back(12);
    counts.push_back(16);
    counts.push_back(20);

    for (unsigned j = 0; j < counts.size() ; j++) {
        const unsigned N = counts[j];
        std::vector<double> times;
        fpgaconvnet::logging::set_level(fpgaconvnet::logging::INFO);
        fpgaconvnet::logging::stdout(fpgaconvnet::logging::INFO)
            << "Measuring latency using " << N << " images" << std::endl;
        fpgaconvnet::logging::set_level(fpgaconvnet::logging::WARNING);

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

    if (argc < 2) {
	std::cout << "Missing network descriptor" << std::endl;
	return 1;
    }

    std::cout << "Loading netowork parameters from " << argv[1] << std::endl;
    fpgaconvnet::protos::Network network_parameters =
	    fpgaconvnet::load_network_proto(argv[1]);

    std::cout << "Reading images ..." << std::endl;
    std::vector<float> pixel_stream =
        load_and_duplicate_float_stream(
                "./testdata/data/input.bin",
                fpgaconvnet::calculation::conv_in_size(network_parameters),
                10,
                N);
    std::vector<float> conv_out = run_feature_extraction(
	    network_parameters, pixel_stream);
    return 0;
}

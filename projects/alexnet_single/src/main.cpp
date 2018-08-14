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
    static const uint64_t N = 96 * 100;
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
    std::vector<std::vector<max_file_t*> > max_files = targets_init();

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
    fpgaconvnet::verify_conv_output(
            network_parameters,
            N,
            &extracted_features[0],
            "./testdata/data/output.bin",
            fpgaconvnet::FORMAT_BINARY);
    return extracted_features;

    /* BEGIN  -- DEBUG Bitstream 0 */
    fpgaconvnet::logging::stdout(fpgaconvnet::logging::INFO)
        << "Checking bitstream 0\n";
    extracted_features = convnet.max_run_inference_with_single_bitstream(
            N, images, 0);
    fpgaconvnet::verify_conv_output(
            network_parameters,
            N,
            &extracted_features[0],
            "./testdata/data/conv7.bin",
            fpgaconvnet::FORMAT_BINARY);
    /* END OF -- DEBUG Bitstream 0 */

    /* BEGIN  -- DEBUG Bitstream 1 */
    std::vector<float> bitstream_1_input =
        load_and_duplicate_float_stream(
                "./testdata/data/conv7.bin",
                fpgaconvnet::calculation::conv_in_size_for_bitstream(
                    network_parameters, 1),
                10,
                N);
    fpgaconvnet::logging::stdout(fpgaconvnet::logging::INFO)
        << "Checking bitstream 1\n";
    extracted_features = convnet.max_run_inference_with_single_bitstream(
            N, bitstream_1_input, 1);
    fpgaconvnet::verify_conv_output(
            network_parameters,
            N,
            &extracted_features[0],
            "./testdata/data/output.bin",
            fpgaconvnet::FORMAT_BINARY);
    /* END OF -- DEBUG Bitstream 1 */

    return extracted_features;
}


int main(int argc, char **argv)
{
    fpgaconvnet::logging::set_level(fpgaconvnet::logging::INFO);

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

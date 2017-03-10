#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstring>

#include "fpgaconvnet/protos/parameters.pb.h"
#include "fpgaconvnet/mnist.h"
#include "fpgaconvnet/convnet.h"
#include "fpgaconvnet/feedforward.h"

#include "test_lrn.h"


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
    max_file_t *max_file = test_lrn_init();
    std::vector<float> extracted_features;
    fpgaconvnet::Convnet convnet(network_parameters, max_file, "");

    /* TODO: load the weights from somewhere. */
    convnet.randomize_weights();

    extracted_features = convnet.max_run_inference(N, images, false);

    /* TODO: Verify the output is correct. You can use the
     * fpgaconvnet::verify_conv_out function for this.
     */
    std::vector<float> expected_output = extracted_features;
    float alpha = network_parameters.layer(0).lrn().alpha();
    float beta = network_parameters.layer(0).lrn().beta();
    int local_size = network_parameters.layer(0).lrn().local_size();
    int num_inputs = network_parameters.layer(0).num_inputs();

    for (int p = 0 ; p < expected_output.size() ; p += num_inputs) {

        for (int c = 0 ; c < network_parameters.layer(0).num_inputs() ; c++) {
            float x = 0.0;

            for (int i = -(local_size / 2) ; i <= (local_size / 2) ; i++) {
                int idx = c + i;

                if (idx < 0) {
                    idx = local_size / 2 - i;

                } else if (idx >= num_inputs) {
                    idx = num_inputs - local_size / 2 - (i + 1);

                }

                x += images[idx] * images[idx];
            }

            x = images[p + c] / std::pow(1.0 + alpha / float(local_size) * x, beta);
            expected_output[p + c] = x;
        }
    }

    std::cout << "size = " << expected_output.size() << ", " << extracted_features.size() << std::endl;
    float err = 0;
    for (int i = 0 ; i < expected_output.size() ; i++) {
        err += std::abs(expected_output[i] - extracted_features[i]);
        std::cout << (i % 128) << " : " << expected_output[i] << ", " << extracted_features[i] << "\n";
    }
    std::cout << "Average error = " << err / expected_output.size() << std::endl;

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
            float x = (float(rand()) / float(RAND_MAX));
	    pixel_stream.push_back((x - 0.5) * 2.0);
	}
    } 

    std::vector<float> conv_out = run_feature_extraction(
	    network_parameters, pixel_stream);
    return 0;
}

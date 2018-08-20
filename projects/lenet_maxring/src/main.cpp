#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstring>

#include "fpgaconvnet/protos/parameters.pb.h"
#include "fpgaconvnet/mnist.h"
#include "fpgaconvnet/convnet.h"
#include "fpgaconvnet/feedforward.h"

#include "targets.h"


#ifdef __SIM__
    static const uint64_t N = 4;
#else
    static const uint64_t N = 1228800;
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

    std::vector<std::string> filenames = {
            "../weights/conv0_kernels.txt",
            "../weights/conv0_bias.txt",
            "../weights/conv2_kernels.txt",
            "../weights/conv2_bias.txt"};
    std::vector<float> extracted_features;

    std::cout << "Constructing fpgaconvnet::Convnet object" << std::endl;
    fpgaconvnet::Convnet convnet(network_parameters, max_files, "*");

    std::cout << "Loading weights to object" << std::endl;
    convnet.load_weights_from_files(filenames, fpgaconvnet::FORMAT_TXT);
    convnet.max_init_weights();

    // warm up the DFE with the weights.
    extracted_features = convnet.max_run_inference(N, images, false);
    extracted_features = convnet.max_run_inference(N, images, true);

    fpgaconvnet::verify_conv_output(
            network_parameters,
            N,
            &extracted_features[0],
            "../test_data/output.txt");

#ifndef __SIM__
    // this is to measure latency
    fpgaconvnet::logging::set_level(fpgaconvnet::logging::WARNING);
    std::vector<double> times;
    for (int i = 0 ; i < 100 ; i++) {
        double p;
        convnet.max_run_inference(1, images, false, &p);
        times.push_back(p);
    }
    fpgaconvnet::logging::set_level(fpgaconvnet::logging::INFO);
    fpgaconvnet::dump_latencies("latency.txt", times);
#endif

    return extracted_features;
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
            pixel_stream.push_back(images[i % images.size()][j]);
        }
    } 

    std::vector<float> conv_out = run_feature_extraction(
            network_parameters, pixel_stream);
    return 0;
}

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
#include "target_2.h"


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
    std::vector<float> extracted_features;

    max_files.push_back(target_0_init());
    max_files.push_back(target_1_init());
    max_files.push_back(target_2_init());

    fpgaconvnet::Convnet convnet(network_parameters, max_files, "");

    /* TODO: load the weights from somewhere. */
    convnet.randomize_weights();
    convnet.max_init_weights();

    /* warm up the DFE with the weights. */
    extracted_features = convnet.max_run_inference(N, images, false);

    /* TODO: Verify the output is correct. You can use the
     * fpgaconvnet::verify_conv_out function for this.
     */

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


int main(int argc, char **argv)
{
    std::vector<int> labels;
    std::vector<float> images;

    if (argc < 2) {
	std::cout << "Missing network descriptor" << std::endl;
	return 1;
    }

    std::cout << "Loading netowork parameters from " << argv[1] << std::endl;
    fpgaconvnet::protos::Network network_parameters =
	    fpgaconvnet::load_network_proto(argv[1]);
    std::cout << network_parameters.DebugString() << std::endl;

    std::cout << "Reading images ..." << std::endl;
    load_cifar10("cifar-10-batches-bin/data_batch_1.bin", images, labels);

    std::vector<float> conv_out = run_feature_extraction(network_parameters, images);
    return 0;
}

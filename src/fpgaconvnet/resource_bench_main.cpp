#include <iostream>
#include <fstream>
#include <vector>
#include <functional>

#include <cstring>

#include "fpgaconvnet/protos/parameters.pb.h"
#include "fpgaconvnet/convnet.h"

#ifdef __SIM__

#include "resource_bench_sim_lookup.h"
const uint64_t N = 1;

#else

#include "resource_bench_dfe_lookup.h"
const uint64_t N = 100;

#endif


void run_feature_extraction(
        const fpgaconvnet::protos::Network & network_parameters,
        max_file_t*(max_file_fnc)(),
        const std::vector<float> & images
)
{
    std::cout << network_parameters.DebugString() << std::endl;
    std::vector<std::string> filenames = {
            "../test_data/resource_bench/kernels.txt",
            "../test_data/resource_bench/bias.txt"
    };
    max_file_t *max_file = max_file_fnc();
    fpgaconvnet::Convnet convnet(network_parameters, max_file, "");

    convnet.load_weights_from_files(filenames);
    convnet.max_init_weights();
    convnet.max_load_input_data(images, N);
    convnet.max_run_inference(N);
    std::vector<float> extracted_features = convnet.max_retrieve_features(N);

    fpgaconvnet::verify_conv_output(
            network_parameters,
            N,
            &extracted_features[0],
            "../test_data/resource_bench/outputs.txt");
    std::cout << "============== END =============" << std::endl;
}


int main (int argc, char **argv)
{
    std::cout << "Running " << argc << " benchmarkings" << std::endl;

    std::vector<float> pixel_stream;
    std::ifstream fin("../test_data/resource_bench/inputs.txt");
    float x;
    while (fin >> x) {
        pixel_stream.push_back(x);
    }
    fin.close();
    std::cout << pixel_stream.size() << std::endl;

    for (int i = 1 ; i < argc ; i++) {
        fpgaconvnet::protos::Network network_parameters =
                fpgaconvnet::load_network_proto(argv[i]);
        max_file_t*(*fnc)() = NULL;
        int lookup_size = sizeof(resource_bench_lookup_table)
                / sizeof(resource_bench_lookup_table[0]);

        for (int j = 0 ; j < lookup_size ; j++) {
            if (std::strcmp(argv[i], resource_bench_lookup_table[j].name) == 0) {
                fnc = resource_bench_lookup_table[j].fnc;
                break;
            }
        }

        if (fnc) {
            run_feature_extraction(network_parameters, fnc, pixel_stream);
        } else {
            std::cout << "Cannot resolve function for " << argv[i] << std::endl;
        }
    }
}

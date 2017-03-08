#include <iostream>
#include <fstream>
#include <vector>
#include <functional>

#include <cstring>

#include "fpgaconvnet/protos/parameters.pb.h"
#include "fpgaconvnet/convnet.h"

#ifdef __SIM__

#include "resource_bench_sim_lookup.h"

#else

#include "resource_bench_dfe_lookup.h"

#endif


const uint64_t N = 5;
void run_feature_extraction(
        const fpgaconvnet::protos::Network & network_parameters,
        max_file_t*(max_file_fnc)()
)
{
    std::cout << network_parameters.DebugString() << std::endl;
    std::vector<std::string> filenames = {
            "../test_data/resource_bench/kernels.txt",
            "../test_data/resource_bench/bias.txt"
    };
    max_file_t *max_file = max_file_fnc();
    fpgaconvnet::Convnet convnet(network_parameters, max_file, "");
    auto first_layer = network_parameters.layer(0);
    uint64_t single_input_size = first_layer.num_inputs()
            * first_layer.input_height()
            * first_layer.input_width();
    std::vector<float> images(single_input_size * N);

    for (unsigned i = 0 ; i < images.size(); i++) {
        images[i] = float(rand()) / float(RAND_MAX);
    }

    convnet.randomize_weights();
    std::vector<float>extracted_features =
            convnet.max_run_inference(N, images, false);

    std::cout << "============== END =============" << std::endl;
}


int main (int argc, char **argv)
{
    std::cout << "Running " << argc - 1 << " resource benchmarks." << std::endl;

    for (int i = 1 ; i < argc ; i++) {
        char buffer[100];
        max_file_t*(*fnc)() = NULL;
        int lookup_size = sizeof(resource_bench_lookup_table)
                / sizeof(resource_bench_lookup_table[0]);

        sprintf(buffer, "../descriptors/resource_bench/%s.prototxt", argv[i]);
        fpgaconvnet::protos::Network network_parameters =
                fpgaconvnet::load_network_proto(buffer);
        fpgaconvnet::set_log_prefix(argv[i]);

        for (int j = 0 ; j < lookup_size ; j++) {
            if (std::strcmp(argv[i], resource_bench_lookup_table[j].name) == 0) {
                fnc = resource_bench_lookup_table[j].fnc;
                break;
            }
        }

        if (fnc) {
            run_feature_extraction(network_parameters, fnc);
        } else {
            std::cout << "Cannot resolve function for " << argv[i] << std::endl;
        }
    }
}

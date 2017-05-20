#include <iostream>
#include <vector>

#include <fpgaconvnet/common.h>


static std::vector<double> calculate_relative_worker_factors(
        const fpgaconvnet::protos::Network & network)
{
    std::vector<double> ret;
    const double base_size =
        network.layer(0).input_height() * network.layer(0).input_width();
    const double base_input_channels = network.layer(0).num_inputs();


    for (auto it = network.layer().begin() ; it != network.layer().end() ; it++) {
        const double size = it->input_height() * it->input_width();
        const double input_chnnels = it->num_inputs();
        double ratio;

        ret.push_back(ratio);
    }

    return ret;
}


int main (int argc, char **argv)
{
    fpgaconvnet::protos::Network network_parameters =
	    fpgaconvnet::load_network_proto(argv[1]);
    const std::vector<double> relative_worker_factors = 
            calculate_relative_worker_factors(network_parameters);
}

#ifndef FPGACONVNET_COMMON_H
#define FPGACONVNET_COMMON_H

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "fpgaconvnet/protos/parameters.pb.h"


namespace fpgaconvnet
{

typedef uint16_t fixed_point_t;
protos::Network load_network_proto(const std::string & filename);


class Exception : public std::exception {
private:
    std::string message;
public:
    Exception(const std::string & message);
    virtual ~Exception() throw();
    virtual const char* what() const throw();
};


namespace logging {

const int DEBUG = 0;
const int INFO = 1;
const int WARNING = 2;
const int ERROR = 3;

std::ostream& stdout(int level = INFO);
void indent();
void dedent();
void log_prefix(const std::string & prefix);

class Indentation {
public:
    Indentation();
    ~Indentation();
};

}  // logging

namespace math {

double rng(const double lo, const double hi);
uint64_t gcd(uint64_t a, uint64_t b);
uint64_t lcm(uint64_t a, uint64_t b);
uint64_t div_ceil(uint64_t a, uint64_t b);


}  // math


namespace calculation
{

uint64_t total_multipliers(const protos::LayerParameter & layer);

/* The number of iterations to perform a pixel calculation. */
uint64_t kernel_iterations(const protos::LayerParameter & layer);
uint64_t convolution_iterations(const protos::LayerParameter & layer);
uint64_t scheduler_iterations(const protos::LayerParameter & layer);
uint64_t total_iterations(const protos::LayerParameter &layer);

/* Weight initialization */
uint64_t total_kernel_weights(const protos::LayerParameter & layer);
uint64_t conv_in_size(const protos::Network & network);
uint64_t total_rom_size(const protos::LayerParameter & layer);
uint64_t weights_vector_size(const protos::LayerParameter & layer);
bool is_layer_cpu_initialized(const protos::LayerParameter & layer);

/* Stream size for inputs */
uint64_t bias_stream_size(const protos::LayerParameter & layer);
uint64_t cpu_weights_stream_size(const protos::LayerParameter & layer);


}  // calculation
}  // fpgaconvnet

#endif

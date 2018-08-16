#ifndef FPGACONVNET_COMMON_H
#define FPGACONVNET_COMMON_H

#include <cmath>
#include <fstream>
#include <iostream>
#include <sys/time.h>

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

const int DDEBUG = 0;
const int DEBUG = 1;
const int INFO = 2;
const int WARNING = 3;
const int ERROR = 4;

std::ostream& stdout(int level = INFO);
void indent();
void dedent();
void log_prefix(const std::string & prefix);
void set_level(int level);
void set_level(const char *level);

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

// In bytes per second
const double PCIE_BANDWIDTH    = 4e9;  // 2GB/s in each direction, so 4GB total
const double LMEM_BANDWIDTH    = 38e9; // 38GB/s in TOTAL
const double MAXRING_BANDWIDTH = 5e9;

enum bottleneck_type_t {
  BOTTLENECK_COMPUTE,
  BOTTLENECK_IO,
  BOTTLENECK_MAXRING,
};


struct compute_bottleneck_t {
  uint32_t layer_id;
};


struct maxring_bottleneck_t {
  uint32_t layer_id;
};


union bottleneck_t {
  compute_bottleneck_t compute;
  maxring_bottleneck_t maxring;
};


struct throughput_t {
  double            throughput;
  bottleneck_type_t bottleneck_type;
  bottleneck_t      bottleneck;

  bool operator<(const fpgaconvnet::calculation::throughput_t &) const;
  bool operator>(const fpgaconvnet::calculation::throughput_t &) const;
};

/* Throughput calculation in terms of images a second.
 *
 * Bitstream, in the comment below, refers to the maxfiles that are used
 * to configured A PIPELINE OF FPGAs that runs simultaneously. (It is
 * useful to think of a bistream as a set of maxfiles that will be
 * reconfigured simultaneously).
 *
 *  The different kinds of throughput means different things:
 *
 *    - [pipeline throughput]: The throughput of a bitstream when compiled
 *                             to several FPGAs arranged in a single
 *                             (massive) pipeline of devices. This is
 *                             the primary evaluation metric when measuring
 *                             the performance of a pipeline that doesn't
 *                             permit reconfiguration. This is useful for
 *                             cases where latency is still somewhat
 *                             sensitive, but not critical (eg: game-playing
 *                             bots which needs <10ms latency) and
 *                             throughput is still very important. (good
 *                             FPS without high latency).
 *           
 *    - [effective throuhgput]: The throughput of a bitstream when
 *                              considering several parallel similarly
 *                              configured pipelines running simultaneously.
 *                              This throughput is not useful for overall
 *                              evaluation, but useful for indentifying
 *                              performance bottleneck of bistreams, besides
 *                              being used to calculate the [real throughput].
 *
 *    - [real throughput]: The real throughput that utilises as much
 *                         resources as required from the given resources.
 *                         This never simply returns the pipeline throughput,
 *                         even when it is possible to get higher pipeline
 *                         throughput whilist sacrificing effective
 *                         throughput. This is the primary evaluation
 *                         metric for latency-insensitive batch processing.
 *
 *                         Unlike pipeline and effective throughput,
 *                         real throughput is a scalar. This is because
 *                         there is not true bottleneck bitstream - the
 *                         numbers are not minimised over, but rather, taken
 *                         as the reciprocal of their sum. Namely:
 *
 *                         th = 1 / (1 / th_1 + 1/th_2 + ... + 1/th_n)
 *
 *                         where th refers to throughput.
 *                         
 */
throughput_t pipeline_throughput(
        const protos::Network & network, const unsigned bitstream_id);
throughput_t effective_throughput(
        const protos::Network & network, const unsigned bitstream_id);
const double bandwidth_throughput_limit(
        const protos::Network & network, const unsigned bitstream_id);

void explain_throughput(const protos::Network & network);
double real_throughput(const protos::Network & network);
double min_num_fpga_real_throughput(const protos::Network & network);
unsigned min_num_fpga_needed(const protos::Network & network);

/* The total number of network operations in the network. */
double ops(const protos::Network & network);

uint64_t total_multipliers(const protos::LayerParameter & layer);

/* The number of iterations to perform a pixel calculation. */
uint64_t kernel_iterations(const protos::LayerParameter & layer);
uint64_t convolution_iterations(const protos::LayerParameter & layer);
uint64_t scheduler_iterations(const protos::LayerParameter & layer);
uint64_t total_iterations(const protos::LayerParameter &layer);

/* Weight initialization */
uint64_t total_kernel_weights(const protos::LayerParameter & layer);
uint64_t total_rom_size(const protos::LayerParameter & layer);
uint64_t weights_vector_size(const protos::LayerParameter & layer);
bool is_layer_cpu_initialized(const protos::LayerParameter & layer);

/* Stream size for weight initialisation */
uint64_t bias_stream_size(const protos::LayerParameter & layer);
uint64_t cpu_weights_stream_size(const protos::LayerParameter & layer);

/* Number of numerical values for a single input image */
uint64_t conv_in_size(const protos::Network & network);
uint64_t conv_in_size_for_bitstream(
        const protos::Network & network, const unsigned bitstream_id);


}  // calculation

protos::Network insert_fpga_positions(protos::Network, std::vector<int>);
std::vector<protos::Network> split_by_bitstreams(protos::Network);

}  // fpgaconvnet


std::ostream& operator<<(std::ostream & o, const fpgaconvnet::calculation::throughput_t &);


/* Auxilary general-purpose utility functions */

static void log_vector(const std::vector<double> & v)
{
    fpgaconvnet::logging::stdout() << "[ ";
    for (int i = 0 ; i < v.size() ; i++) {
        std::cout << v[i] << " ; ";
    }
    std::cout << "]" << std::endl;
}


/* Rounds up x such that ceil % x == 0 */
static uint64_t ceil_divisible(double x, uint64_t ceil) {
    uint64_t ret = std::ceil(x);

    if (ret >= ceil) {
        return ceil;
    }

    for (; ret < ceil ; ret++) {
        if (ceil % ret == 0) {
            return ret;
        }
    }

    return ret;
}


static double
compute_time_difference(timeval t_begin, timeval t_end)
{
    double begin = double(t_begin.tv_sec) * 1000000 + double(t_begin.tv_usec);
    double end = double(t_end.tv_sec) * 1000000 + double(t_end.tv_usec);
    return end - begin;
}


static std::vector<float>
load_float_stream(std::string filename, const unsigned total_size)
{
    std::vector<float> images(total_size);
    std::ifstream fin(filename.c_str());
    for (unsigned i = 0; i < total_size ; i++) {
        fin.read((char*) &images[i], 4);
    }
    fin.close();
    return images;
}

static std::vector<float>
load_and_duplicate_float_stream(
        const std::string & filename,
        const unsigned single_input_size,
        const unsigned num_copies_in_file,
        const unsigned num_copies_to_make)
{
    const std::vector<float> sample = load_float_stream(
            filename.c_str(), num_copies_in_file * single_input_size);
    std::vector<float> pixel_stream(num_copies_to_make * single_input_size);
    const unsigned num_batches = num_copies_to_make / num_copies_in_file;
    const unsigned remainder = num_copies_to_make % num_copies_in_file;

    // copy to pixel stream
    for (unsigned i = 0; i < num_batches; i++) {
        memcpy(&pixel_stream[i * single_input_size * num_copies_in_file],
                &sample[0],
                sizeof(float) * sample.size());
    }

    if (remainder != 0) {
        memcpy(&pixel_stream[num_batches * single_input_size * num_copies_in_file],
                &sample[0],
                sizeof(float) * single_input_size * remainder);
    }

    return pixel_stream;
}

#endif

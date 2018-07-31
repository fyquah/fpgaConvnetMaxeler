#ifndef BUILD_SINGLE_BITSTREAM_H
#define BUILD_SINGLE_BITSTREAM_H

namespace fpgaconvnet {
namespace modelling {

/* Computes the best solution to fit a given (sub-)Network into a piepline
 * of FPGAs, as specified by [reference_.get_avialable_fpga()]. A single
 * bitstream, as suggested by the class name, referes to a set of maxfiles
 * that are simulatneously run on a pipeline of FPGAs.
 *
 * WARNING: This class is not thread safe.
 */
class BuildSingleBitStream {
private:
    bool success_;
    bool done_;
    const fpgaconvnet::protos::Network reference_;
    fpgaconvnet::protos::Network solution_;
public:
    BuildSingleBitStream(fpgaconvnet::protos::Network);
    bool search();
    fpgaconvnet::protos::Network get_result();
};


}  // modelling
}  // fpgaconvnet

#endif

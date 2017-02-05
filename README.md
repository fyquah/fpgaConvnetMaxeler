# fpgaconvnet on Maxeler

Running fast convolutional neural network inference on generic networks using
maxeler DFEs.

## Dependencies

- protoc-3.0.0 compiler (libproto-java.jar is not required, tho)


## Conventions

1. Filters are indexed by `output_channels * input_channels * input_height * input_width`
   4D Array should have the relevant dimensions and 1D flatten array storage
   should have the given output_channels-major configuration.

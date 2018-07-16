# fpgaconvnet on Maxeler

Running fast convolutional neural network (CNN) inference on maxeler dataflow engines (DFE).
This

## Dependencies

- protoc-3.0.0 compiler (libproto-java.jar is not required)
- libprotobuf.so (install this in your `LD_LIBRARY_PATH`)

## Project Structure

## Usage

### Step 0: Write the CNN's Protobuf Desciptor

The protobuf descriptor is given by `protos/fpgaconvnet/protos/parameters.proto`.
You are required to define a `Network` protobuf. Remember to specify the number
of available fpga in num\_fpga\_used.

Refer to `descriptors/lenet.prototxt` for an example.

### Step 1: Generate a project

```bash
python ./scripts/create_project \
  --nets my_net.prototxt \
  --dir projects/my_net
  --max-ring  # This allows the usage of multiple fpgas.
  my_net
```

This creates a project in `projects/my_net` (which is, by default, tracked by
git). In the projects, you will see several files:

```bash
build/
  Makefile  # A generated Makefile. This Makefile, on its own doesn't contain
            # any targets. The targets are included via other Makefiles. You
            # can add custom targets, as long as you don't overwrite the original
            # contents.
src/
  main.cpp  # A default executable that is generated. This passes a random stream
            # of numbers into the generated network, and does not check the output.
            # You (probably) want to modify this, unless you are interested only
            # in performance numbers and is 100% sure about the model's correctness.
descriptors/
  my_net.prototxt  # The net you have created. This is a copy of the file you have
                   # made earlier, not a symlink / hard link.
```

### Step 2: Compiling

Depending on your requirements, you will want to modify `projects/my_net/src/main.cpp`
based on your needs.

Firstly, you need to perform DSE and generate the relevant targets.

```bash
make optimize  # Design space exploration generates <net>.optimized.prototxt
make gen_makefile  # This generates a Makefile, based on the number of FPGA required.
                   # it is possible to use less than the available FPGA to have
                   # better throughput.
```

### Step 4: Simulation

(Optional, but highly recommended) Secondly, you want to simulate the maxfile.

```bash
make sim
make runsim
```

### Step 4: DFE

Once you are ready, you can compile the net into the DFE (This will take awhile)
and you can run it. The default `main.cpp` file will run inference twice and report
their execution times.

```bash
make dfe
make rundfe
```

## Example: Lenet

TODO

## Example: Alexnet

TODO

## Hacking

TODO

##

## License

TBC

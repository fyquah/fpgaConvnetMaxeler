# fpgaconvnet on Maxeler

Running fast convolutional neural network inference on generic networks using
maxeler DFEs.

## Dependencies

- protoc-3.0.0 compiler (libproto-java.jar is not required, tho)


## Design Space Exploration

Recognize that for optimal resource usage at runtime (100% utilization, or near
100%, of all allocated resources), we require:

```
wf(i) / wf(0) = (in(i) / in(0)) * (size(i) / size(0))
```

### Assuming:

- `resource(wf(i), cff(i), kff(i)) > resource(wf(j), cff(j), kff(j))` iff
  `wf(i) > wf(j) && cff(i) > cff(j) && kff(i) > kff(j)`. This is not true in
  practice and we will have to handle such cases in our optimizer.
- Fifos are deep enough to store the intermediate buffers. This is almost
  always true.

### Pseudo Algorithm:

1. Choose a set of wf that hasn't been explored.
2. Adjust wf such that resource usage in FIFOs and sliding windows are minimal
3. Search kff and cff such that
   `input(i) / wf(i) >= total_iterations * size_out(i) / size_in(i)`
   and `total_iterations <= input(i + 1) / wf(i + 1)`
   and constraints are just met. (I.e: use least amount of resources - this can
   be a bit tricky with large nets, as we might end up more than one BRAM per
   multiplier - something we want to avoid).
4. If this meets the constraint, add to set of explored solution. If satisfactory
   return. Otherwise, goto 1.



## Conventions

1. Filters are indexed by `output_channels * input_channels * input_height * input_width`
   4D Array should have the relevant dimensions and 1D flatten array storage
   should have the given output_channels-major configuration.

# Refactor

The point of the refactor is to make ConvolutionUnit a *kernel on its own*. In part because i am getting a headache getting offseting and pipelining to sing well together.

To get ConvolutionLayer to work, there are a few things that we will need to consider:

1. A scheduler that decides what goes into which ConvolutionUnit

- Should SlidingWindow be a kernel on its own, or can it live happily in ConvolutionSchedulerKernel?
- Considering there is no computation happening within ConvolutionSchedulerKernel, it just might be possible 
to do so without much complications.
- Either way, please think about it more in depth.

[ConvolutionSchedulerKernel]
input: 
    - DFEVector[n_input_channels]

output:
    - output p0, p1, p2, p3, .... , p{n_con_units - 1} , y0, y1, y2, ... , y{n_conv_units - 1}
    - Should use a DFEVector for this? Considering they are going to come out at the same time.

2. How do we schedule ConvolutionUnit to do things correctly?

[ConvolutionUnitKernel]

3. Result accumulator that collects the results of convolutions - how would this work?

[ConvolutionAccumulatorKernel]
input:
    - DFEVector[n_conv_units]  -> rationale: all the results from ConvolutionUnitKernel

output:
    - DFEVector[n_output_channels]

4. How can I test this?

- I have no idea :p

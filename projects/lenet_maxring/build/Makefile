##########################################
#     Makefile for project lenet_maxring
#     
#     A good deal of good here is generated, so
#     do not modify (unless you know what you are doing)
##########################################

WHEREISROOT=../../..

PRJ_NAME=lenet
PRJ_DESCRIPTOR=../descriptors/lenet.prototxt
PRJ_OPTIMIZED_DESCRIPTOR=lenet.optimized.prototxt
DFEModel=MAIA

#### BEGIN: ADD CUSTOM TARGETS HERE

MNIST_DATA=mnist/train-images-idx3-ubyte mnist/train-labels-idx1-ubyte \
	    mnist/t10k-images-idx3-ubyte mnist/t10k-labels-idx1-ubyte
ADDITIONAL_OBJS=fpgaconvnet/mnist.o

$(MNIST_DATA):
	mkdir -p mnist/
	curl "http://yann.lecun.com/exdb/$@.gz" >$@.gz
	gzip -d $@.gz

data: $(MNIST_DATA)

#### END: DO NOT ADD STUFF BEYOND THIS LINE

ifeq ($(shell test -e Makefile.targets && echo -n yes), yes)
include Makefile.targets
endif

include $(WHEREISROOT)/Makefile.base
include $(WHEREISROOT)/Makefile.multi.fpga.base

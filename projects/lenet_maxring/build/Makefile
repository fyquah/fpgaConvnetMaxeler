WHEREISROOT=../../..

include $(WHEREISROOT)/Makefile.base

PRJ_NAME=lenet_maxring
PRJ_DESCRIPTOR=../descriptors/lenet.prototxt
PRJ_OPTIMIZED_DESCRIPTOR=lenet.optimized.prototxt

MNIST_DATA=mnist/train-images-idx3-ubyte mnist/train-labels-idx1-ubyte \
	    mnist/t10k-images-idx3-ubyte mnist/t10k-labels-idx1-ubyte
ADDITIONAL_OBJS=fpgaconvnet/mnist.o

$(MNIST_DATA):
	mkdir -p mnist/
	curl "http://yann.lecun.com/exdb/$@.gz" >$@.gz
	gzip -d $@.gz

data: $(MNIST_DATA)


$(PRJ_OPTIMIZED_DESCRIPTOR): fpgaconvnet/modelling/optimizer.py $(PRJ_DESCRIPTOR) $(PYTHON_PROTOS)
	python fpgaconvnet/modelling/optimizer.py \
	    --resource-bench $(WHEREISROOT)/results/resource_bench.yaml \
	    --output $@ \
	    $(PRJ_DESCRIPTOR)


gen_makefile: $(WHEREISROOT)/scripts/generate_multi_fpga_makefile.py $(PRJ_OPTIMIZED_DESCRIPTOR)
	python $< --descriptor $(PRJ_OPTIMIZED_DESCRIPTOR) \
		--output Makefile.targets


ifeq ($(shell test -e Makefile.targets && echo -n yes), yes)

include Makefile.targets
include $(WHEREISROOT)/Makefile.multi.fpga.base

endif

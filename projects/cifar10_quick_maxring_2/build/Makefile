##########################################
#     Makefile for project cifar10_quick
#     
#     A good deal of good here is generated, so
#     do not modify (unless you know what you are doing)
##########################################

WHEREISROOT=../../..

PRJ_NAME=cifar10_quick_maxring
PRJ_DESCRIPTOR=../descriptors/cifar10_quick_maxring.prototxt
PRJ_OPTIMIZED_DESCRIPTOR=cifar10_quick_maxring.optimized.prototxt
DFEModel=MAIA

#### BEGIN: ADD CUSTOM TARGETS HERE

data:
	curl "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz" >cifar10-binary.tar.gz
	tar xzvf cifar10-binary.tar.gz

#### END: DO NOT ADD STUFF BEYOND THIS LINE

ifeq ($(shell test -e Makefile.targets && echo -n yes), yes)
include Makefile.targets
endif

include $(WHEREISROOT)/Makefile.base
include $(WHEREISROOT)/Makefile.multi.fpga.base

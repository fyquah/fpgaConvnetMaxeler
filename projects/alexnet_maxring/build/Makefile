##########################################
#     Makefile for project alexnet
#     
#     A good deal of good here is generated, so
#     do not modify (unless you know what you are doing)
##########################################

WHEREISROOT=../../..

PRJ_NAME=alexnet
PRJ_DESCRIPTOR=../descriptors/alexnet.prototxt
PRJ_OPTIMIZED_DESCRIPTOR=alexnet.optimized.prototxt
DFEModel=MAIA

#### BEGIN: ADD CUSTOM TARGETS HERE

model.caffemodel:
	curl http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel \
	  >$@
	  

deploy.prototxt:
	curl -sSL https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt \
	  >$@

testdata: ../gendata.py deploy.prototxt model.caffemodel
	mkdir -p testdata/weights/
	mkdir -p testdata/data/
	python $<

#### END: DO NOT ADD STUFF BEYOND THIS LINE

ifeq ($(shell test -e Makefile.targets && echo -n yes), yes)
include Makefile.targets
endif

include $(WHEREISROOT)/Makefile.base
include $(WHEREISROOT)/Makefile.multi.fpga.base

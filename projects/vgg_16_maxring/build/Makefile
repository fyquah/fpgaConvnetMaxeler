##########################################
#     Makefile for project alexnet
#     
#     A good deal of good here is generated, so
#     do not modify (unless you know what you are doing)
##########################################

WHEREISROOT=../../..

PRJ_NAME=vgg_16
PRJ_DESCRIPTOR=../descriptors/vgg_16.prototxt
PRJ_OPTIMIZED_DESCRIPTOR=vgg_16_bck.optimized.prototxt
DFEModel=MAIA

#### BEGIN: ADD CUSTOM TARGETS HERE

model.caffemodel:
	curl http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
	  >$@

deploy.prototxt:
	curl -sSL https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt \
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

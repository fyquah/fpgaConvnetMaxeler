WHEREISROOT=../../..

include $(WHEREISROOT)/Makefile.base

PRJ_NAME=vgg_s
PRJ_DESCRIPTOR=../descriptors/vgg_s.prototxt
PRJ_OPTIMIZED_DESCRIPTOR=vgg_s.optimized.prototxt


$(PRJ_OPTIMIZED_DESCRIPTOR): fpgaconvnet/modelling/optimizer $(PRJ_DESCRIPTOR)
	touch $(PRJ_OPTIMIZED_DESCRIPTOR);
	./$< $(PRJ_DESCRIPTOR) $@

gen_makefile: $(WHEREISROOT)/scripts/generate_multi_fpga_makefile.py $(PRJ_OPTIMIZED_DESCRIPTOR)
	python $< --descriptor $(PRJ_OPTIMIZED_DESCRIPTOR) \
		--output Makefile.targets


ifeq ($(shell test -e Makefile.targets && echo -n yes), yes)

include Makefile.targets
include $(WHEREISROOT)/Makefile.multi.fpga.base


debugsim:
	maxdebug -v -g vgg_0_Graph -s vgg_0_Graph -c -d $(USER)a0:$(USER)a \
	    target_5_MAIA_DFE_SIM/results/target_5.max


endif

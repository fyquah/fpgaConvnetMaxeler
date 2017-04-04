WHEREISROOT=../../..

include $(WHEREISROOT)/Makefile.base

PRJ_NAME=cifar10_quick_maxring
PRJ_DESCRIPTOR=../descriptors/cifar10_quick_maxring.prototxt
PRJ_OPTIMIZED_DESCRIPTOR=cifar10_quick_maxring.optimized.prototxt


data:
	curl "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz" >cifar10-binary.tar.gz
	tar xzvf cifar10-binary.tar.gz


$(PRJ_OPTIMIZED_DESCRIPTOR): fpgaconvnet/modelling/optimizer.py \
			     fpgaconvnet/modelling/resource_model.py \
  			     $(PRJ_DESCRIPTOR) $(PYTHON_PROTOS)
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

debugsim:
	maxdebug -v -g cifar10_Graph -s cifar10_Graph -c -d $(USER)a0:$(USER)a \
	    target_1_MAIA_DFE_SIM/results/target_1.max

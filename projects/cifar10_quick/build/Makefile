WHEREISROOT=../../..

include $(WHEREISROOT)/Makefile.base

CIFAR10_QUICK_NAME=cifar10_quick
CIFAR10_QUICK_SIMMAXDIR=$(MAXCOMPILER_BUILD_DIR)/$(CIFAR10_QUICK_NAME)_$(DFEModel)_DFE_SIM/results
CIFAR10_QUICK_DFEMAXDIR=$(MAXCOMPILER_BUILD_DIR)/$(CIFAR10_QUICK_NAME)_$(DFEModel)_DFE/results
CIFAR10_QUICK_SIM_MAXFILE=$(MAXCOMPILER_BUILD_DIR)/$(CIFAR10_QUICK_NAME)_$(DFEModel)_DFE_SIM/results/$(CIFAR10_QUICK_NAME).max
CIFAR10_QUICK_DFE_MAXFILE=$(MAXCOMPILER_BUILD_DIR)/$(CIFAR10_QUICK_NAME)_$(DFEModel)_DFE/results/$(CIFAR10_QUICK_NAME).max
CIFAR10_QUICK_DESCRIPTOR=../descriptors/cifar10_quick.prototxt
CIFAR10_QUICK_OPTIMIZED_DESCRIPTOR=cifar10_quick.optimized.prototxt
CIFAR10_QUICK_DATA=cifar-10-batches-bin/


$(CIFAR10_QUICK_DATA):
	curl "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz" >cifar10-binary.tar.gz
	tar xzvf cifar10-binary.tar.gz

$(CIFAR10_QUICK_OPTIMIZED_DESCRIPTOR): fpgaconvnet/modelling/optimizer \
				       $(CIFAR10_QUICK_DESCRIPTOR)
	touch $@
	./$< $(CIFAR10_QUICK_DESCRIPTOR) $@


# ==============================
#           Simulation
# ==============================

$(CIFAR10_QUICK_SIM_MAXFILE): $(ENGINEFILES) $(JAVA_PROTOS) $(CIFAR10_QUICK_OPTIMIZED_DESCRIPTOR)
	$(MAXJC) $(JFLAGS) $(filter %.java,$^) $(filter %.maxj,$^)
	MAXAPPJCP='.:$(CP)' MAXSOURCEDIRS=$(SOURCE_DIRS) \
		  $(MAXJAVARUN) fpgaconvnet.GenericNetworkBuilder \
		  DFEModel=$(DFEModel) \
		  maxFileName=$(CIFAR10_QUICK_NAME) \
		  target="DFE_SIM" \
		  descriptor=$(CIFAR10_QUICK_OPTIMIZED_DESCRIPTOR) \
		  enableMPCX=$(MPCX)


cifar10_quick_sim.o: $(CIFAR10_QUICK_SIM_MAXFILE)
	$(SLICCOMPILE) $< $@


cifar10_quick_main_sim.o: main.cpp $(CIFAR10_QUICK_SIM_MAXFILE)
	$(CPUCOMPILER) $< $(ALL_FLAGS) \
		-I$(MAXCOMPILER_BUILD_DIR) -I$(CIFAR10_QUICK_SIMMAXDIR) \
		-D__SIM__ -DDESIGN_NAME=$(CIFAR10_QUICK_NAME) -c -o $@


cifar10_quick_sim: $(FPGACONVNET_OBJ_FILES) \
	    cifar10_quick_sim.o cifar10_quick_main_sim.o
	$(CPUCOMPILER) $(ALL_FLAGS) -o $@ $^ $(LFLAGS)


run_cifar10_quick_sim: cifar10_quick_sim $(CIFAR10_QUICK_DATA) $(CIFAR10_QUICK_OPTIMIZED_DESCRIPTOR)
	$(MAXCOMPILERSIM) -n $(USER)a -c$(DEVICENUM) -d$(NUMDEVICES) restart
	SLIC_CONF+="use_simulation=$(USER)a; default_topology_timeout=-1" \
		   LD_PRELOAD=$(MAXOS_SIM) \
		   ./$< $(CIFAR10_QUICK_OPTIMIZED_DESCRIPTOR)
	make stopsim

# ==============================
#          DFE
# ==============================

$(CIFAR10_QUICK_DFE_MAXFILE): $(ENGINEFILES) $(JAVA_PROTOS) $(CIFAR10_QUICK_OPTIMIZED_DESCRIPTOR)
	$(MAXJC) $(JFLAGS) $(filter %.java,$^) $(filter %.maxj,$^)
	MAXAPPJCP='.:$(CP)' MAXSOURCEDIRS=$(SOURCE_DIRS) \
		  $(MAXJAVARUN) fpgaconvnet.GenericNetworkBuilder \
		  DFEModel=$(DFEModel) \
		  maxFileName=$(CIFAR10_QUICK_NAME) \
		  target="DFE" \
		  descriptor=$(CIFAR10_QUICK_OPTIMIZED_DESCRIPTOR) \
		  enableMPCX=$(MPCX)


fpgaconvnet/cifar10_quick_dfe.o: $(CIFAR10_QUICK_DFE_MAXFILE)
	$(SLICCOMPILE) $< $@


fpgaconvnet/cifar10_quick_main_dfe.o: fpgaconvnet/cifar10_quick_main.cpp $(CIFAR10_QUICK_DFE_MAXFILE)
	$(CPUCOMPILER) $< $(ALL_FLAGS) \
		-I$(MAXCOMPILER_BUILD_DIR) -I$(CIFAR10_QUICK_DFEMAXDIR) \
		-DDESIGN_NAME=$(CIFAR10_QUICK_NAME) -c -o $@


cifar10_quick_dfe: $(FPGACONVNET_OBJ_FILES) \
	    fpgaconvnet/cifar10_quick_dfe.o fpgaconvnet/cifar10_quick_main_dfe.o \
	    fpgaconvnet/protos/parameters.pb.o
	$(CPUCOMPILER) $(ALL_FLAGS) -o $@ $^ $(LFLAGS)


run_cifar10_quick_dfe: cifar10_quick_dfe $(CIFAR10_QUICK_DATA) $(CIFAR10_QUICK_OPTIMIZED_DESCRIPTOR)
	SLIC_CONF="$(SLIC_CONF)" LD_PRELOAD=$(MAXOS_HW) ./$< $(CIFAR10_QUICK_OPTIMIZED_DESCRIPTOR)


# ---- Debug ----

debugsim:
	maxdebug -v -g CIFAR10_QUICKGraph -s CIFAR10_QUICKGraph -c -d $(USER)a0:$(USER)a \
	    $(CIFAR10_QUICK_SIM_MAXFILE)

sim: cifar10_quick_sim

runsim: run_cifar10_quick_sim

dfe: cifar10_quick_dfe

rundfe: run_cifar10_quick_dfe

WHEREISROOT=../../..

include $(WHEREISROOT)/Makefile.base

LENET_NAME=lenet
LENET_SIMMAXDIR=$(MAXCOMPILER_BUILD_DIR)/$(LENET_NAME)_$(DFEModel)_DFE_SIM/results
LENET_DFEMAXDIR=$(MAXCOMPILER_BUILD_DIR)/$(LENET_NAME)_$(DFEModel)_DFE/results
LENET_SIM_MAXFILE=$(MAXCOMPILER_BUILD_DIR)/$(LENET_NAME)_$(DFEModel)_DFE_SIM/results/$(LENET_NAME).max
LENET_DFE_MAXFILE=$(MAXCOMPILER_BUILD_DIR)/$(LENET_NAME)_$(DFEModel)_DFE/results/$(LENET_NAME).max
LENET_DESCRIPTOR=../descriptors/lenet.prototxt
LENET_OPTIMIZED_DESCRIPTOR=lenet.optimized.prototxt
LENET_DATA=mnist/train-images-idx3-ubyte mnist/train-labels-idx1-ubyte \
	    mnist/t10k-images-idx3-ubyte mnist/t10k-labels-idx1-ubyte

$(LENET_DATA):
	mkdir -p mnist/
	curl "http://yann.lecun.com/exdb/$@.gz" >$@.gz
	gzip -d $@.gz


$(LENET_OPTIMIZED_DESCRIPTOR): fpgaconvnet/modelling/optimizer $(LENET_DESCRIPTOR)
	./$< $(LENET_DESCRIPTOR) $@


# ==============================
#           Simulation
# ==============================

$(LENET_SIM_MAXFILE): $(ENGINEFILES) $(JAVA_PROTOS) $(LENET_OPTIMIZED_DESCRIPTOR)
	$(MAXJC) $(JFLAGS) $(filter %.java,$^) $(filter %.maxj,$^)
	MAXAPPJCP='.:$(CP)' MAXSOURCEDIRS=$(SOURCE_DIRS) \
		  $(MAXJAVARUN) fpgaconvnet.GenericNetworkBuilder \
		  DFEModel=$(DFEModel) \
		  maxFileName=$(LENET_NAME) \
		  target="DFE_SIM" \
		  descriptor=$(LENET_OPTIMIZED_DESCRIPTOR) \
		  enableMPCX=$(MPCX)


lenet_sim.o: $(LENET_SIM_MAXFILE)
	$(SLICCOMPILE) $< $@


lenet_main_sim.o: main.cpp $(LENET_SIM_MAXFILE)
	$(CPUCOMPILER) $< $(ALL_FLAGS) \
		-I$(MAXCOMPILER_BUILD_DIR) -I$(LENET_SIMMAXDIR) \
		-D__SIM__ -DDESIGN_NAME=$(LENET_NAME) -c -o $@


lenet_sim: $(FPGACONVNET_OBJ_FILES) \
	    lenet_sim.o lenet_main_sim.o
	$(CPUCOMPILER) $(ALL_FLAGS) -o $@ $^ $(LFLAGS)


run_lenet_sim: lenet_sim $(LENET_DATA)
	$(MAXCOMPILERSIM) -n $(USER)a -c$(DEVICENUM) -d$(NUMDEVICES) restart
	SLIC_CONF+="use_simulation=$(USER)a; default_topology_timeout=-1" \
		   LD_PRELOAD=$(MAXOS_SIM) \
		   ./$< $(LENET_OPTIMIZED_DESCRIPTOR)
	make stopsim

# ==============================
#          DFE
# ==============================

$(LENET_DFE_MAXFILE): $(ENGINEFILES) $(JAVA_PROTOS) $(LENET_OPTIMIZED_DESCRIPTOR)
	$(MAXJC) $(JFLAGS) $(filter %.java,$^) $(filter %.maxj,$^)
	MAXAPPJCP='.:$(CP)' MAXSOURCEDIRS=$(SOURCE_DIRS) \
		  $(MAXJAVARUN) fpgaconvnet.GenericNetworkBuilder \
		  DFEModel=$(DFEModel) \
		  maxFileName=$(LENET_NAME) \
		  target="DFE" \
		  descriptor=$(LENET_OPTIMIZED_DESCRIPTOR) \
		  enableMPCX=$(MPCX)


fpgaconvnet/lenet_dfe.o: $(LENET_DFE_MAXFILE)
	$(SLICCOMPILE) $< $@


fpgaconvnet/lenet_main_dfe.o: fpgaconvnet/lenet_main.cpp $(LENET_DFE_MAXFILE)
	$(CPUCOMPILER) $< $(ALL_FLAGS) \
		-I$(MAXCOMPILER_BUILD_DIR) -I$(LENET_DFEMAXDIR) \
		-DDESIGN_NAME=$(LENET_NAME) -c -o $@


lenet_dfe: $(FPGACONVNET_OBJ_FILES) \
	    fpgaconvnet/lenet_dfe.o fpgaconvnet/lenet_main_dfe.o \
	    fpgaconvnet/protos/parameters.pb.o
	$(CPUCOMPILER) $(ALL_FLAGS) -o $@ $^ $(LFLAGS)


run_lenet_dfe: lenet_dfe $(LENET_DATA) $(LENET_DESCRIPTOR)
	SLIC_CONF="$(SLIC_CONF)" LD_PRELOAD=$(MAXOS_HW) ./$< $(LENET_OPTIMIZED_DESCRIPTOR)


# ---- Debug ----

debugsim:
	maxdebug -v -g LENETGraph -s LENETGraph -c -d $(USER)a0:$(USER)a \
	    $(LENET_SIM_MAXFILE)

sim: lenet_sim

runsim: run_lenet_sim

dfe: lenet_dfe

rundfe: run_lenet_dfe

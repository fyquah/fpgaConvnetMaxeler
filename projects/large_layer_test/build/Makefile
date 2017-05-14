WHEREISROOT=../../..

include $(WHEREISROOT)/Makefile.base

LARGE_LAYER_TEST_NAME=large_layer_test
LARGE_LAYER_TEST_SIMMAXDIR=$(MAXCOMPILER_BUILD_DIR)/$(LARGE_LAYER_TEST_NAME)_$(DFEModel)_DFE_SIM/results
LARGE_LAYER_TEST_DFEMAXDIR=$(MAXCOMPILER_BUILD_DIR)/$(LARGE_LAYER_TEST_NAME)_$(DFEModel)_DFE/results
LARGE_LAYER_TEST_SIM_MAXFILE=$(MAXCOMPILER_BUILD_DIR)/$(LARGE_LAYER_TEST_NAME)_$(DFEModel)_DFE_SIM/results/$(LARGE_LAYER_TEST_NAME).max
LARGE_LAYER_TEST_DFE_MAXFILE=$(MAXCOMPILER_BUILD_DIR)/$(LARGE_LAYER_TEST_NAME)_$(DFEModel)_DFE/results/$(LARGE_LAYER_TEST_NAME).max
LARGE_LAYER_TEST_DESCRIPTOR=../descriptors/large_layer_test.prototxt
LARGE_LAYER_TEST_OPTIMIZED_DESCRIPTOR=large_layer_test.optimized.prototxt


$(LARGE_LAYER_TEST_OPTIMIZED_DESCRIPTOR): fpgaconvnet/modelling/optimizer.py $(LARGE_LAYER_TEST_DESCRIPTOR) $(PYTHON_PROTOS)
	python $< --resource-bench $(WHEREISROOT)/results/resource_bench.yaml --output $@ \
	    $(LARGE_LAYER_TEST_DESCRIPTOR)


resource_info: $(PY_PROTOS) fpgaconvnet/modelling/resource_model.py
	python $< $(LARGE_LAYER_TEST_OPTIMIZED_DESCRIPTOR)

# ==============================
#           Simulation
# ==============================

$(LARGE_LAYER_TEST_SIM_MAXFILE): $(ENGINEFILES) $(JAVA_PROTOS) $(LARGE_LAYER_TEST_OPTIMIZED_DESCRIPTOR)
	$(MAXJC) $(JFLAGS) $(filter %.java,$^) $(filter %.maxj,$^)
	MAXAPPJCP='.:$(CP)' MAXSOURCEDIRS=$(SOURCE_DIRS) \
		  $(MAXJAVARUN) fpgaconvnet.GenericNetworkBuilder \
		  DFEModel=$(DFEModel) \
		  maxFileName=$(LARGE_LAYER_TEST_NAME) \
		  target="DFE_SIM" \
		  descriptor=$(LARGE_LAYER_TEST_OPTIMIZED_DESCRIPTOR) \
		  enableMPCX=$(MPCX)


large_layer_test_sim.o: $(LARGE_LAYER_TEST_SIM_MAXFILE)
	$(SLICCOMPILE) $< $@


large_layer_test_main_sim.o: main.cpp $(LARGE_LAYER_TEST_SIM_MAXFILE)
	$(CPUCOMPILER) $< $(ALL_FLAGS) \
		-I$(MAXCOMPILER_BUILD_DIR) -I$(LARGE_LAYER_TEST_SIMMAXDIR) \
		-D__SIM__ -DDESIGN_NAME=$(LARGE_LAYER_TEST_NAME) -c -o $@


large_layer_test_sim: $(FPGACONVNET_OBJ_FILES) \
	    large_layer_test_sim.o large_layer_test_main_sim.o
	$(CPUCOMPILER) $(ALL_FLAGS) -o $@ $^ $(LFLAGS)


run_large_layer_test_sim: large_layer_test_sim $(LARGE_LAYER_TEST_DATA) $(LARGE_LAYER_TEST_OPTIMIZED_DESCRIPTOR)
	$(MAXCOMPILERSIM) -n $(USER)a -c$(DEVICENUM) -d$(NUMDEVICES) restart
	SLIC_CONF+="use_simulation=$(USER)a; default_topology_timeout=-1" \
		   LD_PRELOAD=$(MAXOS_SIM) \
		   ./$< $(LARGE_LAYER_TEST_OPTIMIZED_DESCRIPTOR)
	make stopsim

# ==============================
#          DFE
# ==============================

$(LARGE_LAYER_TEST_DFE_MAXFILE): $(ENGINEFILES) $(JAVA_PROTOS) $(LARGE_LAYER_TEST_OPTIMIZED_DESCRIPTOR)
	$(MAXJC) $(JFLAGS) $(filter %.java,$^) $(filter %.maxj,$^)
	MAXAPPJCP='.:$(CP)' MAXSOURCEDIRS=$(SOURCE_DIRS) \
		  $(MAXJAVARUN) fpgaconvnet.GenericNetworkBuilder \
		  DFEModel=$(DFEModel) \
		  maxFileName=$(LARGE_LAYER_TEST_NAME) \
		  target="DFE" \
		  descriptor=$(LARGE_LAYER_TEST_OPTIMIZED_DESCRIPTOR) \
		  enableMPCX=$(MPCX)


fpgaconvnet/large_layer_test_dfe.o: $(LARGE_LAYER_TEST_DFE_MAXFILE)
	$(SLICCOMPILE) $< $@


fpgaconvnet/large_layer_test_main_dfe.o: fpgaconvnet/large_layer_test_main.cpp $(LARGE_LAYER_TEST_DFE_MAXFILE)
	$(CPUCOMPILER) $< $(ALL_FLAGS) \
		-I$(MAXCOMPILER_BUILD_DIR) -I$(LARGE_LAYER_TEST_DFEMAXDIR) \
		-DDESIGN_NAME=$(LARGE_LAYER_TEST_NAME) -c -o $@


large_layer_test_dfe: $(FPGACONVNET_OBJ_FILES) \
	    fpgaconvnet/large_layer_test_dfe.o fpgaconvnet/large_layer_test_main_dfe.o \
	    fpgaconvnet/protos/parameters.pb.o
	$(CPUCOMPILER) $(ALL_FLAGS) -o $@ $^ $(LFLAGS)


run_large_layer_test_dfe: large_layer_test_dfe $(LARGE_LAYER_TEST_DATA) $(LARGE_LAYER_TEST_OPTIMIZED_DESCRIPTOR)
	SLIC_CONF="$(SLIC_CONF)" LD_PRELOAD=$(MAXOS_HW) ./$< $(LARGE_LAYER_TEST_OPTIMIZED_DESCRIPTOR)


# ---- Debug ----

debugsim:
	maxdebug -v -g LARGE_LAYER_TESTGraph -s LARGE_LAYER_TESTGraph -c -d $(USER)a0:$(USER)a \
	    $(LARGE_LAYER_TEST_SIM_MAXFILE)

sim: large_layer_test_sim

runsim: run_large_layer_test_sim

dfe: large_layer_test_dfe

rundfe: run_large_layer_test_dfe

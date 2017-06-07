WHEREISROOT=../../..

include $(WHEREISROOT)/Makefile.base

GROUPING_TEST_NAME=grouping_test
GROUPING_TEST_SIMMAXDIR=$(MAXCOMPILER_BUILD_DIR)/$(GROUPING_TEST_NAME)_$(DFEModel)_DFE_SIM/results
GROUPING_TEST_DFEMAXDIR=$(MAXCOMPILER_BUILD_DIR)/$(GROUPING_TEST_NAME)_$(DFEModel)_DFE/results
GROUPING_TEST_SIM_MAXFILE=$(MAXCOMPILER_BUILD_DIR)/$(GROUPING_TEST_NAME)_$(DFEModel)_DFE_SIM/results/$(GROUPING_TEST_NAME).max
GROUPING_TEST_DFE_MAXFILE=$(MAXCOMPILER_BUILD_DIR)/$(GROUPING_TEST_NAME)_$(DFEModel)_DFE/results/$(GROUPING_TEST_NAME).max
GROUPING_TEST_DESCRIPTOR=../descriptors/grouping_test.prototxt


# ==============================
#           Simulation
# ==============================

$(GROUPING_TEST_SIM_MAXFILE): $(ENGINEFILES) $(JAVA_PROTOS)
	$(MAXJC) $(JFLAGS) $(filter %.java,$^) $(filter %.maxj,$^)
	MAXAPPJCP='.:$(CP)' MAXSOURCEDIRS=$(SOURCE_DIRS) \
		  $(MAXJAVARUN) fpgaconvnet.GenericNetworkBuilder \
		  DFEModel=$(DFEModel) \
		  maxFileName=$(GROUPING_TEST_NAME) \
		  target="DFE_SIM" \
		  descriptor=$(GROUPING_TEST_DESCRIPTOR) \
		  enableMPCX=$(MPCX)


grouping_test_sim.o: $(GROUPING_TEST_SIM_MAXFILE)
	$(SLICCOMPILE) $< $@


grouping_test_main_sim.o: main.cpp $(GROUPING_TEST_SIM_MAXFILE)
	$(CPUCOMPILER) $< $(ALL_FLAGS) \
		-I$(MAXCOMPILER_BUILD_DIR) -I$(GROUPING_TEST_SIMMAXDIR) \
		-D__SIM__ -DDESIGN_NAME=$(GROUPING_TEST_NAME) -c -o $@


grouping_test_sim: grouping_test_sim.o grouping_test_main_sim.o \
    $(FPGACONVNET_OBJ_FILES)
	$(CPUCOMPILER) $(ALL_FLAGS) -o $@ $^ $(LFLAGS)


run_grouping_test_sim: grouping_test_sim $(GROUPING_TEST_DATA)
	$(MAXCOMPILERSIM) -n $(USER)a -c$(DEVICENUM) -d$(NUMDEVICES) restart
	SLIC_CONF+="use_simulation=$(USER)a; default_topology_timeout=-1" \
		   LD_PRELOAD=$(MAXOS_SIM) \
		   ./$< $(GROUPING_TEST_DESCRIPTOR)
	make stopsim

# ==============================
#          DFE
# ==============================

$(GROUPING_TEST_DFE_MAXFILE): $(ENGINEFILES) $(JAVA_PROTOS) $(GROUPING_TEST_DESCRIPTOR)
	$(MAXJC) $(JFLAGS) $(filter %.java,$^) $(filter %.maxj,$^)
	MAXAPPJCP='.:$(CP)' MAXSOURCEDIRS=$(SOURCE_DIRS) \
		  $(MAXJAVARUN) fpgaconvnet.GenericNetworkBuilder \
		  DFEModel=$(DFEModel) \
		  maxFileName=$(GROUPING_TEST_NAME) \
		  target="DFE" \
		  descriptor=$(GROUPING_TEST_DESCRIPTOR) \
		  enableMPCX=$(MPCX)


fpgaconvnet/grouping_test_dfe.o: $(GROUPING_TEST_DFE_MAXFILE)
	$(SLICCOMPILE) $< $@


fpgaconvnet/grouping_test_main_dfe.o: main.cpp $(GROUPING_TEST_DFE_MAXFILE)
	$(CPUCOMPILER) $< $(ALL_FLAGS) \
		-I$(MAXCOMPILER_BUILD_DIR) -I$(GROUPING_TEST_DFEMAXDIR) \
		-DDESIGN_NAME=$(GROUPING_TEST_NAME) -c -o $@


grouping_test_dfe: $(FPGACONVNET_OBJ_FILES) \
	    fpgaconvnet/grouping_test_dfe.o \
	    grouping_test_main_dfe.o \
	    fpgaconvnet/protos/parameters.pb.o
	$(CPUCOMPILER) $(ALL_FLAGS) -o $@ $^ $(LFLAGS)


run_grouping_test_dfe: grouping_test_dfe $(GROUPING_TEST_DATA) $(GROUPING_TEST_DESCRIPTOR)
	SLIC_CONF="$(SLIC_CONF)" LD_PRELOAD=$(MAXOS_HW) ./$< $(GROUPING_TEST_DESCRIPTOR)


# ---- Debug ----

debugsim:
	maxdebug -v -g GROUPING_TESTGraph -s GROUPING_TESTGraph -c -d $(USER)a0:$(USER)a \
	    $(GROUPING_TEST_SIM_MAXFILE)

sim: grouping_test_sim

runsim: run_grouping_test_sim

dfe: grouping_test_dfe

rundfe: run_grouping_test_dfe

WHEREISROOT=../../..

include $(WHEREISROOT)/Makefile.base

ASPECT_IO_TEST_NAME=aspect_io_test
ASPECT_IO_TEST_SIMMAXDIR=$(MAXCOMPILER_BUILD_DIR)/$(ASPECT_IO_TEST_NAME)_$(DFEModel)_DFE_SIM/results
ASPECT_IO_TEST_DFEMAXDIR=$(MAXCOMPILER_BUILD_DIR)/$(ASPECT_IO_TEST_NAME)_$(DFEModel)_DFE/results
ASPECT_IO_TEST_SIM_MAXFILE=$(MAXCOMPILER_BUILD_DIR)/$(ASPECT_IO_TEST_NAME)_$(DFEModel)_DFE_SIM/results/$(ASPECT_IO_TEST_NAME).max
ASPECT_IO_TEST_DFE_MAXFILE=$(MAXCOMPILER_BUILD_DIR)/$(ASPECT_IO_TEST_NAME)_$(DFEModel)_DFE/results/$(ASPECT_IO_TEST_NAME).max
ASPECT_IO_TEST_DESCRIPTOR=../descriptors/aspect_io_test.prototxt


# ==============================
#           Simulation
# ==============================

$(ASPECT_IO_TEST_SIM_MAXFILE): $(ENGINEFILES) $(JAVA_PROTOS) $(ASPECT_IO_TEST_DESCRIPTOR)
	$(MAXJC) $(JFLAGS) $(filter %.java,$^) $(filter %.maxj,$^)
	MAXAPPJCP='.:$(CP)' MAXSOURCEDIRS=$(SOURCE_DIRS) \
		  $(MAXJAVARUN) fpgaconvnet.GenericNetworkBuilder \
		  DFEModel=$(DFEModel) \
		  maxFileName=$(ASPECT_IO_TEST_NAME) \
		  target="DFE_SIM" \
		  descriptor=$(ASPECT_IO_TEST_DESCRIPTOR) \
		  enableMPCX=$(MPCX)


aspect_io_test_sim.o: $(ASPECT_IO_TEST_SIM_MAXFILE)
	$(SLICCOMPILE) $< $@


aspect_io_test_main_sim.o: main.cpp $(ASPECT_IO_TEST_SIM_MAXFILE)
	$(CPUCOMPILER) $< $(ALL_FLAGS) \
		-I$(MAXCOMPILER_BUILD_DIR) -I$(ASPECT_IO_TEST_SIMMAXDIR) \
		-D__SIM__ -DDESIGN_NAME=$(ASPECT_IO_TEST_NAME) -c -o $@


aspect_io_test_sim: $(FPGACONVNET_OBJ_FILES) \
	    aspect_io_test_sim.o aspect_io_test_main_sim.o
	$(CPUCOMPILER) $(ALL_FLAGS) -o $@ $^ $(LFLAGS)


run_aspect_io_test_sim: aspect_io_test_sim $(ASPECT_IO_TEST_DATA) $(ASPECT_IO_TEST_DESCRIPTOR)
	$(MAXCOMPILERSIM) -n $(USER)a -c$(DEVICENUM) -d$(NUMDEVICES) restart
	SLIC_CONF+="use_simulation=$(USER)a; default_topology_timeout=-1" \
		   LD_PRELOAD=$(MAXOS_SIM) \
		   ./$< $(ASPECT_IO_TEST_DESCRIPTOR)
	make stopsim

# ==============================
#          DFE
# ==============================

$(ASPECT_IO_TEST_DFE_MAXFILE): $(ENGINEFILES) $(JAVA_PROTOS) $(ASPECT_IO_TEST_DESCRIPTOR)
	$(MAXJC) $(JFLAGS) $(filter %.java,$^) $(filter %.maxj,$^)
	MAXAPPJCP='.:$(CP)' MAXSOURCEDIRS=$(SOURCE_DIRS) \
		  $(MAXJAVARUN) fpgaconvnet.GenericNetworkBuilder \
		  DFEModel=$(DFEModel) \
		  maxFileName=$(ASPECT_IO_TEST_NAME) \
		  target="DFE" \
		  descriptor=$(ASPECT_IO_TEST_DESCRIPTOR) \
		  enableMPCX=$(MPCX)


fpgaconvnet/aspect_io_test_dfe.o: $(ASPECT_IO_TEST_DFE_MAXFILE)
	$(SLICCOMPILE) $< $@


fpgaconvnet/aspect_io_test_main_dfe.o: fpgaconvnet/aspect_io_test_main.cpp $(ASPECT_IO_TEST_DFE_MAXFILE)
	$(CPUCOMPILER) $< $(ALL_FLAGS) \
		-I$(MAXCOMPILER_BUILD_DIR) -I$(ASPECT_IO_TEST_DFEMAXDIR) \
		-DDESIGN_NAME=$(ASPECT_IO_TEST_NAME) -c -o $@


aspect_io_test_dfe: $(FPGACONVNET_OBJ_FILES) \
	    fpgaconvnet/aspect_io_test_dfe.o fpgaconvnet/aspect_io_test_main_dfe.o \
	    fpgaconvnet/protos/parameters.pb.o
	$(CPUCOMPILER) $(ALL_FLAGS) -o $@ $^ $(LFLAGS)


run_aspect_io_test_dfe: aspect_io_test_dfe $(ASPECT_IO_TEST_DATA) $(ASPECT_IO_TEST_DESCRIPTOR)
	SLIC_CONF="$(SLIC_CONF)" LD_PRELOAD=$(MAXOS_HW) ./$< $(ASPECT_IO_TEST_DESCRIPTOR)


# ---- Debug ----

debug:
	maxdebug -v -g ASPECT_IO_TESTGraph -s ASPECT_IO_TESTGraph -c -d $(USER)a0:$(USER)a \
	    ASPECT_IO_TEST_SIM_MAXFILE

sim: aspect_io_test_sim

runsim: run_aspect_io_test_sim

dfe: aspect_io_test_dfe

rundfe: run_aspect_io_test_dfe

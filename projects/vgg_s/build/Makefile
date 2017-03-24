WHEREISROOT=../../..

include $(WHEREISROOT)/Makefile.base

PRJ_NAME=vgg_s
PRJ_DESCRIPTOR=../descriptors/vgg_s.prototxt
PRJ_OPTIMIZED_DESCRIPTOR=vgg_s.optimized.prototxt


$(PRJ_OPTIMIZED_DESCRIPTOR): fpgaconvnet/optimizer.py \
    			     $(PRJ_DESCRIPTOR) \
			     $(PYTHON_PROTOS) \
			     sync_py_srcs
	python $< --resource-bench $(WHEREISROOT)/results/resource_bench.yaml \
	    --output $@ $(PRJ_DESCRIPTOR)

gen_makefile: $(WHEREISROOT)/scripts/generate_multi_fpga_makefile.py $(PRJ_OPTIMIZED_DESCRIPTOR)
	python $<


ifeq ($(shell test -e Makefile.targets && echo -n yes), yes)

include Makefile.targets

TARGET_SIMMAXDIRS := $(foreach name,$(TARGET_NAMES),$(MAXCOMPILER_BUILD_DIR)/$(name)_$(DFEModel)_DFE_SIM/results)
TARGET_SIM_MAXFILES := $(foreach name,$(TARGET_NAMES),$(MAXCOMPILER_BUILD_DIR)/$(name)_$(DFEModel)_DFE_SIM/results/$(name).max)
TARGET_SIM_OBJS := $(foreach name,$(TARGET_NAMES),fpgaconvnet/$(name).sim.o)
TARGET_DFEMAXDIRS := $(foreach name,$(TARGET_NAMES),$(MAXCOMPILER_BUILD_DIR)/$(name)_$(DFEModel)_DFE/results)
TARGET_DFE_MAXFILES := $(foreach name,$(TARGET_NAMES),$(MAXCOMPILER_BUILD_DIR)/$(name)_$(DFEModel)_DFE/results/$(name).max)
TARGET_DFE_OBJS := $(foreach name,$(TARGET_NAMES),fpgaconvnet/$(name).dfe.o)

# ==============================
#           Simulation
# ==============================

$(TARGET_SIM_MAXFILES): $(ENGINEFILES) $(JAVA_PROTOS)
	$(MAXJC) $(JFLAGS) $^
	MAXAPPJCP='.:$(CP)' MAXSOURCEDIRS=$(SOURCE_DIRS) \
		  $(MAXJAVARUN) fpgaconvnet.GenericNetworkBuilder \
		  DFEModel=$(DFEModel) \
		  maxFileName=$(basename $(notdir $@)) \
		  target="DFE_SIM" \
                  fpga_id="$(word 1 $(substr _, ,$(notdir $(basename $@))))" \
		  descriptor="../descriptors/target/$(basename $(notdir $@)).prototxt" \
		  enableMPCX=$(MPCX)


$(TARGET_SIM_OBJS): $(TARGET_SIM_MAXFILES)
	$(SLICCOMPILE) $(MAXCOMPILER_BUILD_DIR)/$(basename $(basename $(notdir $@)))_$(DFEModel)_DFE_SIM/results/$(basename $(basename $(notdir $@))).max $@


fpgaconvnet/target_main_sim.o: fpgaconvnet/target_main.cpp $(TARGET_SIM_MAXFILES) target_sim_lookup.h
	$(CPUCOMPILER) $< $(ALL_FLAGS) \
		-I$(MAXCOMPILER_BUILD_DIR) \
		$(foreach name,$(TARGET_SIMMAXDIRS),-I$(name)) \
		-D__SIM__ -c -o $@


target_sim: fpgaconvnet/convnet.o \
		fpgaconvnet/target_main_sim.o \
		fpgaconvnet/protos/parameters.pb.o \
		$(TARGET_SIM_OBJS)
	$(CPUCOMPILER) $(ALL_FLAGS) -o $@ $^ $(LFLAGS)


run_target_sim: target_sim
	$(MAXCOMPILERSIM) -n $(USER)a -c$(DEVICENUM) -d$(NUMDEVICES) restart
	SLIC_CONF+="use_simulation=$(USER)a; default_topology_timeout=-1" \
		   LD_PRELOAD=$(MAXOS_SIM) \
		   ./$< $(TARGET_NAMES)
	make stopsim

# ==============================
#          DFE
# ==============================

target_dfe_lookup.h: $(TARGET_DFE_MAXFILES)
	python ../scripts/generate_target_lookup.py $@ $(TARGET_NAMES)


$(TARGET_DFE_MAXFILES): $(ENGINEFILES) $(JAVA_PROTOS)
	$(MAXJC) $(JFLAGS) $^
	MAXAPPJCP='.:$(CP)' MAXSOURCEDIRS=$(SOURCE_DIRS) \
		  $(MAXJAVARUN) fpgaconvnet.GenericNetworkBuilder \
		  DFEModel=$(DFEModel) \
		  maxFileName=$(basename $(notdir $@)) \
		  target="DFE" \
		  descriptor="../descriptors/target/$(basename $(notdir $@)).prototxt" \
		  enableMPCX=$(MPCX)


$(TARGET_DFE_OBJS): $(TARGET_DFE_MAXFILES)
	$(SLICCOMPILE) $(MAXCOMPILER_BUILD_DIR)/$(basename $(basename $(notdir $@)))_$(DFEModel)_DFE/results/$(basename $(basename $(notdir $@))).max $@


fpgaconvnet/target_main_dfe.o: fpgaconvnet/target_main.cpp $(TARGET_DFE_MAXFILES) target_dfe_lookup.h
	$(CPUCOMPILER) $< $(ALL_FLAGS) \
		-I$(MAXCOMPILER_BUILD_DIR) \
		$(foreach name,$(TARGET_DFEMAXDIRS),-I$(name)) \
		-c -o $@


target_dfe: fpgaconvnet/convnet.o \
		fpgaconvnet/target_main_dfe.o \
		fpgaconvnet/protos/parameters.pb.o \
		$(TARGET_DFE_OBJS)
	$(CPUCOMPILER) $(ALL_FLAGS) -o $@ $^ $(LFLAGS)


run_target_dfe: target_dfe
	SLIC_CONF="$(SLIC_CONF)" LD_PRELOAD=$(MAXOS_HW) ./$< $(TARGET_NAMES)
endif

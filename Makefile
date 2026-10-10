# ColorSplit: build and test.
#
#   make            CPU version (C)                    -> bin/colorsplit
#   make gpu        GPU version (needs nvcc)            -> bin/colorsplit_gpu
#   make gpu-v2     archival GPU version of the paper   -> bin/colorsplit_gpu_v2
#   make emu        GPU code emulated on the CPU        -> bin/colorsplit_emu
#   make test       quick self-test (C, Java if installed, GPU emulation, tools)
#   make check-reductions   the reductions leave every table cell and chunk sum unchanged
#   make clean
#
# On AMD Zen 2 and older CPUs, build with:  make CPU_EXTRA=-DSOFT_PEXT
# For a GPU other than the one in this machine, e.g. an RTX 4090:  make gpu GPU_ARCH=-arch=sm_89

CC        ?= gcc
CXX       ?= g++
NVCC      ?= nvcc
CFLAGS    ?= -O3 -march=native
CPU_EXTRA ?=
GPU_ARCH  ?= -arch=native
BIN       := bin

.PHONY: all cpu gpu gpu-v2 emu test check-reductions clean

all: cpu
cpu: $(BIN)/colorsplit
gpu: $(BIN)/colorsplit_gpu
gpu-v2: $(BIN)/colorsplit_gpu_v2
emu: $(BIN)/colorsplit_emu

$(BIN):
	mkdir -p $(BIN)

$(BIN)/colorsplit: cpu/colorsplit.c | $(BIN)
	$(CC) $(CFLAGS) $(CPU_EXTRA) -pthread -o $@ $<

$(BIN)/colorsplit_gpu: gpu/colorsplit.cu | $(BIN)
	$(NVCC) -O3 $(GPU_ARCH) -o $@ $<

# the settings of the record runs: prefix depth 4, run with -t 512 -batch 32
$(BIN)/colorsplit_gpu_v2: gpu/colorsplit_v2.cu | $(BIN)
	$(NVCC) -O3 $(GPU_ARCH) -DPFX=4 -o $@ $<

$(BIN)/colorsplit_emu: gpu/colorsplit.cu | $(BIN)
	$(CXX) -O2 -x c++ -DEMULATE -o $@ $<

test: cpu emu
	bash tests/run_tests.sh

check-reductions:
	bash tests/check_reductions.sh

clean:
	rm -rf $(BIN)

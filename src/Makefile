# Makefile
GPU_BACKEND ?= auto
CXXSTD      := c++14
LIBS        := -libverbs -lpthread -ldl
INCLUDES    := -I/usr/include -I/usr/include/infiniband -I.
LDFLAGS     := -L/usr/lib/x86_64-linux-gnu

ifeq ($(GPU_BACKEND),auto)
ifneq ($(shell command -v nvcc 2>/dev/null),)
GPU_BACKEND := cuda
else ifneq ($(shell command -v hipcc 2>/dev/null),)
GPU_BACKEND := hip
else
$(error Neither nvcc nor hipcc was found in PATH)
endif
endif

ifeq ($(GPU_BACKEND),cuda)
CUDA_ARCH   ?= compute_80
SM_ARCH     ?= sm_80
GPUCC       := nvcc
GPUCCFLAGS  := -ccbin /usr/bin/g++ -O3 -std=$(CXXSTD) $(INCLUDES) \
               -gencode arch=$(CUDA_ARCH),code=$(SM_ARCH) \
               -gencode arch=$(CUDA_ARCH),code=$(CUDA_ARCH)
else ifeq ($(GPU_BACKEND),hip)
HIP_ARCH    ?=
GPUCC       := hipcc
GPUCCFLAGS  := -O3 -std=$(CXXSTD) $(INCLUDES)
ifneq ($(strip $(HIP_ARCH)),)
GPUCCFLAGS  += --offload-arch=$(HIP_ARCH)
endif
else
$(error Unsupported GPU_BACKEND='$(GPU_BACKEND)'; use cuda, hip, or auto)
endif

CXX          := g++
CXXFLAGS     := -O3 -std=$(CXXSTD) $(INCLUDES)
CXXLDFLAGS   := $(LIBS)

GPU_BINARIES  := gpu_server gpu_memhog gpu_pcie_memcpy
CPU_BINARIES  := cpu_client
OPTIONAL_BINS :=

ifneq ($(wildcard gpu_be_memhog_minimal_task.cu),)
OPTIONAL_BINS += gpu_minimal_memhog
endif

ifneq ($(wildcard gpu_be_pure_computation_task.cu),)
OPTIONAL_BINS += gpu_pure_computation
endif

ALL_BINARIES := $(GPU_BINARIES) $(CPU_BINARIES) $(OPTIONAL_BINS)

all: clean $(ALL_BINARIES) release

gpu_server: gpu_server.cu gpu_rt.h
	$(GPUCC) $< -o $@ $(GPUCCFLAGS) $(LDFLAGS) $(LIBS)

cpu_client: cpu_client.cc
	$(CXX) $< -o $@ $(CXXLDFLAGS) $(CXXFLAGS)

ifneq ($(wildcard gpu_be_memhog_minimal_task.cu),)
gpu_minimal_memhog: gpu_be_memhog_minimal_task.cu gpu_rt.h
	$(GPUCC) $< -o $@ $(GPUCCFLAGS) $(LDFLAGS) $(LIBS)
endif

gpu_memhog: gpu_be_memhog_task.cu gpu_rt.h
	$(GPUCC) $< -o $@ $(GPUCCFLAGS) $(LDFLAGS) $(LIBS)

ifneq ($(wildcard gpu_be_pure_computation_task.cu),)
gpu_pure_computation: gpu_be_pure_computation_task.cu gpu_rt.h
	$(GPUCC) $< -o $@ $(GPUCCFLAGS) $(LDFLAGS) $(LIBS)
endif

gpu_pcie_memcpy: gpu_be_pcie_memcpy_task.cu gpu_rt.h
	$(GPUCC) $< -o $@ $(GPUCCFLAGS) $(LDFLAGS) $(LIBS)

clean:
	rm -f gpu_server cpu_client gpu_minimal_memhog gpu_memhog gpu_pure_computation gpu_pcie_memcpy

release:
	mkdir -p bin
	mv $(ALL_BINARIES) ./bin

# Makefile
CUDA_ARCH   := compute_80
SM_ARCH     := sm_80
CXXSTD      := c++14
NVCC        := nvcc
LIBS        := -libverbs -lpthread
INCLUDES    := -I/usr/include -I/usr/include/infiniband
LDFLAGS     := -L/usr/lib/x86_64-linux-gnu
NVCCFLAGS   := -O3 -std=$(CXXSTD) $(INCLUDES) \
               -gencode arch=$(CUDA_ARCH),code=$(SM_ARCH) \
               -gencode arch=$(CUDA_ARCH),code=$(CUDA_ARCH)

CXX          := g++
CXXFLAGS     := -O3 -std=$(CXXSTD) $(INCLUDES)
CXXLDFLAGS   := $(LIBS)


all: clean gpu_server cpu_client gpu_minimal_memhog gpu_memhog release

gpu_server: gpu_server.cu
	$(NVCC) $< -o $@ $(NVCCFLAGS) $(LDFLAGS) $(LIBS)

cpu_client: cpu_client.cc
	$(CXX) $< -o $@ $(CXXLDFLAGS) $(CXXFLAGS)

gpu_minimal_memhog: gpu_be_memhog_minimal_task.cu
	$(NVCC) $< -o $@ $(NVCCFLAGS) $(LDFLAGS) $(LIBS)

gpu_memhog: gpu_be_memhog_task.cu
	$(NVCC) $< -o $@ $(NVCCFLAGS) $(LDFLAGS) $(LIBS)

clean:
	rm -f gpu_server cpu_client gpu_minimal_memhog gpu_memhog

release:
	mkdir bin
	mv gpu_server cpu_client gpu_minimal_memhog gpu_memhog ./bin

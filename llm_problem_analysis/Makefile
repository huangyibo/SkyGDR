# Makefile for NVLink + decode HBM/KV-cache contention microbenchmarks.
#
# Build:
#   make
#
# Clean:
#   make clean
#
# Override CUDA compiler:
#   make NVCC=/usr/local/cuda/bin/nvcc
#
# Override GPU architecture:
#   make CUDA_ARCH=sm_80
#   make CUDA_ARCH=sm_90

NVCC ?= nvcc

CUDA_ARCH ?= native

CXXFLAGS := -O3 -std=c++17

ifeq ($(CUDA_ARCH),native)
  ARCH_FLAGS :=
else
  ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif

NVCCFLAGS := $(CXXFLAGS) $(ARCH_FLAGS)

TARGETS := \
	kv_nvlink_transfer \
	decode_hbm_stress \
	kv_nvlink_transfer_trace \
	decode_transformer_kvcache_stress \
	decode_vllm_kvcache_hbm_stress

.PHONY: all clean help

all: $(TARGETS)

kv_nvlink_transfer: kv_nvlink_transfer.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

decode_hbm_stress: decode_hbm_stress.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

kv_nvlink_transfer_trace: kv_nvlink_transfer_trace.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

decode_transformer_kvcache_stress: decode_transformer_kvcache_stress.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

decode_vllm_kvcache_hbm_stress: decode_vllm_kvcache_hbm_stress.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(TARGETS) *.o *.log

help:
	@echo "CUDA NVLink + decode HBM contention benchmarks"
	@echo ""
	@echo "Targets:"
	@echo "  make        Build all programs"
	@echo "  make clean  Remove binaries and logs"
	@echo ""
	@echo "Variables:"
	@echo "  NVCC        CUDA compiler, default: nvcc"
	@echo "  CUDA_ARCH   GPU arch, default: native"
	@echo ""
	@echo "Examples:"
	@echo "  make"
	@echo "  make CUDA_ARCH=sm_80"
	@echo "  make CUDA_ARCH=sm_90"
	@echo "  make NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_80"

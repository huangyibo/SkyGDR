#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define GPU_RT_IS_HIP 1
#define GPU_RT_IS_CUDA 0
#else
#define GPU_RT_IS_HIP 0
#define GPU_RT_IS_CUDA 1
#endif

#if GPU_RT_IS_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#define GPU_RT_BACKEND_NAME "CUDA"
#define gpuSuccess cudaSuccess
#define gpuError_t cudaError_t
#define gpuGetErrorString cudaGetErrorString
#define gpuStream_t cudaStream_t
#define gpuStreamNonBlocking cudaStreamNonBlocking
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamCreateWithFlags cudaStreamCreateWithFlags
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuStreamDestroy cudaStreamDestroy
#define gpuDeviceProp cudaDeviceProp
#define gpuSetDevice cudaSetDevice
#define gpuDeviceMapHost cudaDeviceMapHost
#define gpuSetDeviceFlags cudaSetDeviceFlags
#define gpuGetDevice cudaGetDevice
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuDeviceGetPCIBusId cudaDeviceGetPCIBusId
#define gpuDeviceCanAccessPeer cudaDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess cudaDeviceEnablePeerAccess
#define gpuIpcMemHandle_t cudaIpcMemHandle_t
#define gpuIpcMemLazyEnablePeerAccess cudaIpcMemLazyEnablePeerAccess
#define gpuIpcOpenMemHandle cudaIpcOpenMemHandle
#define gpuIpcGetMemHandle cudaIpcGetMemHandle
#define gpuIpcCloseMemHandle cudaIpcCloseMemHandle
#define gpuHostMalloc cudaMallocHost  // no cudaHostMalloc API in CUDA
#define gpuHostAlloc cudaHostAlloc
#define gpuHostAllocMapped cudaHostAllocMapped
#define gpuMalloc cudaMalloc
#define gpuMallocAsync cudaMallocAsync
#define gpuMallocHost cudaMallocHost
#define gpuFree cudaFree
#define gpuFreeAsync cudaFreeAsync
#define gpuFreeHost cudaFreeHost
#define gpuMemset cudaMemset
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyPeerAsync cudaMemcpyPeerAsync
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpyFromSymbol cudaMemcpyFromSymbol
#define gpuMemsetAsync cudaMemsetAsync
#define gpuGetLastError cudaGetLastError
#define gpuErrorPeerAccessAlreadyEnabled cudaErrorPeerAccessAlreadyEnabled
#define gpuErrorNotReady cudaErrorNotReady
#define gpuErrorInvalidValue cudaErrorInvalidValue
#define gpuErrorNotSupported cudaErrorNotSupported
#define gpuEvent_t cudaEvent_t
#define gpuEventCreate cudaEventCreate
#define gpuEventDestroy cudaEventDestroy
#define gpuEventRecord cudaEventRecord
#define gpuEventQuery cudaEventQuery
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuStreamWaitEvent cudaStreamWaitEvent
#define gpuEventCreateWithFlags cudaEventCreateWithFlags
#define gpuEventDefault cudaEventDefault
#define gpuEventDisableTiming cudaEventDisableTiming
#define gpuEventInterprocess cudaEventInterprocess
#define gpuIpcEventHandle_t cudaIpcEventHandle_t
#define gpuIpcGetEventHandle cudaIpcGetEventHandle
#define gpuIpcOpenEventHandle cudaIpcOpenEventHandle
// DMA-BUF / GPU driver types for GPUDirect RDMA
#define gpuDriverResult_t CUresult
#define gpuDevicePtr_t CUdeviceptr
#define gpuDriverSuccess CUDA_SUCCESS
#define gpuMemRangeHandleType CUmemRangeHandleType
#define GPU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD
#define gpuPointerAttribute_t cudaPointerAttributes
#define gpuPointerGetAttributes cudaPointerGetAttributes
#define gpuMemoryTypeDevice cudaMemoryTypeDevice
#define GPU_DRIVER_LIB_NAME "libcuda.so.1"
#define GPU_DRIVER_LIB_NAME_FALLBACK "libcuda.so"
#define GPU_DRIVER_GET_HANDLE_FOR_ADDRESS_RANGE_NAME \
  "cuMemGetHandleForAddressRange"
inline gpuError_t gpuMemGetAddressRange(void** base_ptr, size_t* size,
                                        void* ptr) {
  CUdeviceptr base;
  CUresult result = cuMemGetAddressRange(&base, size, (CUdeviceptr)ptr);
  if (result == CUDA_SUCCESS) {
    *base_ptr = (void*)base;
    return gpuSuccess;
  }
  return gpuError_t(result);
}
#else
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#define GPU_RT_BACKEND_NAME "HIP"
#define gpuSuccess hipSuccess
#define gpuError_t hipError_t
#define gpuGetErrorString hipGetErrorString
#define gpuStream_t hipStream_t
#define gpuStreamNonBlocking hipStreamNonBlocking
#define gpuStreamCreate hipStreamCreate
#define gpuStreamCreateWithFlags hipStreamCreateWithFlags
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuStreamDestroy hipStreamDestroy
#define gpuSetDevice hipSetDevice
#define gpuDeviceMapHost hipDeviceMapHost
#define gpuSetDeviceFlags hipSetDeviceFlags
#define gpuGetDevice hipGetDevice
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuDeviceProp hipDeviceProp_t
#define gpuDeviceGetPCIBusId hipDeviceGetPCIBusId
#define gpuDeviceCanAccessPeer hipDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define gpuIpcMemHandle_t hipIpcMemHandle_t
#define gpuIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess
#define gpuIpcOpenMemHandle hipIpcOpenMemHandle
#define gpuIpcGetMemHandle hipIpcGetMemHandle
#define gpuIpcCloseMemHandle hipIpcCloseMemHandle
#define gpuHostMalloc hipHostMalloc
#define gpuHostAlloc hipHostAlloc
#define gpuHostAllocDefault hipHostAllocDefault
#define gpuHostFree hipHostFree
#define gpuHostAllocMapped hipHostAllocMapped
#define gpuMalloc hipMalloc
#define gpuMallocAsync hipMallocAsync
#define gpuMallocHost hipHostMalloc  // cudaMallocHost Deprecated in ROCm
#define gpuFree hipFree
#define gpuFreeAsync hipFreeAsync
#define gpuFreeHost hipFreeHost
#define gpuMemset hipMemset
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpy hipMemcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyPeerAsync hipMemcpyPeerAsync
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyFromSymbol hipMemcpyFromSymbol
#define gpuMemsetAsync hipMemsetAsync
#define gpuGetLastError hipGetLastError
#define gpuErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define gpuErrorNotReady hipErrorNotReady
#define gpuErrorInvalidValue hipErrorInvalidValue
#define gpuErrorNotSupported hipErrorNotSupported
#define gpuEvent_t hipEvent_t
#define gpuEventCreate hipEventCreate
#define gpuEventDestroy hipEventDestroy
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventQuery hipEventQuery
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuStreamWaitEvent hipStreamWaitEvent
#define gpuEventCreateWithFlags hipEventCreateWithFlags
#define gpuEventDefault hipEventDefault
#define gpuEventDisableTiming hipEventDisableTiming
#define gpuEventInterprocess hipEventInterprocess
#define gpuIpcEventHandle_t hipIpcEventHandle_t
#define gpuIpcGetEventHandle hipIpcGetEventHandle
#define gpuIpcOpenEventHandle hipIpcOpenEventHandle
// DMA-BUF / GPU driver types for GPUDirect RDMA
#define gpuDriverResult_t hipError_t
#define gpuDevicePtr_t hipDeviceptr_t
#define gpuDriverSuccess hipSuccess
#define gpuMemRangeHandleType hipMemHandleType
#define GPU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD hipMemHandleTypeGeneric
#define gpuPointerAttribute_t hipPointerAttribute_t
#define gpuPointerGetAttributes hipPointerGetAttributes
#define gpuMemoryTypeDevice hipMemoryTypeDevice
#define GPU_DRIVER_LIB_NAME "libamdhip64.so"
#define GPU_DRIVER_LIB_NAME_FALLBACK "libamdhip64.so"
#define GPU_DRIVER_GET_HANDLE_FOR_ADDRESS_RANGE_NAME \
  "hipMemGetHandleForAddressRange"
#define gpuMemGetAddressRange hipMemGetAddressRange
#endif

inline gpuError_t gpuIpcCloseEventHandle(gpuEvent_t) { return gpuSuccess; }

#if GPU_RT_IS_CUDA
#define gpuDeviceSynchronize cudaDeviceSynchronize
#else
#define gpuDeviceSynchronize hipDeviceSynchronize
#endif

#if GPU_RT_IS_CUDA
#define gpuHostAllocDefault cudaHostAllocDefault
#endif

#if GPU_RT_IS_CUDA
#define GPU_RT_LAUNCH_KERNEL(kernel, grid, block, shared_mem, stream, ...) \
  kernel<<<(grid), (block), (shared_mem), (stream)>>>(__VA_ARGS__)
#else
#define GPU_RT_LAUNCH_KERNEL(kernel, grid, block, shared_mem, stream, ...) \
  hipLaunchKernelGGL(kernel, dim3((grid)), dim3((block)), (shared_mem),    \
                     (stream), __VA_ARGS__)
#endif

// Function pointer type for DMA-BUF handle export (loaded via dlsym)
typedef gpuDriverResult_t (*gpuMemGetHandleForAddressRange_fn)(
    void*, gpuDevicePtr_t, size_t, gpuMemRangeHandleType, unsigned long long);

inline const char* gpuRuntimeName() { return GPU_RT_BACKEND_NAME; }

inline gpuMemGetHandleForAddressRange_fn
gpuLoadMemGetHandleForAddressRangeFn() {
  static gpuMemGetHandleForAddressRange_fn fn = []() {
    void* handle = dlopen(GPU_DRIVER_LIB_NAME, RTLD_NOW | RTLD_LOCAL);
    if (!handle && std::strcmp(GPU_DRIVER_LIB_NAME, GPU_DRIVER_LIB_NAME_FALLBACK) !=
                       0) {
      handle = dlopen(GPU_DRIVER_LIB_NAME_FALLBACK, RTLD_NOW | RTLD_LOCAL);
    }
    if (!handle) return (gpuMemGetHandleForAddressRange_fn) nullptr;
    return reinterpret_cast<gpuMemGetHandleForAddressRange_fn>(dlsym(
        handle, GPU_DRIVER_GET_HANDLE_FOR_ADDRESS_RANGE_NAME));
  }();
  return fn;
}

inline bool gpuSupportsDmabufExport() {
  return gpuLoadMemGetHandleForAddressRangeFn() != nullptr;
}

inline bool gpuExportDmabufFd(void* ptr, size_t len, int* dmabuf_fd,
                              uint64_t* offset = nullptr,
                              void** alloc_base = nullptr,
                              size_t* alloc_size = nullptr) {
  if (!ptr || !dmabuf_fd || len == 0) return false;

  void* base_ptr = nullptr;
  size_t total_size = 0;
  if (gpuMemGetAddressRange(&base_ptr, &total_size, ptr) != gpuSuccess ||
      !base_ptr || total_size == 0) {
    return false;
  }

  const uintptr_t base = reinterpret_cast<uintptr_t>(base_ptr);
  const uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
  if (start < base || len > total_size || start - base > total_size - len) {
    return false;
  }

  auto fn = gpuLoadMemGetHandleForAddressRangeFn();
  if (!fn) return false;

  int fd = -1;
  gpuDriverResult_t drv_err = fn(&fd, (gpuDevicePtr_t)base_ptr, total_size,
                                 GPU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0ULL);
  if (drv_err != gpuDriverSuccess || fd < 0) return false;

  *dmabuf_fd = fd;
  if (offset) *offset = (uint64_t)(start - base);
  if (alloc_base) *alloc_base = base_ptr;
  if (alloc_size) *alloc_size = total_size;
  return true;
}

inline gpuError_t gpuFlushRemoteWritesToDevice() {
#if GPU_RT_IS_CUDA
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11030)
  return cudaDeviceFlushGPUDirectRDMAWrites(
      cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
      cudaFlushGPUDirectRDMAWritesToOwner);
#else
  return gpuErrorNotSupported;
#endif
#else
  return gpuErrorNotSupported;
#endif
}

#define GPU_RT_CHECK(call)                                         \
  do {                                                             \
    gpuError_t err__ = (call);                                     \
    if (err__ != gpuSuccess) {                                     \
      fprintf(stderr, "GPU error %s:%d: %s\n", __FILE__, __LINE__, \
              gpuGetErrorString(err__));                           \
      std::abort();                                                \
    }                                                              \
  } while (0)

#define GPU_RT_CHECK_ERRORS(msg)                              \
  do {                                                        \
    gpuError_t __err = gpuGetLastError();                     \
    if (__err != gpuSuccess) {                                \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, \
              gpuGetErrorString(__err), __FILE__, __LINE__);  \
      fprintf(stderr, "*** FAILED - ABORTING\n");             \
      exit(1);                                                \
    }                                                         \
  } while (0)

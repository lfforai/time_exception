#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "cuda_fp16.h"
#include "kernel_example.h"
#include "sm_32_intrinsics.h"
//#include "tensorflow/core/util/gpu_kernel_helper.h"

template<typename T>
__device__ __forceinline__ T ldg(const T* ptr) {
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

//template __device__ __forceinline__ float ldg<float>(const float* ptr);
//template __device__ __forceinline__ int ldg<int>(const int* ptr);

template <typename T>
struct ExampleFunctor<GPUDevice,T>{
  ExampleFunctor(){};
  void operator()(const GPUDevice& d, int size, const T* in, T* out);
};

//Define the CUDA kernel.
template <typename T>
__global__ void ExampleCudaKernel(const int size, const T* in, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    out[i] = 3 * ldg(in+i);//ldg
  }
}

//Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void ExampleFunctor<GPUDevice,T>::operator()(
    const GPUDevice& d, int size, const T* in, T* out) {

  int block_count = 1024;
  int thread_per_block = 20;
  ExampleCudaKernel<T><<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
}

template struct ExampleFunctor<GPUDevice,float>;
template struct ExampleFunctor<GPUDevice,int>;
#endif  // GOOGLE_CUDA



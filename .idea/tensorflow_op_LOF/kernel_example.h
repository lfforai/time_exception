#ifndef KERNEL_EXAMPLE_H_
#define KERNEL_EXAMPLE_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename Device,typename T>
struct ExampleFunctor{
  ExampleFunctor(){};
  void operator()(const Device& d, int size, const T* in, T* out);
};
#endif //KERNEL_EXAMPLE_H_

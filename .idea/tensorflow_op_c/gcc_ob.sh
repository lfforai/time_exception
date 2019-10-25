#!/bin/bash
echo "ggc  tensorflow op on gpu!"

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
 
#sleep 2

nvcc -std=c++11 -c -o kernel_example.cu.o kernel_example.cu.cc \
-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC \
-L /usr/local/cuda-10.0/lib64/ \
-I /usr/local/cuda-10.0/include/ \
-L /root/anaconda3/lib/python3.7/site-packages/tensorflow_core \
-l:libtensorflow_framework.so.2 \
--expt-relaxed-constexpr \
-gencode arch=compute_61,code=sm_61
echo "frist nvcc finshed"


#sleep 3

g++ -std=c++11 -shared -o kernel_example.so \
-D_GLIBCXX_USE_CXX11_ABI=0 \
kernel_example.cu.o kernel_example.cc -I $TF_INC -fPIC -lcudart \
-L /usr/local/cuda-10.0/lib64/ \
-L /root/anaconda3/lib/python3.7/site-packages/tensorflow_core \
-l:libtensorflow_framework.so.2
echo "second gcc finshed"


echo "ggc  tensorflow op on gpu  over!"

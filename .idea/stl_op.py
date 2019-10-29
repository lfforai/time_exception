#该模型是基于lstm预测值与实际值偏差在【avg-std*3，avg+std*3】范围以外作为异常值得判断，每BATCH_SIZE一组计算方差和均值，其中范围标准差倍数需要设置
from __future__ import absolute_import, division, print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='3'
#os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] ='4'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
#tf.debugging.set_log_device_placement(True)
#tf.config.set_soft_device_placement(True)
import matplotlib.pyplot as plt



zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)

def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)

csv_path, _ = os.path.splitext(zip_path)
import pandas as pd
df = pd.read_csv(csv_path,converters={'T (degC)':np.float32})

uni_data = df['T (degC)']
uni_data.index = df['Date Time']
print(uni_data.head())
# uni_data.plot(subplots=True)
uni_data = uni_data.values;
so_file = '/root/cuda-workspace/stl/src/kernel_example.so'
# print(tf.__version__)

my_add_module = tf.load_op_library(so_file)
a=tf.constant(np.reshape(uni_data[:3000],(3000,1)),dtype=tf.float32);
# print(a)
T,S,R=my_add_module.example(a,51,50,301,301,2)
# print(T)
# print(R)
# print(S)

#self.assertAllEqual(result, [2, 4, 6, 8, 10])

l=np.arange(3000)

#第一行第一列图形
ax1 = plt.subplot(2,2,1)
#第一行第二列图形
ax2 = plt.subplot(2,2,2)
#第二行
ax3 = plt.subplot(2,2,3)
ax4 = plt.subplot(2,2,4)
plt.sca(ax1)
plt.title("data")
plt.plot(l,uni_data[:3000],'y')
plt.sca(ax2)
plt.title("T")
plt.plot(l,T,'r')
plt.sca(ax3)
plt.title("S")
plt.plot(l,S,'b')
plt.sca(ax4)
plt.title("R")
plt.plot(l,R,'g')
plt.show()





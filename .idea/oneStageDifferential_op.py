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
so_file = '/root/cuda-workspace/oneStageDifferential/src/oneStageDifferential.so'

import pandas as pd
df = pd.read_excel("/lf_tool/data/123456.xls",dtype={'siteZoonId':str},parse_dates=['rowkeyTime'])
street='jaxq_dma_zone_l4'
uni_data = df[df['siteZoonId']==street]['flowCur']
uni_data.index = df[df['siteZoonId']==street]['rowkeyTime']
print(uni_data.head())
# test=uni_data[uni_data.index>datetime.datetime.strptime('2019-08-25','%Y-%m-%d')]

# uni_data.plot(subplots=True)
uni_data = uni_data.values
TRAIN_SPLIT = 400

tf.random.set_seed(13)
univariate_past_history = 20  #调整历史窗口大小
univariate_future_target = 0

BATCH_SIZE = 5
BUFFER_SIZE = 500
#训练集
temp=tf.data.Dataset.from_tensor_slices(tf.cast(uni_data,dtype=tf.float32)).batch(univariate_past_history,drop_remainder=True)
temp_1=temp.map(lambda x:tf.reshape(x,[univariate_past_history,1]))
temp_2=temp.map(lambda x:x[univariate_past_history-1])
last=tf.data.Dataset.zip((temp_1,temp_2)).batch(BATCH_SIZE,drop_remainder=True)

my_add_module = tf.load_op_library(so_file)
a=tf.constant(np.reshape(uni_data[:576],(576,1)),dtype=tf.float32);
R=my_add_module.onestagedifferential(a)

l=np.arange(575)
plt.title("data")
plt.plot(l,uni_data[:575],'y')
plt.plot(l,R,'b')
plt.show()
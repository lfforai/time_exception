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


#
# #os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT']='4'
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)
#
# import contextlib
# # 构建包含上下文管理器的函数，使其可以在with中使用
# @contextlib.contextmanager
# def assert_raises(error_class):
#     try:
#         yield
#     except error_class as e:
#         print('Caught expected exception \n  {}: {}'.format(error_class, e))
#     except Exception as e:
#         print('Got unexpected exception \n  {}: {}'.format(type(e), e))
#     else:
#         raise Exception('Expected {} to be raised but no error was raised!'.format(
#             error_class))
#
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import pandas as pd
#
# #使用的是温度测试数据集
# mpl.rcParams['figure.figsize'] = (8, 6)
# mpl.rcParams['axes.grid'] = False
#
# zip_path = tf.keras.utils.get_file(
#     origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
#     fname='jena_climate_2009_2016.csv.zip',
#     extract=True)
#
# def univariate_data(dataset, start_index, end_index, history_size, target_size):
#     data = []
#     labels = []
#
#     start_index = start_index + history_size
#     if end_index is None:
#         end_index = len(dataset) - target_size
#
#     for i in range(start_index, end_index):
#         indices = range(i-history_size, i)
#         # Reshape data from (history_size,) to (history_size, 1)
#         data.append(np.reshape(dataset[indices], (history_size, 1)))
#         labels.append(dataset[i+target_size])
#     return np.array(data), np.array(labels)
#
# csv_path="C://Users//Administrator//Documents//WeChat Files//yeshimygod//FileStorage//File//2019-10//data.xls"
# import pandas as pd
# import datetime

# pd.set_option('display.max_columns',1000)
# pd.set_option('display.width', 1000)
# pd.set_option('display.max_colwidth',1000)
#
# street='flzA_dma_zone_l4'
# df = pd.read_excel(csv_path,dtype={'siteZoonId':str},parse_dates=['rowkeyTime'])
# uni_data = df[df['siteZoonId']==street]['flowCur']
# uni_data.index = df[df['siteZoonId']==street]['rowkeyTime']
# test=uni_data[uni_data.index>datetime.datetime.strptime('2019-08-25','%Y-%m-%d')]
# print(test)
# uni_data.plot(subplots=True)
# # uni_data = uni_data.values
# # TRAIN_SPLIT = 300
# # uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
# # uni_train_std = uni_data[:TRAIN_SPLIT].std()
# # uni_data = (uni_data-uni_train_mean)/uni_train_std
# plt.show()
# range_ds = tf.data.Dataset.range(100000)
#
# window_size = 5
#
# windows = range_ds.window(window_size, shift=1)
#
# def sub_to_batch(sub):
#     return sub.batch(window_size, drop_remainder=True)
#
# buffer_size=100
# wind= windows.flat_map(sub_to_batch)
# wind=wind.map(lambda x:(tf.reshape(x,[5,1]),x[4])).batch(10,drop_remainder=True).shuffle(buffer_size).repeat()
# for a,b in wind:
# #     print(a,"=>",b)
# print(tf.sysconfig.get_compile_flags())
# print(tf.sysconfig.get_link_flags())
# print(tf.__path__)
#
# print(tf.__version__)
#
# # a=tf.constant([1,2,3],dtype=tf.int32)
# # class ZeroOutTest(tf.test.TestCase):
# #     def testZeroOut(self):
# #         zero_out_module = tf.load_op_library('/root/cuda-workspace/add_out/src/add_out.so')
# #         with self.cached_session():
# #             result = zero_out_module.add_out(a)
# #             self.assertAllEqual(result.eval(),a)
# #
# # a=ZeroOutTest()
# # a.testZeroOut()
#
# import tensorflow as tf
# zero_out_module = tf.load_op_library('/root/cuda-workspace/add_out/src/add_out.so')
# with tf.compat.v1.Session (''):
#     zero_out_module.add_out([[1, 2], [3, 4]]).eval()
#
# # Prints
# array([[1, 0], [0, 0]], dtype=int32)



# print(zero_out_module.zero_out([1,2,3]))






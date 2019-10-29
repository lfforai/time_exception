import tensorflow as tf
import numpy as np
import time
#使用LOF局部异常值检验
start = time.clock()
tf.debugging.set_log_device_placement(True)
tf.config.set_soft_device_placement(True)
#so_file = '/root/cuda-workspace/add_out/src/kernel_example.so'
so_file = '/root/cuda-workspace/LOF_ICAD/src/kernel_example.so'

import pandas as pd

df = pd.read_excel("/lf_tool/data/123456.xls",dtype={'siteZoonId':str},parse_dates=['rowkeyTime'])

street='jaxq_dma_zone_l4'
uni_data = df[df['siteZoonId']==street]['flowCur']
uni_data.index = df[df['siteZoonId']==street]['rowkeyTime']
print(uni_data.head())
# test=uni_data[uni_data.index>datetime.datetime.strptime('2019-08-25','%Y-%m-%d')]

uni_data.plot(subplots=True)
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
#.repeat()

for x,y in last:
    my_add_module = tf.load_op_library(so_file)
    print(x[0])
    print("标致值超过5以上的为怀疑错误值",my_add_module.example(x[0],3))
    print("--------------------------------------")
# print(tf.__version__)
# class MyAddTest(tf.test.TestCase):
#     def testMyAdd(self):
#         my_add_module = tf.load_op_library(so_file)
#         a=tf.constant([[3],[999],[2],[1],[1],[3321],[5],[999],[2],[1],[1],[3321],[5],[1],[3323],[223],[1],[222],[19],[999],[2],[1],[999],[2],[1],[1],[3321],[5],[999],[2],[1],[1],[3321],[5],[1],[3323],[223],[1],[222],[19],[999],[2],[1],[1],[3321],[5],[1],[3323],[223],[1],[222],[19],[899],[3],[11],[150.1],[23.0],[25.0],[16.0],[23.0],[25.0],[16.0],[23.0],[25.0],[16.0],[23.0],[25.0],[16.0],[90.0],[12.0]],dtype=tf.float32)
#         # print(a)
#         result=my_add_module.example(a,5)
#         for e in zip(a,result):
#             print(e[0].numpy(),e[1].numpy())
#         #self.assertAllEqual(result, [2, 4, 6, 8, 10])
#
# a=MyAddTest()
# a.testMyAdd()
#
# #当中是你的程序
# print("大于1的可能是异常值")


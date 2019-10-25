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

#os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT']='4'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import contextlib
# 构建包含上下文管理器的函数，使其可以在with中使用
@contextlib.contextmanager
def assert_raises(error_class):
    try:
        yield
    except error_class as e:
        print('Caught expected exception \n  {}: {}'.format(error_class, e))
    except Exception as e:
        print('Got unexpected exception \n  {}: {}'.format(type(e), e))
    else:
        raise Exception('Expected {} to be raised but no error was raised!'.format(
            error_class))

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

#使用的是温度测试数据集
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

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
# df = pd.read_csv(csv_path,converters={'T (degC)':np.float32})
#
# uni_data = df['T (degC)']
# uni_data.index = df['Date Time']
# print(uni_data.head())


df = pd.read_excel("C://Users//Administrator//Documents//WeChat Files//yeshimygod//FileStorage//File//2019-10//test.xlsx",dtype={'siteZoonId':str},parse_dates=['rowkeyTime'])
# total_df=df['flowCur']
# total_len=df.__len__()
# total_df.index=df['rowkeyTime']
# total_data=total_df.values
# total_mean =total_data[:total_len].mean()
# total_std = total_data[:total_len].std()
# total_data = (total_data-uni_train_mean)/uni_train_std
street='bhjsl_dma_zone_l4'
uni_data = df[df['siteZoonId']==street]['flowCur']
uni_data.index = df[df['siteZoonId']==street]['rowkeyTime']
print(uni_data.head())
# test=uni_data[uni_data.index>datetime.datetime.strptime('2019-08-25','%Y-%m-%d')]

uni_data.plot(subplots=True)
uni_data = uni_data.values
TRAIN_SPLIT = 400
uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()
uni_data = (uni_data-uni_train_mean)/uni_train_std

tf.random.set_seed(13)
univariate_past_history = 20  #调整历史窗口大小
univariate_future_target = 0

BATCH_SIZE = 5
BUFFER_SIZE = 500
#训练集
temp=tf.data.Dataset.from_tensor_slices(tf.cast(uni_data[:TRAIN_SPLIT],dtype=tf.float32)).batch(univariate_past_history,drop_remainder=True)
temp_1=temp.map(lambda x:tf.reshape(x,[univariate_past_history,1]))
temp_2=temp.map(lambda x:x[univariate_past_history-1])
last=tf.data.Dataset.zip((temp_1,temp_2)).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True).repeat()


#测试集
temp_val=tf.data.Dataset.from_tensor_slices(tf.cast(uni_data[TRAIN_SPLIT:-1],dtype=tf.float32)).batch(univariate_past_history,drop_remainder=True)
temp_1_val=temp_val.map(lambda x:tf.reshape(x,[univariate_past_history,1]))
temp_2_val=temp_val.map(lambda x:x[univariate_past_history-1])
last_val=tf.data.Dataset.zip((temp_1_val,temp_2_val)).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True).repeat()


#univariate_data(dataset, start_index, end_index, history_size, target_size):
x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)

x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)
# print ('Single window of past history')
# print (x_train_uni[2])
# print ('\n Target temperature to predict')
# print (y_train_uni[2])

def create_time_steps(length):
    time_steps = []
    for i in range(-length, 0, 1):
        time_steps.append(i)
    return time_steps

def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt

def baseline(history):
    return np.mean(history)
# show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')
# show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,
#           'Baseline Prediction Example')

# train_univariate = tf.data.Dataset.from_tensor_slices((tf.cast(x_train_uni,dtype=tf.float32), tf.cast(y_train_uni,dtype=tf.float32)))
# train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True).repeat()
#
val_univariate = tf.data.Dataset.from_tensor_slices((tf.cast(x_val_uni,dtype=tf.float32), tf.cast(y_val_uni,dtype=tf.float32)))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

#该模型是基于lstm预测值与实际值偏差在【avg-std*3，avg+std*3】范围以外作为异常值得判断，每BATCH_SIZE一组计算方差和均值，其中范围标准差倍数需要设置
#设置异常方差的倍数范围
std_time=3
#设置训练或者执行异常数值检测
absolute=True #绝对偏差
train=False
if train==True:
    #3层lstm
    simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(125, input_shape=x_train_uni.shape[-2:], return_sequences=True),
        tf.keras.layers.LSTM(125, input_shape=[univariate_past_history,125], return_sequences=True),
        tf.keras.layers.LSTM(125, input_shape=[univariate_past_history,125]),
        tf.keras.layers.Dense(1)
    ])
    #2层lstm
    # simple_lstm_model = tf.keras.models.Sequential([
    #     tf.keras.layers.LSTM(125, input_shape=x_train_uni.shape[-2:], return_sequences=True),
    #     tf.keras.layers.LSTM(125, input_shape=[univariate_past_history,125]),
    #     tf.keras.layers.Dense(1)
    # ])

    #1层lstm
    # simple_lstm_model = tf.keras.models.Sequential([
    #     tf.keras.layers.LSTM(125, input_shape=x_train_uni.shape[-2:]),
    #     tf.keras.layers.Dense(1)
    # ])
    simple_lstm_model.compile(optimizer='adam', loss='mae')
    EVALUATION_INTERVAL = 50
    EPOCHS = 20
    # simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
    #                       steps_per_epoch=EVALUATION_INTERVAL,
    #                       validation_data=val_univariate, validation_steps=50)

    simple_lstm_model.fit(last, epochs=EPOCHS,
                          steps_per_epoch=EVALUATION_INTERVAL,
                          validation_data=last_val, validation_steps=5)

    for x, y in val_univariate.take(3):
        print(y[0].numpy())
        print(simple_lstm_model.predict(x)[0])
        print("-----------------------------")
        # plot = show_plot([x[0].numpy(), y[0].numpy(),
        #                   simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
        # plot.show()
    simple_lstm_model.save('D:\data\model_par\path_to_my_model.h5')

else:
    #绝对偏差
    new_model = keras.models.load_model('D:\data\model_par\path_to_my_model.h5')
    print()
    val_univariate = tf.data.Dataset.from_tensor_slices((tf.cast(x_val_uni,dtype=tf.float32), tf.cast(y_val_uni,dtype=tf.float32))).batch(BATCH_SIZE,drop_remainder=True)
    list_var_mean=[]
    list_value=[]

    index_list=list(range(BATCH_SIZE))
    total_mean=0
    total_std=0
    erres=0
    for x, y in val_univariate:
        pre=new_model.predict(x)
        pre=np.reshape(pre,[-1,])
        old=y.numpy()
        cha=(pre-old)/old  #Relative
        # rint(pre-old)
        if absolute==True:
           cha=pre-old
        #计算3倍标准差以内
        total_mean=total_mean*0.8+np.mean(cha)*0.2
        total_std=total_std*0.8+np.std(cha)*0.2
        up=total_mean+std_time*total_std
        down=total_mean-std_time*total_std
        # print(up)
        # print(down)
        list_value=zip(index_list,cha)
        erros_sample=[x[0] for x in list_value if x[1]>up or x[1]<down]
        if erros_sample.__len__()>0:
            erres+=1
            print("------------异常点输出--------------")
            print("异常数据坐标{},".format(erros_sample))
            print()
            for index in erros_sample:
                    print(index)
                    index_o=int(index)
                    print("原始数据{:.3f}，预测数据{:.3f},前后20个数据对比".format(old[index_o],pre[index_o]))
                    print(old)
                    print(pre)
                    # tem_list_old=[]
                    # tem_list_pre=[]
                    # for  i in range(-1,1,1):
                    #     tem_list_old.append(np.round(old[index_o-i],3))
                    #     tem_list_pre.append(np.round(pre[index_o-i],3))
                    # print("原始数据{}".format(tem_list_old))
                    # print("预测数据{}".format(tem_list_pre))
                    # if absolute==True:
                    #    print("abs差值{},|up界限{:.5f},异常{:.5f},down界限{:.5f}".format(np.round(np.array(tem_list_pre)-np.array(tem_list_old),4),up,pre[index_o]-old[index_o],down))
                    # else:
                    #    print("abs差值{},|up界限{:.5f},异常{:.5f},down界限{:.5f}".format(np.round((np.array(tem_list_pre)-np.array(tem_list_old))/np.array(tem_list_old),4),up,(pre[index_o]-old[index_o])/old[index_o],down))
                    print("------------next---------------")
    print("错误条数",erres)




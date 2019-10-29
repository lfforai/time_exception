import tensorflow as tf
import numpy as np
import time
#使用LOF局部异常值检验
start = time.clock()
tf.debugging.set_log_device_placement(True)
tf.config.set_soft_device_placement(True)
#so_file = '/root/cuda-workspace/add_out/src/kernel_example.so'
so_file = '/root/cuda-workspace/LOF_ICAD/src/kernel_example.so'
print(tf.__version__)
class MyAddTest(tf.test.TestCase):
    def testMyAdd(self):
        my_add_module = tf.load_op_library(so_file)
        a=tf.constant([[3],[999],[2],[1],[1],[3321],[5],[999],[2],[1],[1],[3321],[5],[1],[3323],[223],[1],[222],[19],[999],[2],[1],[999],[2],[1],[1],[3321],[5],[999],[2],[1],[1],[3321],[5],[1],[3323],[223],[1],[222],[19],[999],[2],[1],[1],[3321],[5],[1],[3323],[223],[1],[222],[19],[899],[3],[11],[150.1],[23.0],[25.0],[16.0],[23.0],[25.0],[16.0],[23.0],[25.0],[16.0],[23.0],[25.0],[16.0],[90.0],[12.0]],dtype=tf.float32)
        # print(a)
        result=my_add_module.example(a,5)
        for e in zip(a,result):
            print(e[0].numpy(),e[1].numpy())
        #self.assertAllEqual(result, [2, 4, 6, 8, 10])

a=MyAddTest()
a.testMyAdd()

#当中是你的程序
print("大于1的可能是异常值")


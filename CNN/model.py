import tensorflow as tf
import numpy as np


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class CNN_model:

    def __init__(self):
        self.a = 2

    def get_graph(self,input,batch_size,w=100,h=100,output_shape=25,dropout_rate=0.5):

        d_type = tf.float32
        inp_channels = 3
        # batch_size = 64
        # input = tf.placeholder(tf.float32,shape=(batch_size,w,h,inp_channels))
        # self.input = input
        padding = 'SAME'

        #First Conv layer
        out_channels = 64
        b = 16
        filter_1 = tf.Variable(initial_value=tf.random_normal([b,b,inp_channels,out_channels]),
                               name='filter_1')
        stride_1 = [1,4,4,1]
        conv_1 = tf.nn.conv2d(input,filter_1,stride_1,padding)
        bias_1 = bias_variable([out_channels])
        conv_1 = tf.nn.relu(conv_1 + bias_1)

        # First Pool layer
        stride_pool = [1,1,1,1]
        ksize = [1,2,2,1]
        pool1 = tf.nn.max_pool(conv_1,ksize,stride_pool,padding)

        #Second Conv layer
        stride_conv = [1,1,1,1]
        b = 2
        filter_2 = tf.Variable(initial_value=tf.random_normal([b,b,64,112]),name='filter_2')
        conv_2 = tf.nn.conv2d(pool1,filter_2,stride_conv,padding)
        bias_2 = bias_variable([112])
        conv_2 = tf.nn.relu(conv_2 + bias_2)


        #Third Conv layer
        stride_conv = [1,1,1,1]
        b = 3
        filter_3 = tf.Variable(initial_value=tf.random_normal([b,b,112,80]),name='filter_3')
        conv_3 = tf.nn.conv2d(conv_2,filter_3,stride_conv,padding)
        bias_3 = bias_variable([80])
        conv_3 = tf.nn.relu(conv_3 + bias_3)

        shape = int(np.prod(conv_3.get_shape()[1:]))
        conv_3_flat = tf.reshape(conv_3, [-1, shape])

        fc1w = tf.Variable(tf.truncated_normal([shape, 4096], dtype=tf.float32, stddev=1e-1), name='fc1w')
        fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                           trainable=True, name='fc1b')

        fc1 = tf.nn.bias_add(tf.matmul(conv_3_flat, fc1w), fc1b)
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1,0.5)

        shape = int(np.prod(fc1.get_shape()[1:]))

        fc2w = tf.Variable(tf.truncated_normal([shape,output_shape**2], dtype=tf.float32,stddev=1e-1), name='fc2w')
        fc2b = tf.Variable(tf.constant(1.0, shape=[output_shape**2], dtype=tf.float32), trainable=True,name='fc2b')
        fc2 = tf.nn.relu(tf.matmul(fc1,fc2w) +fc2b)
        # fc2 = tf.nn.dropout(fc2,0.5)

        self.pred = tf.reshape(fc2,[batch_size,output_shape,output_shape],name='pred')

        # self.pred = tf.sigmoid(self.pred)
        # self.pred = tf.reshape(self.pred,[self.pred.get_shape()[0],w,h])
        return self.pred










# model = CNN_model()
# preds = model.get_graph(256.0,256.0,64.0)
# a= 2

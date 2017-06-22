import tensorflow as tf


class CNN_model:

    def __init__(self):
        self.a = 2

    def get_graph(self,w,h,batch_size):

        d_type = tf.float32
        inp_channels = 3
        input = tf.placeholder(tf.float32,shape=(batch_size,w,h,inp_channels))
        padding = 'SAME'


        #First Conv layer
        out_channels = 64
        b = 16
        filter_1 = tf.Variable(initial_value=tf.random_normal([b,b,inp_channels,out_channels]))
        stride_1 = [1,4,4,1]
        conv_1 = tf.nn.conv2d(input,filter_1,stride_1,padding)
        conv_1 = tf.nn.relu(conv_1)

        # First Pool layer
        stride_pool = [1,1,1,1]
        b = 2
        ksize = [1,2,2,1]
        pool1 = tf.nn.max_pool(conv_1,ksize,stride_pool,padding)

        #Second Conv layer
        stride_conv = [1,1,1,1]
        b = 2
        filter_2 = tf.Variable(initial_value=tf.random_normal([b,b,64,112]))
        conv_2 = tf.nn.conv2d(pool1,filter_2,stride_conv,padding)
        conv_2 = tf.nn.relu(conv_2)


        #Third Conv layer
        stride_conv = [1,1,1,1]
        b = 3
        filter_3 = tf.Variable(initial_value=tf.random_normal([b,b,112,80]))
        conv_3 = tf.nn.conv2d(conv_2,filter_3,stride_conv,padding)

        a = 2
        # #First FC layer 1
        # drop_out = 0.5
        # # fc = tf.reshape(conv_3, (64,))
        # fc = tf.nn.dropout(conv_3,drop_out)
        # weights = tf.Variable(initial_value=tf.random_variable([3920,4096]))
        # biases = tf.Variable(initial_value=tf.random_variable(4096))
        # fc = tf.nn.relu(tf.matmul(fc, weights) + biases)




model = CNN_model()
model.get_graph(10,10,64)
a= 2

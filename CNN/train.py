import math
import numpy as np
import tensorflow as tf
from glob import glob
import os
from model import vgg16
import utils
from skimage.io import imread
import pprint
pp = pprint.PrettyPrinter(depth=6)
# Load data
os.environ["CUDA_VISIBLE_DEVICES"] = "2" ##os.environ['SGE_GPU']
flags = tf.app.flags
tf.flags.DEFINE_integer("learning_rate", 1e-3, "learning rate (default: 1e-3)")
tf.flags.DEFINE_integer("batch_size", 2, "Batch Size")
FLAGS = tf.flags.FLAGS
def main(_):
    pp.pprint(tf.flags.FLAGS.__flags)
    output_image_dims = [2,2]
    w = output_image_dims[0]
    h = output_image_dims[1]
    image_dims = [128, 128, 3]
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:
        inputs = tf.placeholder( tf.float32, [FLAGS.batch_size] + image_dims, name='input')
        labels = tf.placeholder( tf.float32, [FLAGS.batch_size] + output_image_dims, name='output')
        model = vgg16(FLAGS.batch_size, w, inputs, 'vgg16_weights.npz', sess)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model.pred, labels=labels))
        t_vars = tf.trainable_variables()
        loss_sum = tf.summary.scalar("loss", loss)
        saver = tf.train.Saver(var_list=t_vars)
        summ = tf.merge_summary([loss_sum])
        writer =  tf.summary.FileWriter(os.path.join("summary"), sess.graph)
        d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss, var_list=t_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        counter = 0
        epoch = 24
        window_size = 128
        output_size = 2
        data = glob(os.path.join( "aerialOrg","train","*.jpg"))
        path_label = glob(os.path.join( "mapOrg","train","*.jpg"))
        im = imread(data[0])
        img_dim = np.min(im.shape)
        path_num = math.floor((img_dim - window_size) / output_size) + 1
        x = [window_size * i for i in range(0, path_num)]
        y = x
        for epoch in range(0, epoch):
            batch_idxs = len(data)// FLAGS.batch_size
            for i in x:
                for j in y:
                    for idx in range(0, batch_idxs):
                        batch_images, batch_files_label =utils.getBatch(i,j,window_size,data[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size],path_label[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size])
                        err,_, summary_str = sess.run([loss,d_optim, summ],
                                               feed_dict={model.inputs: batch_images, model.labels:batch_files_label})
                        writer.add_summary(summary_str, counter)
                        counter +=1
                        print(
                            "Epoch: [%2d] [%4d/%4d] loss: %.8f" \
                            % (epoch, idx, batch_idxs,err))
            saver.save(sess, os.path.join("checkpoint", "CIL"), global_step=epoch)

import math
import numpy as np
import tensorflow as tf
from glob import glob
import os
from model import vgg16
import utils
from skimage.io import imread, imsave
import pprint
pp = pprint.PrettyPrinter(depth=6)
# Load data
os.environ["CUDA_VISIBLE_DEVICES"] = "2" ##os.environ['SGE_GPU']

class CNN(object):
    def __init__(self, sess, img_height = 400, input_height=128, input_width=128, batch_size=2, output_height=16, output_width=16, input_fname_pattern='*.jpg', checkpoint_dir="checkpoint", sample_dir="samples", name="CIL",dataset=""):
        self.sess = sess
        self.batch_size = batch_size
        self.img_height = img_height
        self.input_height = input_height
        self.input_width = input_height
        self.output_height = output_height
        self.output_width = output_height
        self.dataset_name = dataset
        self.input_fname_pattern = input_fname_pattern
        self.name = name+"_"+str(self.output_height)+"_"+str(self.input_height)
        self.class_dim = 1
        self.checkpoint_dir = os.path.join(checkpoint_dir, self.name)        
        print self.name
        self.sample_dir = sample_dir
        self.build_model()
        
    def build_model(self): 
        self.inputs = tf.placeholder( tf.float32, [FLAGS.batch_size, self.input_height, self.input_height, 3], name='input')
        self.labels = tf.placeholder( tf.float32, [FLAGS.batch_size, self.output_height*self.output_height], name='output')
        self.model = vgg16(self.batch_size, self.output_height*self.output_height, self.inputs, 'vgg16_weights.npz', self.sess)
        #which loss function to use?
        #try this sometime: tf.nn.weighted_cross_entropy_with_logits
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.model.pred, self.labels))
        loss_sum = tf.summary.scalar("loss", self.loss)
        
        if not os.path.exists(os.path.join(self.checkpoint_dir)):
            os.makedirs(os.path.join(self.checkpoint_dir))
        if not os.path.exists(os.path.join(self.sample_dir,self.name)):
            os.makedirs(os.path.join(self.sample_dir,self.name))            
        if not os.path.exists(os.path.join("summary", self.name)):
            os.makedirs(os.path.join("summary", self.name))
        self.summ = tf.merge_summary([loss_sum])   
        self.summVal = tf.merge_summary([loss_sum])   
        self.writer =  tf.summary.FileWriter(os.path.join("summary", self.name), self.sess.graph)

        
    def train(self, config):  
        t_vars = tf.trainable_variables()
        print [v.name for v in t_vars]
        self.saver = tf.train.Saver(var_list=t_vars, max_to_keep=0)

        #tf.train.GradientDescentOptimizer
        self.d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss, var_list=t_vars)        
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        counter = 0
        bol, counter = self.load()
        data = glob(os.path.join(self.dataset_name, "train", self.input_fname_pattern))
        #glob(os.path.join( "aerialOrg","train","*.jpg"))
        data_label = glob(os.path.join( "mapOrg","train", self.input_fname_pattern))
        path_num = math.floor(self.img_height/self.output_height)
        print "number of patch per image ", path_num
        x = [self.output_height * i for i in range(0, int(path_num))]
        y = x
        print "the coordinates ", x
        for epoch in range(0, config.epoch):
            batch_idxs = len(data)// self.batch_size
            for i in x:
                for j in y:
                    for idx in range(0, batch_idxs):
                        batch_images, batch_files_label =utils.getBatch(i,j,self.input_height,self.output_height, data[idx*self.batch_size:(idx+1)*self.batch_size],data_label[idx*self.batch_size:(idx+1)*self.batch_size])
                        err,_, summary_str = self.sess.run([self.loss,self.d_optim, self.summ],
                                               feed_dict={self.inputs: batch_images, self.labels:batch_files_label})
                        self.writer.add_summary(summary_str, counter)
                        counter +=1
                        print(
                            "Epoch: [%2d] [%4d/%4d] loss: %.8f" \
                            % (epoch, idx, batch_idxs,err))
            self.saver.save(self.sess, os.path.join(self.checkpoint_dir,"CIL"), global_step=epoch)
            self.test(epoch)
    def test(self, epoch): 
        path_num = math.floor(self.img_height/self.output_height)
        x = [self.output_height * i for i in range(0, int(path_num))]
        y = x
        data_val = glob(os.path.join(self.dataset_name, "val", self.input_fname_pattern))
        #glob(os.path.join( "aerialOrg","train","*.jpg"))
        data_label_val = glob(os.path.join( "mapOrg","val", self.input_fname_pattern))   
        batch_idxs = len(data_val)// self.batch_size
        for ind in range(0,batch_idxs):
            img = np.zeros((self.img_height,self.img_height)).astype(int)
            total_loss = 0 
            for i in x:
                for j in y:
                    batch_images, batch_files_label =utils.getBatch(i,j,self.input_height,self.output_height,data_val[ind*self.batch_size:(ind+1)*self.batch_size], data_label_val[ind*self.batch_size:(ind+1)*self.batch_size])
                    result, loss, summary_str = self.sess.run([self.model.result, self.loss, self.summVal],
                                               feed_dict={self.inputs: batch_images, self.labels:batch_files_label})
                    result = np.reshape(result[0,:], [self.output_height,self.output_height])
                    img[i:i+self.output_height,j:j+self.output_height] = result
                    total_loss += loss
                    self.writer.add_summary(summary_str, epoch)
            total_loss = total_loss/float(len(x)*len(x))       
            "Test Epoch: [%2d] loss: %.8f"  % (epoch,total_loss)     
            imsave('./{}/test_{:02d}_{:d}.png'.format(os.path.join(self.sample_dir, self.name), epoch, ind),img)

            
    def load(self):
       import re
       print(" [*] Reading checkpoints...")
       checkpoint_dir = os.path.join(self.checkpoint_dir)
       ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
       if ckpt and ckpt.model_checkpoint_path:
         ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
         self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
         counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
         print(" [*] Success to read {}".format(ckpt_name))
         return True, counter 
       else:
         print(" [*] Failed to find a checkpoint")
         return False, 0
            
            
            
flags = tf.app.flags
flags.DEFINE_integer("epoch", 24, "epoch")
flags.DEFINE_integer("batch_size", 2, "Batch Size")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("input_height", 128, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("output_height", 16, "The size of the output images to produce [64]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("name", "CIL", "the string for experiment [*]")
flags.DEFINE_string("dataset", "aerialOrg", "The name of dataset [aerialOrg]")

FLAGS = tf.flags.FLAGS    
def main(_):
    pp.pprint(flags.FLAGS.__flags)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True
    with tf.Session(config=run_config) as sess:
        cnn = CNN(
            sess,
            input_height=FLAGS.input_height,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            input_fname_pattern=FLAGS.input_fname_pattern,
            checkpoint_dir=FLAGS.checkpoint_dir,
            name = FLAGS.name,
            dataset = FLAGS.dataset)
        if cnn.output_height == 8:
            t_vars = tf.trainable_variables()
            print [v.name for v in t_vars]
            cnn.saver = tf.train.Saver(var_list=t_vars, max_to_keep=0)
            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()
            counter = 0
            print cnn.name
            bol, counter = cnn.load()
            if bol:
                cnn.test(0)
            else:
                print "ow no"
        else:    
            cnn.train(FLAGS)
        
if __name__ == '__main__':
  tf.app.run()

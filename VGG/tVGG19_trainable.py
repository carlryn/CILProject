"""
Simple tester for the vgg19_trainable
"""
import skimage.transform
import tensorflow as tf
from skimage.io import imread,imsave
import VGG19_trainable as vgg19
import utils

img1 = utils.imread("../data/training/images/satImage_001.png")[0:224,0:224]
img1_true_result = [1 if i == 292 else 0 for i in range(625)]  # 1-hot result for tiger

batch1 = img1.reshape((1, 224, 224, 3))

# def load_image(path):
#     # load image
#     img = skimage.io.imread(path)
#     img = img / 255.0
#     assert (0 <= img).all() and (img <= 1.0).all()
#     # print "Original Image Shape: ", img.shape
#     # we crop image from center
#     short_edge = min(img.shape[:2])
#     yy = int((img.shape[0] - short_edge) / 2)
#     xx = int((img.shape[1] - short_edge) / 2)
#     crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
#     # resize to 224, 224
#     resized_img = skimage.transform.resize(crop_img, (224, 224))
#     return resized_img

# path = "../data/training/images/satImage_001.png"
#
# img = load_image(path)
a = 2

with tf.device('/cpu:0'):
    sess = tf.Session()

    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [1, 625])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg19.Vgg19('../pretrained_models/vgg19.npy')
    vgg.build(images, train_mode)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print(vgg.get_var_count())

    sess.run(tf.global_variables_initializer())

    # test classification
    prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    # utils.print_prob(prob[0], './synset.txt')

    # simple 1-step training
    cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    sess.run(train, feed_dict={images: batch1, true_out: [img1_true_result], train_mode: True})

    # test classification again, should have a higher probability about tiger
    prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    # utils.print_prob(prob[0], './synset.txt')

    # test save
    # vgg.save_npy(sess, './test-save.npy')
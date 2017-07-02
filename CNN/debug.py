import math
import numpy as np
import tensorflow as tf
from glob import glob
import os
import utils
from skimage.io import imread
counter = 0
epoch = 24
batch_size = 2
window_size = 128
output_size = 4
data = glob(os.path.join( "aerialOrg","train","*.jpg"))
path_label = glob(os.path.join( "mapOrg","train","*.jpg"))
im = imread(data[0])
img_dim = np.min(im.shape[0:1])
#pad the image
path_num =  math.floor(img_dim/output_size)# math.floor((img_dim - window_size) / output_size) + 1
x = [output_size * i for i in range(0, path_num)]
y = x
for epoch in range(0, epoch):
    batch_idxs = len(data)// batch_size
    for i in x:
        for j in y:
            print(i,j)
            for idx in range(0, batch_idxs):
                batch_images, batch_files_label = utils.getBatch(i, j, window_size, output_size, data[idx * batch_size:(idx + 1) * batch_size],path_label[idx * batch_size:(idx + 1) * batch_size])
                print(batch_files_label[0].shape)
                print(batch_images[0].shape)
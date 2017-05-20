import os
from glob import glob # for lists of files
from skimage.io import imread, imshow, imsave # for reading images

from skimage.morphology import dilation, binary_dilation, closing, opening
import numpy as np
from PIL import Image, ImageDraw
import cclabel
from itertools import product
import utils


# Takes around 2 minutes, removes directories first
def preprocess(DATA_ROOT):
    list_imgs = glob(os.path.join(DATA_ROOT, '*.jpg'))
    os.rmdir(os.path.join(DATA_ROOT, "groundtruth"))
    os.rmdir(os.path.join(DATA_ROOT, "images"))
    os.mkdir(os.path.join(DATA_ROOT, "groundtruth"))
    os.mkdir(os.path.join(DATA_ROOT, "images"))
    for i,img_file in enumerate(list_imgs):
        img = imread(img_file)
        aerial = img[:, 0:int(img.shape[1] / 2), :]
        map = img[:, int(img.shape[1] / 2):, :]
        imsave(os.path.join(DATA_ROOT,'groundtruth',os.path.basename(img_file)),map)
        imsave(os.path.join(DATA_ROOT,'images',os.path.basename(img_file)),aerial)





data_path = '../maps/train'

DATA_ROOT = '../maps/train/groundtruth'

image_files = glob(os.path.join(DATA_ROOT, '*.jpg'))
im_path = image_files[57]
inx = [os.path.basename(img_file) for img_file in image_files]
for k in range(0, len(inx)):
    im_path = image_files[k]
    print(im_path)
    # im_path = "./maps/train/groundtruth\\618.jpg"
    im_data = imread(im_path)
    utils.binary_th(im_data, inx[k], DATA_ROOT, im_path)
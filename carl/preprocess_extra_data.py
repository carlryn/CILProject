import numpy as np
from skimage.io import imread, imshow, imsave # for reading images
import os
from glob import glob # for lists of files


def preprocess(DATA_ROOT):
    list_files = glob(os.path.join(DATA_ROOT, '*.jpg'))
    imread(list_files[0])



data_path = '../maps/train'
preprocess(data_path)
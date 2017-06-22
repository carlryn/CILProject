import numpy as np
import os
from skimage.io import imread, imshow, imsave

def load_train_data(train_path,sample=None):
    data = get_data(train_path,sample)
    return data, None

def get_data(path,sample):
    images_names = os.listdir(path)
    data = []
    for i,img_name in enumerate(images_names):
        data.append(imread(os.path.join(path,img_name)))
        data[i] = data[i][100:-100][100:-100]

    if sample is not None:
        data = data[:sample]

    return data


train_path = '../idil/forTraining'
data = load_train_data(train_path)

a = 2
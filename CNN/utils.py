import numpy as np
import os
from skimage.io import imread

def load_train_data(train_path,sample=None):
    data, data_labels = get_data(train_path,sample)
    return data, data_labels

def get_data(path,sample):
    images_names = os.listdir(path)
    data = []
    data_labels =[]
    for i,img_name in enumerate(images_names):
        img = imread(os.path.join(path,img_name))
        cut = img.shape[1] / 2
        img_1 = img[:,:600,:]
        img_2 = img[:,600:,:]
        data.append(img_1)
        data_labels.append(img_2)
        data[i] = data[i][100:-100,100:-100,:]
        data_labels[i] = data_labels[i][100:-100,100:-100,:]
        if sample is not None:
            if sample < i:
                break

    return data,data_labels



#
# train_path = '../idil/forTraining'
# data,labels = load_train_data(train_path)
#
# a = 2
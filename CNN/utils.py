import numpy as np
import os
from skimage.io import imread,imsave
from glob import glob
import math
#CIL_data/aerialOrg/train
#CIL_data/mapsOrg/train
def set_data():
    #to divide train to train and validation
    val = 10
    train = 90
    if not os.path.isdir(os.path.join("aerialOrg","val")):
        os.mkdir(os.path.join("aerialOrg","val"))
        os.mkdir(os.path.join("mapOrg", "val"))

    aerial_name = glob(os.path.join("aerialOrg","train","*.jpg"))
    map_name = glob(os.path.join("mapOrg","train","*.jpg"))
    print(aerial_name)
    trainNum = math.floor((len(aerial_name)/3)*train/100)
    print(trainNum)
    print(range(trainNum,100))
    for ind in range(trainNum,100):
        for thr in [0,1,2]:
            path = os.path.join("mapOrg","train",str(ind+100*thr)+".jpg")
            path_aerial = os.path.join("aerialOrg","train",str(ind+100*thr)+".jpg")
            im = imread(path_aerial)
            im_map = imread(path)
            print(os.path.join("aerialOrg","val","%s.jpg") % str((ind-trainNum)*3+thr))
            imsave(os.path.join("aerialOrg","val","%s.jpg") % str((ind-trainNum)*3+thr),im)
            imsave(os.path.join("mapOrg", "val", "%s.jpg") % str((ind-trainNum)*3+thr), im_map)
            os.remove(os.path.join("aerialOrg","train","%s.jpg") % str(ind+100*thr))
            os.remove(os.path.join("mapOrg", "train", "%s.jpg") % str(ind + 100 * thr))

def getPatches(path,window_size,  output_size):
    #all patches for one image
    step=(window_size-output_size)/2
    im = imread(path)
    img_dim = np.min(im.shape)
    path_num = math.floor((img_dim-window_size)/output_size)+1
    x = [window_size*i for i in range(0,path_num)]
    y = x
    for i in x:
        for j in y:
            path = im[i:i+window_size,j:j+window_size,:]

def getBatch(i,j,window_size,path,path_label):
    image_patchs = []
    image_labels = []
    for ind,im_path in enumerate(path):
        im = imread(im_path)
        im_label = imread(path_label[ind])
        image_patchs.append(im[i:i+window_size,j:j+window_size,:])
        image_labels.append(im_label[i:i + window_size, j:j + window_size, :]/255.)
    return image_patchs, image_labels

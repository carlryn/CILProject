import numpy as np
import os
from skimage.io import imread,imsave, imshow
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

def getBatch(i,j,window_size,output_size, path,path_label):
    im = imread(path[0])
    num = int((window_size - output_size) / 2)
    image_patchs = np.zeros((len(path), window_size, window_size,im.shape[2])).astype(int)
    image_labels = np.zeros((len(path),output_size*output_size)).astype(int)
    for ind,im_path in enumerate(path):
        im = imread(im_path)
        #pad it with smt that makes more sense
        im_pad = np.zeros((num*2+im.shape[0], num*2+int(im.shape[0]),im.shape[2])).astype(int)
        im_pad[num:num+im.shape[0],num:num+im.shape[1],:]=im
        im_label = imread(path_label[ind])
        if len(im_label.shape)>2:
            im_label=im_label[:,:,0]
            #print "wtf"
        batch_files_label = np.reshape(im_label[i:i + output_size, j:j + output_size]/255., [1, -1])

        image_patchs[ind]=im_pad[i:i+window_size,j:j+window_size,:]
        image_labels[ind] = batch_files_label
        #imsave("a.png", image_patchs[ind])
        #imsave("b.png",np.reshape(image_labels[ind]*255,[output_size,output_size]))
    return image_patchs, image_labels

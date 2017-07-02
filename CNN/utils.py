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

#def getBatch(i,j,window_size,output_size, path,path_label):
#    im = imread(path[0])
#    num = int((window_size - output_size) / 2)
#    image_patchs = np.zeros((len(path), window_size, window_size,im.shape[2])).astype(int)
#    image_labels = np.zeros((len(path),output_size*output_size)).astype(int)
#    for ind,im_path in enumerate(path):
#        im = imread(im_path)
#        #pad it with smt that makes more sense
#        im_pad = np.zeros((num*2+im.shape[0], num*2+int(im.shape[0]),im.shape[2])).astype(int)
#        im_pad[num:num+im.shape[0],num:num+im.shape[1],:]=im
#        im_label = imread(path_label[ind])
#        if len(im_label.shape)>2:
#            im_label=im_label[:,:,0]
#            #print "wtf"
#        batch_files_label = np.reshape(im_label[i:i + output_size, j:j + output_size]/255., [1, -1])
#
#        image_patchs[ind]=im_pad[i:i+window_size,j:j+window_size,:]
#        image_labels[ind] = batch_files_label
#        #imsave("a.png", image_patchs[ind])
#        #imsave("b.png",np.reshape(image_labels[ind]*255,[output_size,output_size]))
#    return image_patchs, image_labels

def getBatch(i,j,window_size,output_size, path):
    im = imread(path[0])
    num = int((window_size - output_size) / 2)
    image_patchs = np.ones((len(path), window_size, window_size,im.shape[2]))
    
    for ind,im_path in enumerate(path):
        im = imread(im_path)
        #pad it with smt that makes more sense
        im_pad = np.zeros((num*2+im.shape[0], num*2+int(im.shape[0]),im.shape[2])).astype(int)
        im_pad[:,:,0] = 88.66068507
        im_pad[:,:,1] = 87.67575921
        im_pad[:,:,2] = 77.93427463
        im_pad[num:num+im.shape[0],num:num+im.shape[1],:]=im
        image_patchs[ind]=im_pad[i:i+window_size,j:j+window_size,:]
    return image_patchs
    


def createLabels(path, path_val, img_height, output_height):
    for Currpath in [path, path_val]: 
        print glob(Currpath)
        labels = np.zeros((len(glob(Currpath)), img_height/output_height, img_height/output_height))
        inx = [(name.split(Currpath.split("*")[0])[1]).split(".jpg")[0] for name in glob(Currpath)]
        inx =np.asarray(inx).astype(int)     
        print inx
        for ind, im_path in enumerate(glob(Currpath)):
            im = imread(im_path)
            patch_size = output_height
            for j in range(0, im.shape[1], patch_size):
                for i in range(0, im.shape[0], patch_size):
                    patch = im[i:i + patch_size, j:j + patch_size]
                    labels[inx[ind],i/patch_size,j/patch_size] = patch_to_label(patch)
            path_ = Currpath.split("*")[0]  
        print path_
        np.save(path_+"labels.npy", labels)        
    
def patch_to_label(patch):
   df = np.mean(patch)
   foreground_threshold = 0.25   
   if df > foreground_threshold:
       return 1
   else:
       return 0
    
def fuck():
        path_data_train = os.path.join( "aerialOrg","train","ok", "*.jpg")
        path_map_train = os.path.join( "mapOrg","train","ok", "*.jpg")
        path_data_val = os.path.join( "aerialOrg","val","ok", "*.jpg")
        path_map_val = os.path.join( "mapOrg","val","ok", "*.jpg")
        inx = [(name.split(path_data_train.split("*")[0])[1]).split(".jpg")[0] for name in glob(path_data_train)]
        inx =np.asarray(inx).astype(int)  
        inx_map = [(name.split(path_map_train.split("*")[0])[1]).split(".jpg")[0] for name in glob(path_map_train)]
        inx_map =np.asarray(inx_map).astype(int)
        print inx_map
        print path_data_train.split("ok")[0]
        if np.array_equal(inx_map,inx):
            count = 1
            for ind, im_path in enumerate(glob(path_data_train)):
                print im_path
                if ind == 12:
                    im = imread(im_path)
                    #print glob( path_map_val)[ind]
                    im_map = imread(glob(path_map_train)[ind])
                    print str(ind)
                    imsave(path_data_train.split("ok")[0]+str(ind)+".jpg", im)
                    #imsave(path_map_train.split("ok")[0]+str(ind)+".jpg", im_map)
                    count +=count
        else:
            "ups"

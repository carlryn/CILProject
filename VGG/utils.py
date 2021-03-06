import numpy as np
import os
from skimage.io import imread,imsave
import skimage.transform

inp_window_size = 224
out_window = 25

import warnings
warnings.filterwarnings('ignore')




def load_train_data(train_path,train_path_label,sample=None):
    data = get_data(train_path,sample)
    data_labels = get_labels(train_path_label,sample)
    # permute = np.random.permutation(len(data))
    # data = list(np.asarray(data)[permute])
    # data_labels = list(np.asarray(data_labels)[permute])
    return data, data_labels


def get_data(path,sample=None):
    images_names = os.listdir(path)
    data = []
    for i,img_name in enumerate(images_names):
        img = imread(os.path.join(path,img_name))
        w,h,_ = img.shape
        center_index = (out_window // 2) + 1
        inp_half = (inp_window_size // 2)
        diff = np.abs(center_index - inp_half)
        top_bot = np.zeros((diff,w+diff+diff,_))
        left_right = np.zeros((h,diff,_))
        extra_right = np.zeros((h+diff+diff,1,_)) # These are ugly
        extra_bot = np.zeros((1,w+diff+diff+1,_))   # These are ugly
        img_padded = np.concatenate((left_right,img,left_right),axis=1)
        img_padded = np.concatenate((top_bot,img_padded,top_bot))
        img_padded = np.concatenate((img_padded,extra_right),axis=1)
        img_padded = np.concatenate((img_padded,extra_bot),axis=0)

        for j in range(0,len(img),out_window):
            if sample is not None:
                if len(data) > sample:
                    break
            for k in range(0,len(img),out_window):
                window = img_padded[j:j+inp_window_size,k:k+inp_window_size]
                data.append(window)
    return data

def get_labels(path,sample=None):
    images_names = os.listdir(path)
    data = []
    for i,img_name in enumerate(images_names):
        img = imread(os.path.join(path,img_name))
        for j in range(0, len(img), out_window):
            if sample is not None:
                if len(data) > sample:
                    break
            for k in range(0, len(img), out_window):
                window = img[j:j + out_window, k:k + out_window] / 255
                data.append(window)

    return data

def load_image(img):
    # load image
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img




# def get_data(path,sample=None):
#     images_names = os.listdir(path)
#     data = []
#     for i,img_name in enumerate(images_names):
#         img = imread(os.path.join(path,img_name))
#         w,h,_ = img.shape
#         data.append(img[0:224,0:224])
#
#     return data
#
# def get_labels(path,sample=None):
#     images_names = os.listdir(path)
#     data = []
#     for i,img_name in enumerate(images_names):
#         img = imread(os.path.join(path,img_name))
#         data.append(img[0:224,0:224])
#     return data
#
# def load_for_testing(path,path_gt,sample = None):
#     images_to_predict = get_img_list(path,sample)
#     images_gt = get_img_list(path_gt,sample)
#     return np.asarray(images_to_predict), np.asarray(images_gt), os.listdir(path)


# def get_img_list(path,sample):
#     images_dir = os.listdir(path)
#
#     images = []
#     for i, img_name in enumerate(images_dir):
#         if sample is not None:
#             if i >= sample:
#                 break
#         img = imread(os.path.join(path, img_name))
#         img = get_patch(start, end, img)
#         images.append(img)
#     if sample is not None:
#         return images
#     return images[:sample]

def get_patch(start,end,img):
    patch_size = end - start
    img = img[start:end,start:end]
    return img

def save_image(save_path,orig_images,predicted_imgs, img_gts,img_names):

    for i,img in enumerate(predicted_imgs):
        img_gt = img_gts[i]
        orig_img = orig_images[i]
        img_gt = fix_img_label(img_gt)
        # to_save = np.concatenate((img,img_gt,orig_img),axis=1)
        ones = np.ones((img.shape[0],1))
        to_save = np.concatenate((img,ones,img_gt),axis=1)
        img_name = img_names[i]
        save_img_path = os.path.join(save_path,img_name)
        imsave(save_img_path,to_save)


def fix_img_label(img_label):
    w,h = img_label.shape
    img_label_fix = np.zeros((w, h))
    for j in range(len(img_label)):
        row = img_label[j]
        for k in range(len(row)):
            dp = row[k]
            if dp > 10:
                img_label_fix[j,k] = 1
            else:
                img_label_fix[j,k] = 0

    return img_label_fix
#
# train_path = '../idil/forTraining'
# data,labels = load_train_data(train_path)
#
# a = 2

# Just some code for saving down images and transforming
'''
        for j in range(0, len(img), out_window):
            if sample is not None:
                if len(data) > sample:
                    break
            for k in range(0, len(img), out_window):
                window = img[j:j + out_window, k:k + out_window]
                window = load_image(window)
                imsave('../data/windows/{}_{}.jpg'.format(img_name,k),window)
'''
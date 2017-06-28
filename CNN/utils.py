import numpy as np
import os
from skimage.io import imread,imsave

inp_window_size = 100
out_window = 25
img_shape = 400

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
        img_1 = img[:,:img_shape,:]
        img_2 = img[:,img_shape:,:]
        center_index = out_window // 2
        inp_half = inp_window_size / 2

        for h_pos in range(out_window,img_shape,out_window):
            h_center = h_pos - center_index
            h_beg, h_end = h_center - inp_half, h_center + inp_half
            for w_pos in range(out_window,img_shape,out_window):
                #Find center index
                w_center = w_pos - center_index
                w_beg, w_end = w_center - inp_half, w_center + inp_half
                inp_window = get_window(img,h_beg,h_end,w_beg,w_end)

                # Create input window

        w,h = img_2.shape[:-1]
        # img_label = fix_img_label(img_2)
        img_label = np.zeros((w,h))
        for j in range(len(img_2)):
            row = img_2[j]
            for k in range(len(row)):
                dp = row[k]
                if dp[0] > 10:
                    img_label[j,k] = 1
                else:
                    img_label[j,k] = 0

        data.append(img_1)
        data_labels.append(img_label)
        if sample is not None:
            if sample < i:
                break
    return data,data_labels

'''
This will pad img with zeros to create a matrix shaped (window_size,window_size)
'''
def get_window(img,h_start,h_end,w_start,w_end):
    h,w = img.shape
    h_pad = 0
    w_pad = 0

    if h_start < 0:
        h_pad = np.abs(h_start)

    if w_start < 0:
        w_pad = np.abs(w_start)

    if ( w_start < 0 or h_start < 0 ) and (w_end < w) and (h_end < h):

        # Upper left
        if w_start < 0 and h_start < 0:
            h_pad_matr = np.zeros((h_pad,inp_window_size-w_pad))
            w_pad_matr = np.zeros((inp_window_size,w_pad))

        # Upper
        elif h_start < 0 and w_start >= 0:
            h_pad_matr = np.zeros((h_pad,inp_window_size))

        # Left
        elif h_start >= 0 and w_start < 0:
            w_pad_matr = np.zeros((inp_window_size,w_pad))

    if (h_end >= h  or w_end >=w) and (h_start >= 0) and (w_start >= 0):

        # Down right
        if h_end >= h and w_end >= w:
            h_pad_matr = np.zeros((inp_window_size))



def load_for_testing(path,path_gt,sample = None):
    images_to_predict = get_img_list(path,sample)
    images_gt = get_img_list(path_gt,sample)
    return np.asarray(images_to_predict), np.asarray(images_gt), os.listdir(path)


def get_img_list(path,sample):
    images_dir = os.listdir(path)

    images = []
    for i, img_name in enumerate(images_dir):
        if sample is not None:
            if i >= sample:
                break
        img = imread(os.path.join(path, img_name))
        img = get_patch(start, end, img)
        images.append(img)
    if sample is not None:
        return images
    return images[:sample]

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
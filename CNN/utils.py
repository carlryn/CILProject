import numpy as np
import os
from skimage.io import imread,imsave

start = 100
end = 120

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
        img_1 = img[:,:600,:][100:120,100:120,:]
        img_2 = img[:,600:,:][100:120,100:120,:]

        w,h = img_2.shape[:-1]
        img_label = np.zeros((w,h))
        for j in range(len(img_2)):
            row = img_2[i,j]
            for k in range(len(row)):
                dp = row[k]
                if dp.all() > 10:
                    img_label[i,j] = 1
                else:
                    img_label[i,j] = 0

        data.append(img_1)
        data_labels.append(img_label)
        # data[i] = data[i][100:-100,100:-100,:]
        # data_labels[i] = data_labels[i][100:-100,100:-100,:]
        # data[i] = data[i]
        # data_labels[i] = data_labels[i]

        # for j in range(len(data_labels[i])):
        #     row = data_labels[i][j]
        #     for k in range(len(row)):
        #         dp = row[k]
        if sample is not None:
            if sample < i:
                break
    return data,data_labels


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
            if dp.all() > 10:
                img_label_fix[j,k] = 1
            else:
                img_label_fix[j,k] = 0

    return img_label_fix
#
# train_path = '../idil/forTraining'
# data,labels = load_train_data(train_path)
#
# a = 2
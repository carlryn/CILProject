import numpy as np
import os
from skimage.io import imread,imsave

inp_window_size = 100
out_window = 25
img_shape = 400

def load_train_data(train_path,train_path_label,sample=None):
    data = get_data(train_path,sample)
    data_labels = get_data(train_path_label,sample)
    return data, data_labels

def get_data(path,sample=None):
    images_names = os.listdir(path)
    data = []
    for i,img_name in enumerate(images_names):
        img = imread(os.path.join(path,img_name))
        center_index = out_window // 2
        inp_half = inp_window_size // 2
        for h_pos in range(out_window,img_shape,out_window):
            h_center = h_pos - center_index - 1 #-1 to get the right index
            h_beg, h_end = h_center - inp_half, h_center + inp_half
            for w_pos in range(out_window,img_shape,out_window):
                #Find center index
                w_center = w_pos - center_index
                w_beg, w_end = w_center - inp_half, w_center + inp_half
                inp_window = get_window(img,h_beg,h_end,w_beg,w_end)
                data.append(inp_window)
                # Create input window
                if sample is not None:
                    if sample < i:
                        break

    return data


def extract_patch(img,h_start,h_end,w_start,w_end,dims):
    if dims is 2:
        return img[h_start:h_end,w_start:w_end]
    if dims is 3:
        return img[h_start:h_end,w_start:w_end,:]



'''
This will pad img with zeros to create a matrix shaped (window_size,window_size)
8 cases around the matrix img can occur
'''
def get_window(img,h_start,h_end,w_start,w_end):

    if len(img.shape) > 2:
        h,w,_ = img.shape
    else:
        h,w = img.shape

    dims = len(img.shape)
    h_pad = 0
    w_pad = 0
    if dims is 2:
        new_img = np.zeros((inp_window_size,inp_window_size))
    if dims is 3:
        new_img = np.zeros((inp_window_size,inp_window_size,3))


    '''MAKE ALL INT MFO'''

    new_h_start = h_start
    new_h_end = h_end
    new_w_start = w_start
    new_w_end = w_end

    if ( w_start < 0 or h_start < 0 ) and (w_end < w) and (h_end < h):

        if h_start < 0:
            new_h_start = int(np.abs(h_start))

        if w_start < 0:
            new_w_start = int(np.abs(w_start))

        # Upper left
        if w_start < 0 and h_start < 0:
            # new_img[new_h_start:,new_w_start:] = img[0:h_end,0:w_end]
            new_img[new_h_start:,new_w_start:] = extract_patch(img,0,h_end,0,w_end,dims)


        # Upper mid
        elif h_start < 0 and w_start >= 0:
            new_img[new_h_start:,:] = img[0:h_end,w_start:w_end]

        # Left Mid
        elif h_start >= 0 and w_start < 0:
            # new_img[:,new_h_start] = img[h_start:h_end,:w_end]
            new_img[:,new_w_start:] = extract_patch(img,h_start,h_end,0,w_end,dims)


    elif (h_end >= h or w_end >= w) and (h_start >= 0) and (w_start >= 0):

        if h_end >= h:
            new_h_end = h_end - h

        if h_end >= w:
            new_w_end = w_end - w

        # Down right
        if h_end >= h and w_end >= w:
            new_img[0:new_h_end,0:new_w_end] = img[h_start:,w_start:]

        # Down Mid
        if h_end >= h and w_start >= 0 and w_end < w:
            new_img[0:new_h_end,:] = img[h_start:h,w_start:w_end]

        # Right Mid
        if w_end >= w and h_start >= 0 and h_end < h:
            new_img[:,0:w_end] = img[h_start:h_end,w_start:]

    # Upper right
    elif h_start < 0 and w_end >= w:
        new_img[np.abs(h_start):,0:inp_window_size-(w_end - w)] = img[0:h_end,w_start:]

    # Down left
    elif h_end >= h and w_start < 0:
        new_img[0:h_end-h,np.abs(w_start):] = img[h_start:,0:w_end]

    else:
        new_img = img[h_start:h_end,w_start:w_end]

    return new_img

    # if ( w_start < 0 or h_start < 0 ) and (w_end < w) and (h_end < h):
    #
    #     if h_start < 0:
    #         h_pad = np.abs(h_start)
    #
    #     if w_start < 0:
    #         w_pad = np.abs(w_start)
    #
    #     # Upper left
    #     if w_start < 0 and h_start < 0:
    #         h_pad_matr = np.zeros((h_pad,inp_window_size-w_pad))
    #         w_pad_matr = np.zeros((inp_window_size,w_pad))
    #
    #     # Upper Mid
    #     elif h_start < 0 and w_start >= 0:
    #         h_pad_matr = np.zeros((h_pad,inp_window_size))
    #
    #     # Left Mid
    #     elif h_start >= 0 and w_start < 0:
    #         w_pad_matr = np.zeros((inp_window_size,w_pad))

    # if (h_end >= h  or w_end >= w) and (h_start >= 0) and (w_start >= 0):
    #
    #     if h_end >= h:
    #         h_pad = h_end - h
    #
    #     if h_end >= w:
    #         w_pad = w_end - w
    #
    #     # Down right
    #     if h_end >= h and w_end >= w:
    #         h_pad_matr = np.zeros((h_pad,inp_window_size - w_pad))
    #         w_pad_matr = np.zeros((inp_window_size,w_pad))
    #
    #     # Down Mid
    #     if h_end >= h and w_start >= 0 and w_end < w:
    #         h_pad_matr = np.zeros((h_pad,inp_window_size))
    #
    #     # Right Mid
    #     if w_end >= w and h_start >= 0 and h_end < h:
    #         w_pad_matr = np.zeros((inp_window_size,w_pad))




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
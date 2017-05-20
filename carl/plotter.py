from skimage.io import imread, imshow # for reading images
import matplotlib.pyplot as plt # for showing plots
from skimage.measure import label # for labeling regions
import numpy as np # for matrix operations and array support
from skimage.color import label2rgb # for making overlay plots
from skimage.color import rgb2gray, rgb2hsv # making grayscale images
import matplotlib.patches as mpatches # for showing rectangles and annotations
from skimage.morphology import opening, closing # for removing small objects
from skimage.morphology import medial_axis # for finding the medial axis and making skeletons
from skimage.morphology import skeletonize, skeletonize_3d # for just the skeleton code
import pandas as pd # for reading the swc files (tables of somesort)
import os
from glob import glob # for lists of files
import  utils
from copy import  copy


path ='../maps/train/groundtruth/320.jpg'
im_path = '../maps/train/groundtruth/320.jpg'

data = imread(path)

#Arrow  is < 210
zeros = np.zeros(3)
subtract = np.array([250,250,250])

# for i,row in enumerate(data):
#     for j,column in enumerate(row):
#         data[i,j] = zeros if (column - subtract <= zeros).all() else column

#imshow(data)


# im_path = data
# im_data = imread(path)
#mk_data = read_swc(mask_path)
# def thresh_image(in_img):
#     v_img = rgb2hsv(im_data)[:,:,2]
#     th_img = v_img>0.5
#     op_img = opening(th_img)
#     cl_img = closing(op_img)
#     return cl_img
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (10, 2))
# ax1.imshow(im_data)
# data = thresh_image(im_data)
# ax2.imshow(t_img, cmap = 'bone')
# ax2.set_title('Thresheld Image')

# ax3.imshow(t_img,cmap = 'bone')
# ax3.scatter(mk_data['x'], mk_data['y'], s = mk_data['width'])
# ax3.set_title('Image with Overlay')



# skel, distance = medial_axis(t_img, return_distance=True)
# dist_on_skel = np.zeros_like(distance)
# dist_on_skel[skel] = distance[skel]
# dist_on_skel[skel==0] = np.nan
#
#
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (10, 2))
# ax1.imshow(im_data)
# # ax2.scatter(mk_data['x'], mk_data['y'], s = mk_data['width']/20,alpha = 1)
# ax2.imshow(distance, cmap = 'magma', vmin = 0, vmax = 10)
# ax2.set_title('Distance Map with Overlay')
#
# ax3.imshow(dist_on_skel,cmap = 'jet', alpha =1.0)
# ax3.set_title('Skeleton with Overlay')



def threshold_img(img_data):
    a = 2
    tmp_data = copy(img_data)
    for i, row in enumerate(img_data):
        for j, col in enumerate(img_data):

            if img_data[i,j][2]  < 180 and img_data[i,j][0] > 210:
                tmp_data[i,j] = img_data[i,j]
            else:
                tmp_data[i, j] = [0,0,0]

            if (img_data[i,j]  < 210).all() and (img_data[i,j][0] > 190).all():
                tmp_data[i, j] = [255,255,255]

    return tmp_data



#
th_data = threshold_img(data)
th_data = utils.cclabel_image(th_data,im_path=im_path)
plt.imshow(data)
plt.show()
plt.imshow(th_data)
#
plt.show()
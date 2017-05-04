from skimage.io import imread, imshow, imsave # for reading images
from skimage.exposure import equalize_hist
from scipy.misc import toimage
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt # for showing plots
from skimage.measure import label # for labeling regions
import numpy as np # for matrix operations and array support
from skimage.color import label2rgb # for making overlay plots
from skimage.color import rgb2gray, rgb2hsv # making grayscale images
import matplotlib.patches as mpatches # for showing rectangles and annotations
from skimage.morphology import opening, closing, binary_dilation # for removing small objects
from skimage.morphology import medial_axis # for finding the medial axis and making skeletons
from skimage.morphology import skeletonize, skeletonize_3d # for just the skeleton code
import pandas as pd # for reading the swc files (tables of somesort)
import os
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import disk
from skimage.filters.rank  import median
from glob import glob # for lists of files
from skimage import img_as_uint

def readImg():
    DATA_ROOT = './maps/train/'
    image_files = glob(os.path.join(DATA_ROOT, '*.jpg'))
    print(image_files)
    a = imread(image_files[1])
    if not os.path.exists(os.path.join(DATA_ROOT,"groundtruth")):
        os.mkdir(os.path.join(DATA_ROOT,"groundtruth"))
        os.mkdir(os.path.join(DATA_ROOT,"images"))
        inx = [int((i.split('./maps/train\\',1)[1]).split(".jpg",1)[0]) for i in image_files]
        print(inx)
        print(a.shape)
        for i, img_file in enumerate(image_files):
            print(img_file)
            img = imread(img_file)
            aerial  = img[:,0:int(img.shape[1]/2),:]
            map     = img[:,int(img.shape[1]/2):,:]
            imsave(os.path.join(DATA_ROOT,"groundtruth",str(inx[i])+".jpg"), map)
            imsave(os.path.join(DATA_ROOT,"images",str(inx[i]) + ".jpg"), aerial)
            b = 0
        #image_mask_files = [imread(c_file) for c_file in image_files]
def thresh_image(in_img):
    ones_arr = np.ones(in_img.shape)
    zeros_arr = np.zeros(in_img.shape[0:2])
    print(zeros_arr.shape)
    # dist = cdist(ones_arr, in_img, metric = "euclidean")
    for i in range(0, in_img.shape[0]):
        for j in range(0, in_img.shape[1]):
            zeros_arr[i, j] = np.sqrt(
                np.square(1 - in_img[i, j, 0]) + np.square(1 - in_img[i, j, 1]) + np.square(1 - in_img[i, j, 2]))
    # print(zeros_arr)
    # plt.imshow(zeros_arr, cmap='gray')
    # plt.show()
    # print(v_img)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))
    th_img = 0.3 > zeros_arr
    ax1.imshow(zeros_arr, cmap='gray')
    # plt.show()
    op_img = closing(th_img)
    ax2.imshow(op_img, cmap='gray')
    # plt.show()
    cl_img = opening(op_img)
    ax3.imshow(cl_img, cmap='gray')
    plt.show()
    return cl_img
#expend histogram
#do connectivity
DATA_ROOT = './maps/train/groundtruth'
if not os.path.exists(os.path.join(DATA_ROOT, "processed")):
    os.mkdir(os.path.join(DATA_ROOT, "processed"))
image_files = glob(os.path.join(DATA_ROOT, '*.jpg'))
im_path = image_files[57]
inx = [int((i.split('./maps/train/groundtruth\\',1)[1]).split(".jpg",1)[0]) for i in image_files]
print(inx)
#im_path = "./maps/train/groundtruth\\618.jpg"
for k in range(0,len(inx)):
    im_path = image_files[k]
    print(im_path)
    im_data = imread(im_path)
    #fig, ((ax1, ax2, ax3), (ax4,ax5,ax6)) = plt.subplots(2, 3)

    for i in range(0, im_data.shape[0]):
        for j in range(0, im_data.shape[1]):
            #orange
            if np.sqrt(np.square(251 - im_data[i, j, 0]) + np.square(158 - im_data[i, j, 1]) + np.square(34- im_data[i, j, 2])) <80:
                im_data[i,j,:] = 255
            #yellow
            elif np.sqrt(np.square(254 - im_data[i, j, 0]) + np.square(224 - im_data[i, j, 1]) + np.square(164- im_data[i, j, 2])) <40:
                im_data[i,j,:] = 255
            #elif (((im_data[i, j, :]-[200,200,200])<0).any()):
            #    im_data[i, j, :] = [232, 229, 224]
                #threshold
            #if np.sqrt(np.square(255 - im_data[i, j, 0]) + np.square(255 - im_data[i, j, 1]) + np.square(255 - im_data[i, j, 2])) > 40:
            #    im_data[i, j, :] = 0
    a = rgb2gray(im_data)
    b = a
    b[a<0.96] = 0
    imshow(b)
    plt.show()
    b = median(b, disk(3))
    imshow(b)
    plt.show()
    img5 = binary_dilation(b)
    imshow(img5)
    plt.show()
    img5 = binary_dilation(img5)
    imshow(img5)
    plt.show()
    img6 = binary_dilation(img5)
    imshow(img6)
    plt.show()
    strr = os.path.join(DATA_ROOT, "processed",str(inx[k])+".png")
    imsave(strr,img_as_uint(img6))
#ax1.imshow(a, cmap='gray')
#ax2.hist(a)
# ax3.imshow(b, cmap='gray')
# ax6.imshow(median(b, disk(4)))
# b = median(b, disk(4))
# cl_img = closing(b)
# #cl_img = opening(op_img)
# ax4.imshow(cl_img, cmap='gray')
# ax5.imshow(imread(im_path))
# imsave("current.png",cl_img)
# #ax6.imshow(median(cl_img, disk(3)))
# plt.show()
# asd = 0


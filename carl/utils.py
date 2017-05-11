import os
from glob import glob # for lists of files
from skimage.io import imread, imshow, imsave # for reading images

from skimage.morphology import dilation, binary_dilation, closing, opening
import numpy as np
from PIL import Image, ImageDraw
import cclabel
from itertools import product
from skimage.color import rgb2gray, rgb2hsv


def combineImg(names):
    images = map(Image.open, names)
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in names:
        img = Image.open(im)
        new_im.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    return new_im



def binary_th(im_data, inx, DATA_ROOT,im_path, save = True):
    for i in range(0, im_data.shape[0]):
        for j in range(0, im_data.shape[1]):
            # orange
            if np.sqrt(np.square(251 - im_data[i, j, 0]) + np.square(158 - im_data[i, j, 1]) + np.square(
                            34 - im_data[i, j, 2])) < 80:
                im_data[i, j, :] = 0
            # yellow
            elif np.sqrt(np.square(254 - im_data[i, j, 0]) + np.square(224 - im_data[i, j, 1]) + np.square(
                            164 - im_data[i, j, 2])) < 40:
                im_data[i, j, :] = 0
            elif np.sqrt(np.square(255 - im_data[i, j, 0]) + np.square(255 - im_data[i, j, 1]) + np.square(
                            255 - im_data[i, j, 2])) > 20:
                im_data[i, j, :] = 255
            else:
                im_data[i, j, :] = 0



    imsave("example.png", im_data)
    img = cclabel.main("example.png")
    data = img.load()
    im = Image.new("RGB", img.size, "black")
    data_im = im.load()
    width, height = img.size
    for y, x in product(range(height), range(width)):
        if data[x, y] != (0, 0, 0):
            data_im[x, y] = (255, 255, 255)
    im.save('example2.png')
    im = imread('example2.png')
    im = closing(im)
    imsave("example2_b.png", im)
    img2 = cclabel.main("example2_b.png")
    data_im2 = img2.load()
    for y, x in product(range(height), range(width)):
        if data_im2[x, y] == (0, 0, 0):
            data_im2[x, y] = (255, 255, 255)
        else:
            data_im2[x, y] = (0, 0, 0)
    img2.save('out3.jpg')
    # combine image
    strr = [im_path, 'out3.jpg']
    new_im = combineImg(strr)
    if save == True:
        new_im.save(os.path.join(DATA_ROOT, "processed", str(inx) + '.jpg'))
    else:
        return new_im


def cclabel_image(im_data, im_path = '../maps/train/groundtruth/1093.jpg'):
    imsave("example.png", im_data)
    img = cclabel.main("example.png")
    data = img.load()
    im = Image.new("RGB", img.size, "black")
    data_im = im.load()
    width, height = img.size
    for y, x in product(range(height), range(width)):
        if data[x, y] != (0, 0, 0):
            data_im[x, y] = (255, 255, 255)
    im.save('example2.png')
    im = imread('example2.png')
    im = closing(im)  #Dilation followed by erosion, removes small black spots
    imsave("example2_b.png", im)
    img2 = cclabel.main("example2_b.png")
    data_im2 = img2.load()
    for y, x in product(range(height), range(width)):
        if data_im2[x, y] == (0, 0, 0):
            data_im2[x, y] = (255, 255, 255)
        else:
            data_im2[x, y] = (0, 0, 0)

    # return data_im2
    img2.save('out3.jpg')
    #combine image
    strr = [im_path, 'out3.jpg']
    new_im = combineImg(strr)
    return new_im
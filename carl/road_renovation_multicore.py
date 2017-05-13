from skimage.io import imread, imshow, imsave
import matplotlib.pyplot as plt
from copy import copy
from math import pi, cos, sin
from multiprocessing import Process, Manager
import numpy as np
import time
import os


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='../idil/processed' ,help='skip the feature extraction')
parser.add_argument('--n_images', default=1, type=int, help='Should we run on all images')
parser.add_argument('--n_processes', default=4, type=int, help='How many threads will run the restorate method')
parser.add_argument('--save_dir', default='../data/restoration', help="Folder for saving processed images")
args = parser.parse_args()


'''
The pixel being looked at is the one that is in the middle. The ray will go both "back and forth".
NOTE! This method is using multithreading. Change n_processes for as many process as you would like.
'''
def restorate(img_data, pixel_radius, directions,score_min,index_start, index_stop, return_dict):
    h, w, d = img_data.shape
    img_data_new = copy(img_data)[index_start:index_stop]
    angles = get_angles(directions)
    steps = pixel_radius

    #Iterate over the pixels,
    for i in range(index_start, index_stop):
        row = img_data[i]
        #print("Row:", i)
        for j, pixel in enumerate(row):
            scores = []
            angle_steps = []
            for a,angle in enumerate(angles):
                scores.append(0)
                angle_steps.append(0)
                for k in range(2):
                    y_step = cos(angle) * pixel_radius/steps if k == 0 else -cos(angle) * pixel_radius/steps
                    x_step = sin(angle) * pixel_radius/steps if k == 0 else -sin(angle) * pixel_radius/steps
                    y_pos = j
                    x_pos = i
                    for _ in range(pixel_radius):
                        x_pos += x_step
                        y_pos += y_step
                        x_index = int(x_pos)
                        y_index = int(y_pos)
                        if x_index >= 0 and y_index >=0 and x_index < h and y_index < w:
                            angle_steps[a] += 1
                            pixel = img_data[x_index, y_index]
                            if (pixel >= 100).all():
                                scores[a] += 1

            #Pick angle with the highest score
            for a in range(len(angles)):
                scores[a] /= angle_steps[a]

            best_score = scores[np.argmax(scores)]
            if best_score > score_min:
                img_data_new[i - index_start, j] = [255, 255, 255]
            else:
                img_data_new[i -  index_start, j] = [0, 0, 0]

    return_dict[index_start] = img_data_new

def get_angles(directions):
    max_degr = pi
    angles = []
    per_angle = max_degr/directions
    for i in range(directions):
        angles.append(i*per_angle)
    return angles

def find_pixel_goal(h, w,angle, pixel_radius):
    x = np.cos(angle) * pixel_radius
    y = np.sin(angle) * pixel_radius
    y_pixels  = y/h * 100
    x_pixels = x/w * 100
    return x_pixels, y_pixels

"""
1. Check that the directory exist
"""
if os.path.exists(args.save_dir) == False:
    os.mkdir(args.save_dir)

"""
2. Get list of all images in the directory
"""
images = os.listdir(args.data_path)[:args.n_images]


"""
3. Create lists for testing the different parameters, e.g score, pixel_radius, directions
"""
# pixel_radius_list = [20,30,40,50,60,70,80]
# scores = [0.4,0.5,0.6,0.7,0.8]
# directions = 60
pixel_radius_list = [20]
scores = [0.4]
directions = 2

#Create folders for the different pixel radiuses
for i, radius in enumerate(pixel_radius_list):
    for j, score in enumerate(scores):
        dir = os.path.join(args.save_dir,"radius_{}_score_{}".format(radius,score))
        if os.path.exists(dir) == False:
            os.mkdir(dir)

for i, img_path in enumerate(images):
    img_full_path = os.path.join(args.data_path, img_path)
    img_data = imread(img_full_path)
    img_data = img_data[:, img_data.shape[1] // 2:]
    start = time.time()
    for j,pixel_radius in enumerate(pixel_radius_list):
        for k, score in enumerate(scores):
            manager = Manager()
            return_dict = manager.dict()
            n_processes = args.n_processes
            x,y,d = img_data.shape
            interval = x//n_processes #Make sure image is dividable with n_processes
            processes = []
            for l in range(n_processes):
                p = Process(target=restorate, args=(img_data,pixel_radius, directions, score,l*interval,(l + 1)*interval, return_dict))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            is_set = False
            for key,value in sorted(return_dict.items()):
                if is_set == False:
                    img_new = value
                    is_set = True
                else:
                    img_new = np.concatenate((img_new, value), axis = 0)
            save_dir_path = os.path.join(args.save_dir,"radius_{}_score_{}".format(radius,score))
            save_img_path = os.path.join(save_dir_path,img_path)
            imsave(save_img_path,img_new)

    end = time.time()
    print("Total time image:", img_path ,end-start, ", n_processes:", n_processes)


from skimage.io import imread, imshow, imsave
import matplotlib.pyplot as plt
from copy import copy
from math import pi, cos, sin
from multiprocessing import Process, Manager
import numpy as np
import time
import os
from scipy.spatial.distance import cdist
import math

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='../idil/processed' ,help='skip the feature extraction')
# parser.add_argument('--data_path', default='../data' ,help='skip the feature extraction') # For test
parser.add_argument('--n_images', default=1, type=int, help='Should we run on all images')
parser.add_argument('--n_processes', default=4, type=int, help='How many threads will run the restorate method')
parser.add_argument('--save_dir', default='../data/restoration', help="Folder for saving processed images")
args = parser.parse_args()

'''
The pixel being looked at is the one that is in the middle. The ray will go both "back and forth".
NOTE! This method is using multithreading. Change n_processes for as many process as you would like.
'''
def restorate(img_data,pixels,pixel_radius,directions,score_min,return_dict, process_name):
    h, w, d = img_data.shape
    angles = get_angles(directions)
    steps = pixel_radius
    pixels_to_transform = list()
    #Iterate over the pixels,
    for coord in pixels:
        x = coord[0]
        y = coord[1]
        scores = []
        total_steps = pixel_radius * 2
        for a,angle in enumerate(angles):
            scores.append(0)
            for k in range(2):
                y_step = cos(angle) * pixel_radius/steps if k == 0 else -cos(angle) * pixel_radius/steps
                x_step = sin(angle) * pixel_radius/steps if k == 0 else -sin(angle) * pixel_radius/steps
                y_pos = y
                x_pos = x
                for _ in range(pixel_radius):
                    x_pos += x_step
                    y_pos += y_step
                    x_index = int(x_pos)
                    y_index = int(y_pos)
                    if x_index >= 0 and y_index >=0 and x_index < h and y_index < w:
                        pixel = img_data[x_index, y_index]
                        if (pixel >= 100).all():
                            scores[a] += 1
            scores[a] /= total_steps
            if scores[a] > score_min:
                pixels_to_transform.append(coord)
                break # If found angle with score > score_min : skip rest of angles (y)

    return_dict[process_name] = pixels_to_transform

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

'''
Evaluating for all pixels is not necessary.
Can find quick first intuition if transformation to white is possible by looking at nearest neighbour distance.
'''
def find_evaluation_coords(img_data, pixel_radius, score):
    n,d,_ = img_data.shape
    min_white_pixels = int(2*pixel_radius*score)
    # Set up two lists with coords
    whites = list()
    blacks = list()

    start = time.time()

    for i in range(pixel_radius, n-pixel_radius,1):
        for j in range(pixel_radius, d - pixel_radius, 1):
            dp = img_data[i,j]
            if (dp < 100).all():
                blacks.append([i,j])
            else:
                whites.append([i,j])


    splits = 40 #Calculating pairwise distances uses A LOT of RAM.
    # Split the Blacks into many pieces to require less RAM.
    blacks_per_split = math.ceil(len(blacks)/splits)
    blacks_in_pieces = [blacks[i:i + blacks_per_split]
                          if len(blacks) > (i + blacks_per_split)
                          else blacks[i:-1]
                          for i in range(0, len(blacks), blacks_per_split)]

    pixels_for_evaluation = list()
    whites_arr = np.asarray(whites)
    for j, blacks_split in enumerate(blacks_in_pieces):
        blacks_arr = np.asarray(blacks_split)
        if len(blacks_arr) > 0 and len(whites_arr) > 0:
            distances = cdist(blacks_arr,whites_arr)
            n,d = distances.shape
            for i in range(n):
                count = len(np.where(distances[i] < pixel_radius)[0])

                if count > min_white_pixels:
                    pixels_for_evaluation.append(blacks_split[i])

    end = time.time()

    print("Filtering pixels time:", end-start)
    p_eval = float(len(pixels_for_evaluation) / (n-pixel_radius*2)**2 ) * 100
    print("p_eval:", p_eval)
    return pixels_for_evaluation

def main():
    """
    1. Check that the directory exist
    """
    if os.path.exists(args.save_dir) == False:
        os.mkdir(args.save_dir)

    """
    2. Get list of all images in the directory
    """
    images = os.listdir(args.data_path)
    #images = ['1065.jpg', '1028.jpg','144.jpg','249.jpg','250.jpg','255.jpg','1041.jpg','1051.jpg','1029.jpg']
    # images = ['road_renovation_test_image.png']
    save_path = '../data/restoration/radius_65_score_0.65' #TODO this is hard coded atm
    processed_images = os.listdir(save_path)
    images = [x for x in images if not x in processed_images]
    """
    3. Create lists for testing the different parameters, e.g score, pixel_radius, directions
    """
    #1011
    pixel_radius_list = [65]
    scores = [0.65]
    directions = 60


    #Create folders for the different pixel radiuses
    for i, radius in enumerate(pixel_radius_list):
        for j, score in enumerate(scores):
            dir = os.path.join(args.save_dir,"radius_{}_score_{}".format(radius,score))
            if os.path.exists(dir) == False:
                os.mkdir(dir)

    for i, img_path in enumerate(images):
        print('image:', img_path)
        img_full_path = os.path.join(args.data_path, img_path)
        img_data = imread(img_full_path)
        img_data = img_data[:, img_data.shape[1] // 2:]
        start = time.time()
        for j,pixel_radius in enumerate(pixel_radius_list):
            for k, score in enumerate(scores):

                manager = Manager()
                return_dict = manager.dict()
                n_processes = args.n_processes
                pixels_to_evaluate = find_evaluation_coords(img_data, pixel_radius, score)
                if len(pixels_to_evaluate) < n_processes:
                    continue
                pixels_per_process = math.ceil(len(pixels_to_evaluate)/n_processes)
                distributed_pixels = [pixels_to_evaluate[i:i+pixels_per_process]
                                      if len(pixels_to_evaluate) > (i+pixels_per_process)
                                      else pixels_to_evaluate[i:-1]
                                      for i in range(0,len(pixels_to_evaluate),pixels_per_process)]
                x,y,d = img_data.shape
                interval = x//n_processes #Make sure image is dividable with n_processes
                processes = []
                for l in range(n_processes):
                    p = Process(target=restorate, args=(img_data,distributed_pixels[l],pixel_radius, directions, score ,return_dict, l))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
                img_new = copy(img_data)
                for key,pixels in sorted(return_dict.items()):
                    for coord in pixels:
                        x = coord[0]
                        y = coord[1]
                        img_new[x,y] = [255,255,255]

                save_dir_path = os.path.join(args.save_dir,"radius_{}_score_{}".format(pixel_radius,score))
                save_img_path = os.path.join(save_dir_path,img_path)
                imsave(save_img_path,img_new)

        end = time.time()
        print("Total time image:", img_path ,end-start, ", n_processes:", n_processes)

main()

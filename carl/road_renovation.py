import numpy as np
from skimage.io import imread, imshow
import matplotlib.pyplot as plt


'''
The pixel being looked at is the one that is in the middle. The ray will go both "back and forth".
'''
def renovate(img_data, pixel_radius = 40, directions = 40):
    h, w = img_data.shape
    angles = get_angles(directions)

    # pixel_goals = []
    # for angle in angles:
    #     pixel_goals.append(find_pixel_goal(h,w,angle, pixel_radius))

    #Iterate over the pixels,
    for i, row in enumerate(img_data):
        for j, pixel in enumerate(row):
            scores = []
            # Ugly duplicated code
            for angle in angles:
                x_step = np.cos(angle)
                y_step = np.sin(angle)
                x_pos = 0
                y_pos = 0
                for k in range(pixel_radius):
                    x_pos += x_step
                    y_pos += y_step
                    x_index = int(x_pos)
                    y_index = int(y_pos)
                    if x_index >= 0 and y_index >=0 and x_index < w and y_index < h:
                        pixel = img_data[x_index, y_index]
                        a = 0

                x_step = -np.cos(angle)
                y_step = -np.sin(angle)
                x_pos = 0
                y_pos = 0
                for k in range(pixel_radius):
                    x_pos += x_step
                    y_pos += y_step
                    x_index = int(x_pos)
                    y_index = int(y_pos)
                    if x_index >= 0 and y_index >=0 and x_index < w and y_index < h:
                        pixel = img_data[x_index, y_index]








def get_angles(directions):
    max_degr = 180
    degrees = [max_degr/i for i in range(directions)]
    return degrees

# def find_unit_goal(angles, unit_radius):
#     x = np.cos(degree) * unit_radius
#     y = np.sin(degree) * unit_radius
#     return x,y

def find_pixel_goal(h, w,angle, pixel_radius):
    x = np.cos(angle) * pixel_radius
    y = np.sin(angle) * pixel_radius
    y_pixels  = y/h * 100
    x_pixels = x/w * 100
    return x_pixels, y_pixels





data_test_path = '../data/road_renovation_test_image'
img_data = imread(data_test_path)
img_renovated = renovate(img_data)

import numpy as np
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from copy import copy
from math import pi, cos, sin
import time

'''
The pixel being looked at is the one that is in the middle. The ray will go both "back and forth".
'''
def restorate(img_data, pixel_radius = 40, directions = 40):
    img_data_new = copy(img_data)
    h, w, d = img_data_new.shape
    angles = get_angles(directions)
    steps = pixel_radius
    # pixel_goals = []
    # for angle in angles:
    #     pixel_goals.append(find_pixel_goal(h,w,angle, pixel_radius))

    #Iterate over the pixels,
    for i, row in enumerate(img_data_new):
        print("Row:", i)
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
                            #img_data[x_index,y_index] = [0,0,125] #For coloring when testing

            #Pick angle with the highest score
            for a in range(len(angles)):
                scores[a] /= angle_steps[a]

            best_score = scores[np.argmax(scores)]
            if best_score > 0.55:
               # print("Score higher than 0.6 pixel:", i, j, "Score:", best_score)
                img_data_new[i, j] = [255, 255, 255]
            else:
                img_data_new[i, j] = [0, 0, 0]

    return img_data_new


def get_angles(directions):
    max_degr = pi
    angles = []
    per_angle = max_degr/directions
    for i in range(directions):
        angles.append(i*per_angle)
    return angles

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
img_data = img_data[25:]
# The img is a screenshot, clean it up
for i, row in enumerate(img_data):
    for j, data in enumerate(row):
        if (data < 100).all():
            img_data[i,j] = [0,0,0]


start = time.time()
img_restorated = restorate(img_data, pixel_radius=20, directions=10)
end = time.time()

print("Total time:", end-start)

fig = plt.figure()
a=fig.add_subplot(1,2,1)
img = img_data
imgplot = plt.imshow(img_data)
a.set_title('Before')
a=fig.add_subplot(1,2,2)
imgplot = plt.imshow(img_restorated)
imgplot.set_clim(0.0,0.7)
a.set_title('After')

imshow(img_restorated)
plt.show()

a = 2

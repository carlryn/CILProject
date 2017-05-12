import os
print(os.getcwd())
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
# img_data = imread('../data/road_renovation_test_image')
# imshow(img_data[25:])
# plt.show()


import numpy as np
from copy import copy
from multiprocessing import Process, Manager
import time

manager = Manager()
return_dict = manager.dict()

def what(index,jump, thread_name, ones, return_dict):
    #ones_copy = copy(ones)
    n, d = ones.shape
    for i in range(index,n,jump):
        print(i)
        for j in range(d):
            a = 2
        name = thread_name
    return_dict[i] = thread_name

ones = np.ones((8,100))

# start = time.time()
# what(0,1,'One thread', ones)
# end = time.time() - start
# print("one ... ", end)


threads = 4
start = time.time()

processes = []

for i in range(threads):
    p = Process(target=what, args=(i,threads,'Thread_{}'.format(i), ones, return_dict))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

end = time.time() - start

print("4 ", end)
a = 2

print(return_dict.items())
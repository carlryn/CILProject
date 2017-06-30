import os

data = "../data/training/images"
labels = "../data/windows"

d = os.listdir(data)
l = os.listdir(labels)

for i in range(len(d)):
    if d[i] != l[i]:
        print(False)
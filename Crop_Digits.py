import matplotlib.image as np_image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

mnist = input_data.read_data_sets("Dataset/", one_hot=True)
from PIL import Image
import cv2


def convert(image):
    A = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3
    B = A.copy()
    k = np.mean(A[:]) * 0.82
    B[A < k] = 255
    B[A >= k] = 0
    return B


image_1 = np_image.imread("Images(Crop_Digits)/1.png")
image_2 = convert(image_1)


check = np.zeros(shape=[image_2.shape[0], image_2.shape[1]])
stack = [[], []]

dx = [-1, -1, -1, 0, 0, 1, 1, 1]
dy = [-1, 0, 1, -1, 1, -1, 0, 1]

px = [[]]
py = [[]]

count = 0


def Dfs(i, j, cou):
    check[i, j] = 1
    stack[0].append(i)
    stack[1].append(j)
    while (len(stack[0]) > 0):
        u = stack[0].pop()
        v = stack[1].pop()
        px[cou].append(u)
        py[cou].append(v)
        for k in range(8):
            x = u + dx[k]
            y = v + dy[k]
            if (x < image_2.shape[0]) and (y < image_2.shape[1]) and (x >= 0) and (y >= 0) and (check[x, y] == 0) and (
                image_2[x, y] == 255):
                check[x, y] = 1
                stack[0].append(x)
                stack[1].append(y)

for j in range(image_2.shape[1]):
    for i in range(image_2.shape[0]):
        if (check[i, j] == 0) and (image_2[i, j] == 255):
            count += 1
            px.append([])
            py.append([])
            Dfs(i, j, count)

def add_padding(image):
    W = image.shape[0] // 4
    H = image.shape[1] // 4
    W_1 = image.shape[0] + W * 2
    H_1 = image.shape[1] + H * 2
    A = np.zeros(shape=[W_1, H_1])
    A[W:(W + image.shape[0]), H:(H + image.shape[1])] = image
    return A

files = os.listdir('Images')
for file in files:
    os.remove('Images/'+file)

for i in range(1, count + 1):
    maxx = 0
    maxy = 0
    minx = 1000000
    miny = 1000000
    for j in range(len(px[i])):
        maxx = max(maxx, px[i][j])
        minx = min(minx, px[i][j])
        maxy = max(maxy, py[i][j])
        miny = min(miny, py[i][j])
    if ((maxx - minx) * (maxy - miny) >= 1500) and ((maxx - minx) // (maxy - miny) >= 1) and (
            (maxx - minx) // (maxy - miny) <= 5):
        tmp = image_2[minx: maxx, miny: maxy]
        tmp2 = tmp
        new_image = Image.fromarray(tmp2)
        x =  new_image.resize(size=(32, 64))
        tmp3 = np.asarray(x)
        plt.imshow(tmp3, cmap='gray')
        plt.show()
        cv2.imwrite("Images/%d.png" % i, tmp3)


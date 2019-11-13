#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   demo.py
@time    :   2019/11/13 12:20:01
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   pencil drawing
"""

__author__ = "XiaoY"

# %%
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import median
import os

from pencil_drawing import pencil_drawing

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# %%
if __name__ == "__main__":

    pencil_pattern_path = "./img/pencil1.jpg"
    image_path = "./input/3--17.jpg"
    filename = os.path.split(image_path)[-1]
    filename = os.path.splitext(filename)[0]

    image = imread(fname=image_path)
    image = median(image=image, behavior="ndimage")

    # colorful sketch
    sketch_color = pencil_drawing(image, pencil_pattern_path, is_gray=False)

    # gray sketch
    sketch_gray = pencil_drawing(image, pencil_pattern_path)


    fig = plt.figure(figsize=(12, 8), facecolor="white")

    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(image)
    ax.set_title("原始图像", color="black")
    ax.axis("off")

    ax = fig.add_subplot(2, 2, 3)
    ax.imshow(sketch_gray, cmap=plt.cm.gray)
    ax.set_title("素描", color="black")
    ax.axis("off")

    ax = fig.add_subplot(2, 2, 4)
    ax.imshow(sketch_color)
    ax.set_title("彩铅素描", color="black")
    ax.axis("off")

    plt.savefig("./output/{}.png".format(filename), transparent=True)
    plt.show()


# %%

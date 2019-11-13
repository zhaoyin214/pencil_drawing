#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   pencil_drawing.py
@time    :   2019/11/01 17:13:34
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   Combining Sketch and Tone for Pencil Drawing Production
"""

__author__ = "XiaoY"


# %%
from skimage.io import imread
from skimage.color import rgb2yuv, rgb2gray, yuv2rgb
from skimage.transform import rotate, resize
from skimage import img_as_float

import numpy as np
from scipy.signal import convolve2d
from scipy.sparse import spdiags, linalg


DIM_X = 1
DIM_Y = 0
DIM_KERNEL = 2

EPSILON = 1e-5
LAMBDA = 0.2

# %%
class PencilDrawing(object):
    """
    Combining Sketch and Tone for Pencil Drawing Production
    """

    def __init__(self,
                 stroke_kernel_size=8, stroke_num_kernels=8, stroke_width=3,
                 tone_weights=[52, 37, 11], tone_num_bins=256,
                 tone_laplace_sigma=9 / 255,
                 tone_uniform_bounds=[105 / 255., 225 / 255],
                 tone_gaussian_mu=90 / 255, tone_gaussian_sigma=11,
                 pencil_patter_path="./img/pencil0.jpg"):

        self._pencil_pattern = rgb2gray(imread(fname=pencil_patter_path))

        # stroke
        self._stroke_kernel_size = stroke_kernel_size // 2
        self._stroke_num_kernels = stroke_num_kernels
        self._stroke_width = stroke_width // 2

        # tone
        self._tone_hist(
            tone_weights, tone_num_bins,
            tone_laplace_sigma, tone_uniform_bounds,
            tone_gaussian_mu, tone_gaussian_sigma
        )


    def _stroke(self, gray_scale):
        """
        stroke structure
        """
        height, width = gray_scale.shape

        # gradients
        grad_x = np.diff(gray_scale, axis=DIM_X)
        grad_x = np.hstack([grad_x, np.zeros(shape=(height, 1))])
        grad_y = np.diff(gray_scale, axis=DIM_Y)
        grad_y = np.vstack([grad_y, np.zeros(shape=(1, width))])
        edge = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # convolutional kernel with the horizontal direction
        conv_kernel_0 = np.zeros(
            shape=(2 * self._stroke_kernel_size + 1,
                   2 * self._stroke_kernel_size + 1)
        )
        conv_kernel_0[self._stroke_kernel_size, :] = 1

        # classification
        response_kernel = np.zeros(
            shape=(height, width, self._stroke_num_kernels)
        )
        for idx_kernel in range(self._stroke_num_kernels):
            direction = idx_kernel * 180 / self._stroke_num_kernels
            conv_kernel = rotate(image=conv_kernel_0, angle=direction)
            response_kernel[:, :, idx_kernel] = convolve2d(
                in1=edge, in2=conv_kernel, mode="same"
            )
        edge_kernel_indicator = np.argmax(a=response_kernel, axis=DIM_KERNEL)

        # stroke
        edge_kernel = np.empty_like(prototype=response_kernel)
        for idx_kernel in range(self._stroke_num_kernels):
            edge_kernel[:, :, idx_kernel] = \
                edge * (edge_kernel_indicator == idx_kernel)

        stroke_width = np.min([self._stroke_kernel_size, self._stroke_width])
        for pixel in range(1, stroke_width + 1):
            conv_kernel_0[self._stroke_kernel_size - pixel] = 1
            conv_kernel_0[self._stroke_kernel_size + pixel] = 1

        strokes = np.zeros(shape=(height, width))
        for idx_kernel in range(self._stroke_num_kernels):
            direction = idx_kernel * 180 / self._stroke_num_kernels
            conv_kernel = rotate(image=conv_kernel_0, angle=direction)
            strokes += convolve2d(
                in1=edge_kernel[:, :, idx_kernel], in2=conv_kernel, mode="same"
            )

        # min-max normalization (rescaling)
        strokes -= np.min(strokes)
        strokes /= np.max(strokes) + EPSILON

        return 1 - strokes

    def _tone_hist(self, tone_weights, tone_num_bins,
                   tone_laplace_sigma, tone_uniform_bounds,
                   tone_gaussian_mu, tone_gaussian_sigma):
        """
        tone histogram pattern of a skecth
        """

        self._tone_bins = np.linspace(start=0, stop=1, num=tone_num_bins)
        # bright layer, laplacian distribution
        prob_bright = 1 / tone_laplace_sigma * \
            np.exp((self._tone_bins - 1) / tone_laplace_sigma)
        # mild layer, uniform distribution
        prob_mild = (self._tone_bins >= tone_uniform_bounds[0]) & \
            (self._tone_bins <= tone_uniform_bounds[1])
        prob_mild = prob_mild / (tone_uniform_bounds[1] - tone_uniform_bounds[0])
        # dark layer, gaussian distribution
        prob_dark = 1 / np.sqrt(2 * np.pi) / tone_gaussian_sigma * \
            np.exp(
                -1 * (self._tone_bins - tone_gaussian_mu) ** 2 / \
                (2 * tone_gaussian_sigma ** 2)
            )

        self._tone_hist_pattern = tone_weights[0] * prob_bright + \
            tone_weights[1] * prob_mild + tone_weights[2] * prob_dark
        self._tone_hist_pattern /= np.sum(self._tone_hist_pattern)

    def _tone_hist_map(self, gray_scale):
        """
        tone map (histogram matching)
        """

        _, src_unique_indices, src_counts = np.unique(
            gray_scale.ravel(), return_counts=True, return_inverse=True
        )
        src_quantiles = np.cumsum(src_counts) / gray_scale.size
        tmpl_quantiles = np.cumsum(self._tone_hist_pattern)
        matched = np.interp(src_quantiles, tmpl_quantiles, self._tone_bins)
        matched = matched[src_unique_indices].reshape(gray_scale.shape)

        return matched

    def _texture_rendering(self, gray_scale):

        height, width = gray_scale.shape
        size = gray_scale.size

        # log(J), tone map
        tone_matched = self._tone_hist_map(gray_scale) + EPSILON

        log_tone_matched = np.log(tone_matched.ravel())

        # log(H), pencil texture
        pencil_pattern = resize(
            image=self._pencil_pattern, output_shape=gray_scale.shape
        ) + EPSILON
        log_pencil_pattern = np.log(pencil_pattern.ravel())

        # dx, dy
        vec_ones = np.ones(shape=(size, ))
        diff_x = spdiags(
            data=[-1 * vec_ones, vec_ones], diags=[0, width], m=size, n=size
        )
        diff_y = spdiags(
            data=[-1 * vec_ones, vec_ones], diags=[0, 1], m=size, n=size
        )

        # Ax = b
        # A = log(H) * log(H) + LAMBDA * (dx * dx.T + dy * dy.T)
        # b = log(H) * log(J)
        # multiplication in sparse - matrix multiplication
        A = LAMBDA * (diff_x * diff_x.T + diff_y * diff_y.T) + spdiags(
            data=log_pencil_pattern ** 2, diags=0, m=size, n=size
        )
        b = log_pencil_pattern * log_tone_matched
        beta, info = linalg.cg(A=A, b=b, tol=1e-6)

        assert info == 0, "cg works failed!!!"

        texture = pencil_pattern ** beta.reshape((height, width))

        return texture

    def _get_gray_scale(self, image, is_gray=True):

        if image.ndim == 3:
            yuv = rgb2yuv(image)
            gray_scale = yuv[:, :, 0]
        else:
            gray_scale = img_as_float(image)

        if is_gray:
            return gray_scale
        else:
            return gray_scale, yuv

    def _drawing_gray(self, gray_scale):

        # generate strokes
        strokes = self._stroke(gray_scale)
        # generate texture
        texture = self._texture_rendering(gray_scale)

        sketch = strokes * texture

        return sketch

    def drawing_gray(self, image):

        gray_scale = self._get_gray_scale(image)

        return self._drawing_gray(gray_scale)

    def drawing_color(self, image):

        gray_scale, yuv = self._get_gray_scale(image, is_gray=False)
        yuv[:, :, 0] = self._drawing_gray(gray_scale)

        return yuv2rgb(yuv)

    @property
    def pencil_pattern(self):
        return self._pencil_pattern

    @pencil_pattern.setter
    def pencil_pattern(self, pencil_path):
        self._pencil_pattern = imread(pencil_path)


# %%
def pencil_drawing(
    image, pencil_pattern_path, is_gray=True
):

    pencil_drawing = PencilDrawing(pencil_patter_path=pencil_pattern_path)

    if is_gray:
        sketch = pencil_drawing.drawing_gray(image)
    else:
        sketch = pencil_drawing.drawing_color(image)

    return sketch

# %%
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from skimage.filters import median

    image = imread(fname="./input/15--298.jpg")
    image = median(image=image, behavior="ndimage")
    pencil_drawing = PencilDrawing()
    gray_scale = pencil_drawing._get_gray_scale(image)
    strokes = pencil_drawing._stroke(gray_scale)
    gray_scale_matched = pencil_drawing._tone_hist_map(gray_scale)
    texture = pencil_drawing._texture_rendering(gray_scale)
    sketch = pencil_drawing.drawing_gray(image)
    sketch_color = pencil_drawing.drawing_color(image)

    fig = plt.figure(figsize=(15, 10))

    ax = fig.add_subplot(3, 1, 1)
    ax.imshow(image)
    ax.set_title("input")
    ax.axis("off")

    ax = fig.add_subplot(3, 3, 4)
    ax.imshow(gray_scale, cmap="gray")
    ax.set_title("gray scale")
    ax.axis("off")


    ax = fig.add_subplot(3, 3, 5)
    ax.imshow(strokes, cmap="gray")
    ax.set_title("strokes")
    ax.axis("off")

    ax = fig.add_subplot(3, 3, 6)
    ax.imshow(gray_scale_matched, cmap="gray")
    ax.set_title("histogram matching")
    ax.axis("off")

    ax = fig.add_subplot(3, 3, 7)
    ax.imshow(texture, cmap="gray")
    ax.set_title("texture")
    ax.axis("off")

    ax = fig.add_subplot(3, 3, 8)
    ax.imshow(sketch, cmap="gray")
    ax.set_title("sketch")
    ax.axis("off")

    ax = fig.add_subplot(3, 3, 9)
    ax.imshow(sketch_color)
    ax.set_title("sketch color")
    ax.axis("off")

    plt.show()

# %%

#-*- coding:utf-8-*-
'''
montage:镜头组接,组合画
'''
import numpy as np


def slice_montage(montage, img_h, img_w, n_imgs):
    """Slice a montage image into n_img h x w images.

    Performs the opposite of the montage function.  Takes a montage image and
    slices it back into a N x H x W x C image.

    Parameters
    ----------
    montage : np.ndarray
        Montage image to slice.
    img_h : int
        Height of sliced image
    img_w : int
        Width of sliced image
    n_imgs : int
        Number of images to slice

    Returns
    -------
    sliced : np.ndarray
        Sliced images as 4d array.
    """
    sliced_ds = []
    for i in range(int(np.sqrt(n_imgs))):
        for j in range(int(np.sqrt(n_imgs))):
            sliced_ds.append(montage[
                1 + i + i * img_h:1 + i + (i + 1) * img_h,
                1 + j + j * img_w:1 + j + (j + 1) * img_w])
    return np.array(sliced_ds)
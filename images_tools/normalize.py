#-*- coding:utf-8-*-
import numpy as np
from images_tools import colorspace

def normalize(a, s=0.1):
    '''Normalize the image range for visualization'''
    return np.uint8(np.clip(
        (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5,
        0, 1) * 255)



def imnormalize(img, mean, std, to_rgb=True):
    img = img.astype(np.float32)
    if to_rgb:
        img = colorspace.bgr2rgb(img)
    return (img - mean) / std


def imdenormalize(img, mean, std, to_bgr=True):
    img = (img * std) + mean
    if to_bgr:
        img = colorspace.rgb2bgr(img)
    return img

#-*-coding:utf-8-*-
import numpy as np
from skimage.measure import label, regionprops
import cv2

def processImage(imgs):
    """
        process images before feeding to CNNs
        imgs: N x 1 x W x H
    """
    imgs = imgs.astype(np.float32)
    for i, img in enumerate(imgs):
        mean = img.mean()
        std = img.std()
        imgs[i] = (img - mean) / std
    return imgs

def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel

def color_image(image, num_classes=20):
    import matplotlib as mpl
    import matplotlib.cm
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))


def preproc(image):
    
    gray = np.mean(image, axis=2)
    key = np.median(gray[gray.shape[0]//2:,:])
    contrast = 2.0
    center = 127
    x = np.arange(256)
    lookup_table = 2*center/(1+np.exp(-2*contrast*(x-key)/center))
    factor = lookup_table[gray.astype(np.uint8)]/(gray+0.1)
    factor = np.expand_dims(factor, axis=2)
    red, green, blue = np.split(image, 3, axis=2)
    image_out = np.concatenate((factor*red,factor*green,factor*blue), axis=2)

    return image_out


def bwproc(image):
    if len(image.shape) != 2:
        print('bwproc only support 2D inputs.')
        return image

    image_label = label(image, connectivity=2)
    props = regionprops(image_label)
    num = len(props)
    max_area = 0
    max_idx = 0
    for i in range(num):
        if props[i].area > max_area:
            max_area = props[i].area
            max_idx = props[i].label

    if max_area > 0:
        image = (image_label == max_idx) * 100       
        mask = np.zeros((image.shape[0]+2,image.shape[1]+2),np.uint8)
        mask_fill = 1
        flags = 4|(mask_fill<<8)|cv2.FLOODFILL_FIXED_RANGE
        image = image.astype(np.uint8)
        cv2.floodFill(image, mask, (0,0), 1, 20, 20, flags)
        image = (image != 1)        

    return image
#-*- coding:utf-8-*-
from PIL import Image
import numpy as np
import scipy.misc

def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]
    return img

def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)


def imread_indexed(filename):
  """ Load image given filename."""

  im = Image.open(filename)

  annotation = np.atleast_3d(im)[...,0]
  return annotation,np.array(im.getpalette()).reshape((-1,3))

def imwrite_indexed(filename,array,color_palette):
  """ Save indexed png."""

  if np.atleast_3d(array).shape[2] != 1:
    raise Exception("Saving indexed PNGs requires 2D array.")

  im = Image.fromarray(array)
  im.putpalette(color_palette.ravel())
  im.save(filename, format='PNG')
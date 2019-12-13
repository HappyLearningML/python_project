#-*-coding:utf-8-*-
import os
import sys 
import numpy as np
from numpy import *
import cv2 
from matplotlib import pyplot
import operator as op
import collections
import shutil

def flatten(x):
    b = str(x)
    b = b.replace('[', '') 
    b = b.replace(']', '') 
    a = list(eval(b))
    return a


def pHash(imagefile):
    """get image pHash value"""
    image = cv2.imread(imagefile, 0)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)

    #create the two-diminese list
    h, w = image.shape[:2]
    vis0 = np.zeros((h,w), np.float32)
    vis0[:h, :w] = image

    #dct
    vis1 = cv2.dct(cv2.dct(vis0))
    vis1 = vis1[0:8, 0:8]

    img_list = flatten(vis1.tolist())

    avg = np.sum(img_list) * 1./len(img_list)
    avg_list = ['0' if i < avg else '1' for i in img_list]

    return ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, 8 * 8, 4)])

def hammingDist(s1,s2):
    assert len(s1) == len(s2)
    return np.sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])

def test(imagefile1, imagefile2, thres):
	hash1 = pHash(imagefile1)
	hash2 = pHash(imagefile2)

	out_score = 1 - hammingDist(hash1, hash2) * 1. / (8*8/4)

	if out_score > thres:
		print("%s and %s is similarity"%(imagefile1, imagefile2))


#-*-coding:utf-8 -*-
'''
这里需要调用BBox.py中的检测框类
'''
import h5py
import cv2
import numpy as np
from images_tools.processImage import processImage

def generate_hdf5(data, output='shit.h5'):
    lines = []
    dst = 'tf_test/'
    imgs = []
    labels = []
    for (imgPath, bbx, landmarks) in data:
        im = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        imgName = imgPath.split('/')[-1][:-4]
        
        bbx_sc = bbx.bbxScale(im.shape, scale=1.1)
        #print bbx_sc.x, bbx_sc.y, bbx_sc.w, bbx_sc.h
        im_sc = im[bbx_sc.y:bbx_sc.y+bbx_sc.h, bbx_sc.x:bbx_sc.x+bbx_sc.w]
        im_sc = cv2.resize(im_sc, (39, 39))
        imgs.append(im_sc.reshape(39, 39, 1))
        name = dst+imgName+'sc.jpg'
        lm_sc = bbx_sc.normalizeLmToBbx(landmarks)
        labels.append(lm_sc.reshape(10))
        lines.append(name + ' ' + ' '.join(map(str, lm_sc.flatten())) + '\n')
    imgs, labels = np.asarray(imgs), np.asarray(labels)
    imgs = processImage(imgs)
    with h5py.File(output, 'w') as h5:
        h5['data'] = imgs.astype(np.float32)
        h5['landmark'] = labels.astype(np.float32)
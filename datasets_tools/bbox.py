#-*-coding:utf-8 -*-
'''
这是一个检测框的类。主要作用是将检测框的一些操作
'''
import numpy as np

class BBox(object):

    def __init__(self, bbx):
        self.x = bbx[0]
        self.y = bbx[2]
        self.w = bbx[1] - bbx[0]
        self.h = bbx[3] - bbx[2]


    def bbxScale(self, im_size, scale=1.3):
        '''
        主要是判断scale之后是否大于1
        '''
        # We need scale greater than 1 #
        assert(scale > 1)
        x = np.around(max(1, self.x - (scale * self.w - self.w) / 2.0))
        y = np.around(max(1, self.y - (scale * self.h - self.h) / 2.0))
        w = np.around(min(scale * self.w, im_size[1] - x))
        h = np.around(min(scale * self.h, im_size[0] - y))
        return BBox([x, x+w, y, y+h])

    def bbxShift(self, im_size, shift=0.03):
        '''
        检测框转换
        '''
        direction = np.random.randn(2)
        x = np.around(max(1, self.x - self.w * shift * direction[0]))
        y = np.around(max(1, self.y - self.h * shift * direction[1]))
        w = min(self.w, im_size[1] - x)
        h = min(self.h, im_size[0] - y)
        return BBox([x, x+w, y, y+h])

    def normalizeLmToBbx(self, landmarks):
        result = []
        # print self.x, self.y, self.w, self.h
        # print landmarks
        lmks = landmarks.copy()
        for lm in lmks:
            lm[0] = (lm[0] - self.x) / self.w
            lm[1] = (lm[1] - self.y) / self.h
            result.append(lm)
        result = np.asarray(result)
        
        return result
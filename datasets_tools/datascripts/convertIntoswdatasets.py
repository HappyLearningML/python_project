#-*-coding:utf-8-*-
import os
import cv2
import sys

from sys_tools import cv2_tools

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append('../')

if __name__ == "__main__":
    
    imgdir = "E:/sw_seg_datasets/bdd100k"
    dstdir = "E:\\sw_seg_datasets"
    '''
    for roots, parents, files in os.walk(imgdir):
        for imagefile in files:
            srcimagename = os.path.join(roots, imagefile)
            predirs = srcimagename.split('\\')[-3] + '\\' + srcimagename.split('\\')[-2]
            dstimagename = os.path.join(dstdir, predirs + '\\bdd_' + imagefile)
            if not os.path.exists(dstimagename):
                print(dstimagename)
                cv2_tools.rename(srcimagename, dstimagename)


    image_citydir = "E:\\sw_seg_datasets\\leftImg8bit"
    for roots, parents, files in os.walk(image_citydir):
        for imagefile in files:
            srcimagename = os.path.join(roots, imagefile)
            predirs = srcimagename.split('\\')[-3] + '\\cityscapes_'
            dstimagename = os.path.join(dstdir, "images\\"+predirs + imagefile.split('.')[0] + '.jpg')
            print(dstimagename)
            cv2_tools.rename(srcimagename, dstimagename)

    '''
    color_citydir = "E:\\sw_seg_datasets\\gtFine"
    for roots, parents, files in os.walk(color_citydir):
        for imagefile in files:
            srcimagename = os.path.join(roots, imagefile)
            if(srcimagename.endswith("png")):
                preimages = imagefile.split('_')[-1]
                predirs = srcimagename.split('\\')[-3] + '\\cityscapes_'
                if (preimages == "color.png"):
                    dstimagename = os.path.join(dstdir, "color_labels\\" + predirs + imagefile)
                    print(dstimagename)
                    cv2_tools.rename(srcimagename, dstimagename)
                elif (preimages == "instanceIds.png"):
                    dstimagename = os.path.join(dstdir, "labels\\" + predirs + imagefile)
                    print(dstimagename)
                    cv2_tools.rename(srcimagename, dstimagename)
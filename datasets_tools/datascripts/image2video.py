#-*-coding:utf-8-*-
import cv2
import os

sourcedirs = "C:\\Users\\swei\\Desktop\\result\\results"
dstdirs = "C:\\Users\\swei\\Desktop\\result"

def image2video(imagedir, videoname, frameWidth, frameHeight, fps=30):
    video_writer = cv2.VideoWriter(videoname, fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps=fps, frameSize=(width, height))
    
    for roots, parents, files in os.walk(imagedir):
        for image in files:
            imagename = os.path.join(roots, image)
            imat = cv2.imread(imagename)
            cv2.waitKey(10)
            video_writer.write(imat)
    video_writer.release()


for name in os.listdir(sourcedirs):
    sourcename = os.path.join(sourcedirs, name)
    videoname = os.path.join(dstdirs, name + '.avi')
    height, width, _ = cv2.imread(sourcename + "\\" + os.listdir(sourcename)[0]).shape
    image2video(sourcename, videoname, width, height)
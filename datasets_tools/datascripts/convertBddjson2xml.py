#-*-coding:utf-8-*-
import os
import cv2
import sys
import json

sys.path.append(os.getcwd())

from sys_tools import cv2_tools
bdd_dirs = "E:\\sw_datasets\\adas_detection\\bdd100k"
imagedst_dirs = "E:\\sw_datasets\\adas_detection\\images"
labels_dirs = "E:\\sw_datasets\\adas_detection\\labels"

label_lists = ['car', 'bus', 'person', 'bike', 'truck', 'motor', 'train', 'rider', 'traffic sign', 'traffic light']

def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def json2txt(jsonname, dstlabelsname):
    inputfile = open(jsonname, "rb")
    outputfile = open(dstlabelsname, "w")
    fileJson = json.load(inputfile)
    field = fileJson["name"]
    if "bdd100k_" + field == (dstlabelsname.split("\\")[-1]).split(".")[0]:
        frame = fileJson["frames"]
        objects = frame[0]['objects']
        for i in range(len(objects)):
            label_flag = objects[i]['category']
            if label_flag in label_lists:
                boxes = objects[i]['box2d']
                boxes_top_x = str(boxes["x1"])
                boxes_top_y = str(boxes["y1"])
                boxes_bot_x = str(boxes["x2"])
                boxes_bot_y = str(boxes["y2"])
                line_content = label_flag + "," + boxes_top_x + "," + boxes_top_y + "," + boxes_bot_x + "," + boxes_bot_y
                outputfile.write(line_content)
                outputfile.write("\n")



        
        #print(boxes)


    #fileJson.close()
    outputfile.close()




for roots, parents, images in os.walk(bdd_dirs):
    for imagefile in images:
        srcimagename = os.path.join(roots, imagefile)
        dirprefixs = srcimagename.split("\\")[-2]
        if srcimagename.endswith("jpg"):
            dstdirs = os.path.join(imagedst_dirs, dirprefixs)
            make_dirs(dstdirs)
            dstimagename = os.path.join(dstdirs, "bdd100k_" + imagefile)
            print(dstimagename)
            cv2_tools.rename(srcimagename, dstimagename)
        elif srcimagename.endswith("json"):
            dstdirs = os.path.join(labels_dirs, dirprefixs)
            make_dirs(dstdirs)
            imageprefixs = imagefile.split(".")[0]
            dstlabelsname = os.path.join(dstdirs, "bdd100k_" + imageprefixs + ".txt")
            print(dstlabelsname)
            json2txt(srcimagename, dstlabelsname)
            
            






        
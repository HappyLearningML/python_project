#-*-coding:utf-8-*-
import os
import cv2
import json

cur_pwd = os.getcwd()
print(cur_pwd)

label_lists = ['car', 'bus', 'person', 'bike', 'truck', 'motor', 'train', 'rider', 'traffic sign', 'traffic light']

imagedirs = "E:\\sw_datasets\\adas_detection\\leftImg8bit_trainvaltest\\leftImg8bit"
#jsondirs = "E:\\sw_datasets\\adas_detection\\gtCoarse\\gtCoarse"
jsondirs = "E:\\sw_datasets\\adas_detection\\gtFine"

dstimagedirs = "E:\\sw_datasets\\adas_detection\\images"
dstjsondirs = "E:\\sw_datasets\\adas_detection\\labels"

def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def png2jpeg(srcimage, dstimage):
    mat = cv2.imread(srcimage)
    cv2.imwrite(dstimage, mat)
    cv2.waitKey(10)

def json2txt(jsonname, dstlabelsname, imagename):
    inputfile = open(jsonname, "rb")
    img = cv2.imread(imagename)
    #outputfile = open(dstlabelsname, "w")
    fileJson = json.load(inputfile)
    objects = fileJson['objects']
    for i in range(len(objects)):
        label_flag = objects[i]['label']
        if label_flag not in label_lists:
            '''
            boxes = objects[i]['box2d']
            boxes_top_x = str(boxes["x1"])
            boxes_top_y = str(boxes["y1"])
            boxes_bot_x = str(boxes["x2"])
            boxes_bot_y = str(boxes["y2"])
            '''
            polygon = objects[i]['polygon']
            for j in range(len(polygon)):
                cv2.circle(img, (polygon[j][0], polygon[j][1]), 1, (0, 0, 255), 4)

    cv2.imshow("show", img)
    cv2.waitKey(1000)



for roots, parents, images in os.walk(imagedirs):
    for image in images:
        imagename = os.path.join(roots, image)
        dirprefix = roots.split('leftImg8bit\\')[-1]
        prefixs = dirprefix.split('\\')[0]
        dstimagename = os.path.join(dstimagedirs, prefixs + "\\cityscapes_" + image.split('.')[0] + '.jpg')
        #png2jpeg(imagename, dstimagename)
        #print(dstimagename)
        if prefixs == "test":
            continue
        else:
            jsondir = os.path.join(jsondirs, dirprefix)
            #namepostfix = image.split('leftImg8bit')[0] + 'gtCoarse_polygons.json'
            namepostfix = image.split('leftImg8bit')[0] + 'gtFine_polygons.json'
            jsonname = os.path.join(jsondir, namepostfix)
            dstjsonname = os.path.join(dstjsondirs, prefixs + "\\cityscapes_" + image.split('.')[0] + '.txt')
            json2txt(jsonname, dstjsonname, imagename)


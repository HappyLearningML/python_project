#-*-coding:utf-8-*-
import cv2
import os


sourcedirs = "C:\\Users\\swei\\Desktop\\result\\source"
dstdirs = "C:\\Users\\swei\\Desktop\\result\\results"

label_list = []

def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

for roots, parents, files in os.walk(sourcedirs):
    for imagefile in files:
        absimagefile = os.path.join(roots, imagefile)
        if absimagefile.endswith('.jpg'):
            absdst = os.path.join(dstdirs, roots.split("\\")[-1])
            make_dirs(absdst)
            dstname = os.path.join(absdst, imagefile)
            print(absimagefile, dstname, parents, roots)
            txtimagefile = absimagefile.split('.')[0] + ".txt"
            txt = open(txtimagefile, 'r')
            mat = cv2.imread(absimagefile)


            for line in txt.readlines():
                lineList = line.strip().split(',')
                label = lineList[0]
                xmin = int(lineList[1])
                ymin = int(lineList[2])
                xmax = int(lineList[3])
                ymax = int(lineList[4])
            
                if label not in label_list:
                    label_list.append(label)

                if label == "car":
                    cv2.rectangle(mat, (xmin, ymin), (xmax, ymax), (0,0,255), 2) 
                    mat = cv2.putText(mat, label, (xmin - 2, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0,0,255),2)
                elif label == "trafficSign":
                    cv2.rectangle(mat, (xmin, ymin), (xmax, ymax), (0,255,255), 2)
                    mat = cv2.putText(mat, label, (xmin - 2, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0,255,255),2)
                else:
                    cv2.rectangle(mat, (xmin, ymin), (xmax, ymax), (255,255,0), 2)
                    mat = cv2.putText(mat, label, (xmin - 2, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255,255,0),2)
                
            cv2.imwrite(dstname, mat)
            cv2.imshow("test", mat)
            cv2.waitKey(10)
            txt.close()

print(label_list)

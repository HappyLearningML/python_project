#-*-coding:utf-8-*-
import os
import cv2
import sys
import xml
import  xml.dom.minidom

dstdirs = "C:\\Users\\swei\\Desktop\\result\\results"
#orgdirs = "C:\\Users\\swei\\Desktop\\result\\org"
orgdirs = "C:\\Users\\swei\\Desktop\\result\\20190306170310_1"

def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
'''
def xmlToTxt(xmlfile, txtfile):
    txt = open(txtfile, "w")
    dom = xml.dom.minidom.parse(xmlfile)
    root = dom.documentElement
    outputs=root.getElementsByTagName('outputs')[0]
    son = outputs.getElementsByTagName('object')[0]
    sonSon = son.getElementsByTagName('item')
    
    for i in range(len(sonSon)):
        labelname = (sonSon[i].getElementsByTagName('name')[0]).childNodes[0].data
        bndboxvalues = sonSon[i].getElementsByTagName('bndbox')[0]
        box_xmin = (bndboxvalues.getElementsByTagName('xmin')[0]).childNodes[0].data
        box_ymin = (bndboxvalues.getElementsByTagName('ymin')[0]).childNodes[0].data
        box_xmax = (bndboxvalues.getElementsByTagName('xmax')[0]).childNodes[0].data
        box_ymax = (bndboxvalues.getElementsByTagName('ymax')[0]).childNodes[0].data
        
        linecont = labelname + "," + box_xmin + "," + box_ymin + "," + box_xmax + "," + box_ymax
        txt.write(linecont)
        txt.write("\n")
    txt.close()


for roots, parents, images in os.walk(orgdirs):
    for imagefile in images:
        absimagefile = os.path.join(roots, imagefile)
        if absimagefile.endswith("jpg"):
            #imageMat = cv2.imread(absimagefile)
            #cv2.imshow("test", imageMat)
            #cv2.waitKey(10)

            dirsprefixs = absimagefile.split("\\")[-2]
            absdirs = os.path.join(orgdirs, dirsprefixs)
            make_dirs(absdirs)
            fileprefixs = imagefile.split(".")[0]
            xmlfile = fileprefixs + ".xml"
            absxmlfile = os.path.join(absdirs, "outputs\\" + xmlfile)

            txtfile = fileprefixs + ".txt"
            txtdirs = os.path.join(dstdirs, dirsprefixs)
            make_dirs(txtdirs)
            abstxtfile = os.path.join(txtdirs, txtfile)
            xmlToTxt(absxmlfile, abstxtfile)

            print(abstxtfile)
'''

def xmlToTxt_1(xmlfile, txtfile):
    txt = open(txtfile, "w")
    dom = xml.dom.minidom.parse(xmlfile)
    root = dom.documentElement
    outputs=root.getElementsByTagName('object')

    for i in range(len(outputs)):
        sonoutput = outputs[i].getElementsByTagName('name')[0]
        labelname = sonoutput.childNodes[0].data
        bndboxvalues = outputs[i].getElementsByTagName('bndbox')[0]
        box_xmin = (bndboxvalues.getElementsByTagName('xmin')[0]).childNodes[0].data
        box_ymin = (bndboxvalues.getElementsByTagName('ymin')[0]).childNodes[0].data
        box_xmax = (bndboxvalues.getElementsByTagName('xmax')[0]).childNodes[0].data
        box_ymax = (bndboxvalues.getElementsByTagName('ymax')[0]).childNodes[0].data
        
        linecont = labelname + "," + box_xmin + "," + box_ymin + "," + box_xmax + "," + box_ymax
        txt.write(linecont)
        txt.write("\n")

    txt.close()

for roots, parents, images in os.walk(orgdirs):
    for imagefile in images:
        absimagefile = os.path.join(roots, imagefile)
        
        if absimagefile.endswith("jpg"):
            dirsprefixs = absimagefile.split("\\")[-2]
            #absdirs = os.path.join(orgdirs, dirsprefixs)
            fileprefixs = imagefile.split(".")[0]
            xmlfile = fileprefixs + ".xml"
            absxmlfile = os.path.join(orgdirs, "outputs\\" + xmlfile)

            txtfile = fileprefixs + ".txt"
            txtdirs = os.path.join(dstdirs, dirsprefixs)
            make_dirs(txtdirs)
            abstxtfile = os.path.join(txtdirs, txtfile)
            xmlToTxt_1(absxmlfile, abstxtfile)
            print(abstxtfile)


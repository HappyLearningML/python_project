#-*-coding:utf-8-*-
'''
该文档主要计算两张图片的相似度问题
'''
import os
import cv2
import numpy as np

def getHash(image):
	average = np.mean(image)
	hash = []
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if image[i, j] > average:
				hash.append(1)
			else:
				hash.append(0)
	return hash

def Hamming_distance(hash1, hash2):
	num = 0
	for index in range(len(hash1)):
		if hash1[index] != hash2[index]:
			num += 1
	return num

def classify_aHash(image1, image2):
	image1 = imageConvert(image1, (256, 256), True)
	image2 = imageConvert(image2, (256, 256), True)
	hash1 = getHash(image1)
	hash2 = getHash(image2)
	return Hamming_distance(hash1, hash2)

def classify_pHash(image1, image2):
	image1 = imageConvert(image1, (256, 256), True)
	image2 = imageConvert(image2, (256, 256), True)

	dct1 = cv2.dct(np.float32(image1))
	dct2 = cv2.dct(np.float32(image2))

	dct1_roi = dct1[0:8, 0:8]
	dct2_roi = dct2[0:8, 0:8]
	hash1 = getHash(dct1_roi)
	hash2 = getHash(dct2_roi)
	return Hamming_distance(hash1, hash2)



def imageInfo(image):
	mat = cv2.imread(image)
	shape = mat.shape
	return mat, shape[0], shape[1], shape[2]

def imageConvert(image, size=(1280,720), greyscale=False):
	image = cv2.resize(image, size, cv2.INTER_AREA)
	if greyscale:
		if image.shape[2] == 3:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		elif image.shape[2] == 4:
			image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

	return image



def classify_gray_hist(image1, image2, size=(256,256)):
	'''
	calculate grayhist to compare the difference 
	'''
	image1 = imageConvert(image1, size, True)
	image2 = imageConvert(image2, size, True)
	hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
	hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])

	# jisuan chong he du
	degree = 0
	for i in range(len(hist1)):
		if hist1[i] != hist2[i]:
			degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
		else:
			degree = degree + 1
	degree = degree / len(hist1)
	return degree

def calculate(image1, image2):
	hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
	hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])

	degree = 0
	for i in range(len(hist1)):
		if hist1[i] != hist2[i]:
			degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
		else:
			degree = degree + 1
	degree = degree / len(hist1)
	return degree

def classify_hist_with_split(image1, image2, size = (256, 256)):
	image1 = cv2.resize(image1, size)
	image2 = cv2.resize(image2, size)
	sub_image1 = cv2.split(image1)
	sub_image2 = cv2.split(image2)
	sub_data = 0
	for im1, im2 in zip(sub_image1, sub_image2):
		sub_data += calculate(im1, im2)
	sub_data = sub_data / 3
	return sub_data


def selectBase(baseImage, fileImage):
    '''
    baseImage：基准图片
    fileImage：参与比较的图片
    返回值：
        distance：计算图片与基准图片之间的hist距离
    '''

    baseIm = cv2.imread(baseImage)
    fileIm, file_h, file_w, _ = imageInfo(fileImage)
    distance = classify_hist_with_split(baseIm, fileIm, size=(file_w, file_h))
    return distance


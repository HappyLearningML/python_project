#-*-coding:utf-8-*-
'''
# HOG特征提取算法的实现过程

HOG特征提取方法就是将一个image（你要检测的目标或者扫描窗口）：

1）灰度化（将图像看做一个x,y,z（灰度）的三维图像）；

2）采用Gamma校正法对输入图像进行颜色空间的标准化（归一化）；目的是调节图像的对比度，降低图像局部的阴影和光照变化所造成的影响，同时可以抑制噪音的干扰；

3）计算图像每个像素的梯度（包括大小和方向）；主要是为了捕获轮廓信息，同时进一步弱化光照的干扰。

4）将图像划分成小cells（例如6*6像素/cell）；

5）统计每个cell的梯度直方图（不同梯度的个数），即可形成每个cell的descriptor；

6）将每几个cell组成一个block（例如3*3个cell/block），一个block内所有cell的特征descriptor串联起来便得到该block的HOG特征descriptor。

7）将图像image内的所有block的HOG特征descriptor串联起来就可以得到该image（你要检测的目标）的HOG特征descriptor了。这个就是最终的可供分类使用的特征向量了。

## python实现过程
### 步骤（1）灰度化
#### img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

cv2.IMREAD_COLOR：读入一副彩色图片；

cv2.IMREAD_GRAYSCALE：以灰度模式读入图片；

cv2.IMREAD_UNCHANGED：读入一幅图片，并包括其alpha通道;
'''

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


class Hog_descriptor():
    """
    HOG特征提取算法的实现过程
    """
    def __init__(self, img, cell_size=16, bin_size=8):
        self.img = img
        self.img = np.sqrt(img / float(np.max(img)))
        self.img = self.img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = int(360 / self.bin_size)
        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        assert type(self.angle_unit) == int, "bin_size should be divisible by 360"

    def extract(self):
        height, width = self.img.shape
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        print("the pixel of the input img：%d * %d\n每个cells大小：%d" % (height, width, self.cell_size))
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size))
        print("the no. of cell_gradient_vector:", cell_gradient_vector.shape)

        for i in range(cell_gradient_vector.shape[0]):
            # shape[0]取行数
            for j in range(cell_gradient_vector.shape[1]):
                # shape[1]取列数
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
        return hog_vector, hog_image

    def global_gradient(self):
        """
        计算图像每个像素的梯度（包括大小和方向）
        :return:大小和方向
        """
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=3)
        # cv2.imshow('sobel_x', gradient_values_x)
        # cv2.waitKey(0)

        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=3)
        # cv2.imshow('sobel_y', gradient_values_y)
        # cv2.waitKey(0)

        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        # cv2.imshow('sobel_x+y', gradient_magnitude)
        # cv2.waitKey(0)

        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        # 最后一个参数为True表示结果用角度制表示，否则用弧度制
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        if idx == self.bin_size:
            return idx - 1, (idx) % self.bin_size, mod
        return idx, (idx + 1) % self.bin_size, mod

    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image

#========Test========
# 读取图像和显示图像操作
img = cv2.imread('data/person2.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.imshow('picture1', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# #dv2.destroyWindow('picture1')


hog = Hog_descriptor(img, cell_size=8, bin_size=8)
vector, image = hog.extract()
print(image.shape)
print(image)

plt.imshow(image, cmap=plt.cm.gray)

plt.show()

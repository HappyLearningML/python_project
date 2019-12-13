#-*- coding:utf-8-*-
'''
Canny边缘检测算法由5个步骤组成：

1)降噪:边缘检测结果对图像噪声高度敏感, 
    消除图像噪声的一种方法是使用高斯模糊平滑图像。
    为此，图像卷积技术应用高斯核(3x3, 5x5, 7x7等)。
    核大小取决于预期的模糊效果。基本上，核越小，模糊就越不明显。在我们的例子中，我们将使用一个5×5的高斯核函数
2)梯度计算:梯度计算步骤通过使用边缘检测算子计算图像的梯度来检测边缘强度和方向
    边缘对应于像素强度的变化。要检测它，最简单的方法是应用filters，在两个方向上突出这种强度变化:水平(x)和垂直(y).当平滑图像时，计算导数Ix和Iy
3)非最大抑制:理想情况下，最终的图像应该有细边。因此，我们必须执行非最大抑制以使边缘变细
    算法遍历梯度强度矩阵上的所有点，并找到边缘方向上具有最大值的像素
4)双阈值：
    双阈值步骤旨在识别3种像素：强，弱和不相关：

    强像素是指像素的强度如此之高，以至于我们确信它们有助于最终的边缘。
    弱像素是具有不足以被视为强的强度值的像素，但是还不足以被认为与边缘检测不相关。
    其他像素被认为与边缘无关。
5)滞后边缘跟踪：根据阈值结果，当且仅当被处理像素周围至少有一个像素为强像素时，滞后由弱像素转换为强像素构成
'''
import numpy as np
from scipy import ndimage

#降噪
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

    return g
#梯度计算
def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (G, theta)
#非最大抑制
def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)

    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q = 255
                r = 255

                #angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]

                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                
                #angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                
                #angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass
        
    return Z

#双阈值
def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)

#滞后边缘跟踪
def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if(img[i, j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) 
                        or (img[i+1, j] == strong) 
                        or (img[i+1, j+1] == strong) 
                        or (img[i, j-1] == strong) 
                        or (img[i, j+1] == strong) 
                        or (img[i-1, j-1] == strong) 
                        or (img[i-1, j] == strong) 
                        or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass

    return img
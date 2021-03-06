import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt

def rgb2hsv(img):
    r,g,b = cv2.split(img)      # 分離RGB
    r, g, b = r/255.0, g/255.0, b/255.0
    # HSV 分別承接經過公式轉換後的H,S,V值    # 高(直行)img.shape[0]     # 寬(橫列)img.shape[1]   # \ 多行語句
    H, S, V = np.zeros((img.shape[0], img.shape[1]), np.float32), \
        np.zeros((img.shape[0], img.shape[1]), np.float32),  np.zeros((img.shape[0], img.shape[1]), np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            max_ = max((b[i, j], g[i, j], r[i, j]))
            min_ = min((b[i, j], g[i, j], r[i, j]))
            # H
            if max_ == min_:           # 當 max = min 時，相位角為'0度'
                H[i, j] = 0
            elif max_ == r[i, j]:      # 當 r  為 max 時，相位角為'60度'
                if g[i, j] >= b[i, j]:      # 當 g >= b 時，+0
                    H[i, j] = (60 * ((g[i, j]) - b[i, j]) / (max_ - min_))
                else:                       # 當 g < b 時，+360
                    H[i, j] = (60 * ((g[i, j]) - b[i, j]) / (max_ - min_))+360
            elif max_ == g[i, j]:       # 當 g 為 max 時，相位角為'60度' + 120
                H[i, j] = 60 * ((b[i, j]) - r[i, j]) / (max_ - min_) + 120
            elif max_ == b[i, j]:       # 當 b 為 max 時，相位角為'60度' + 240
                H[i, j] = 60 * ((r[i, j]) - g[i, j]) / (max_ - min_)+ 240
            # hsv 中的值是 0-360, 0-1, 0-1，但是在 openCV 中 hsv 的值是 0-180, 0-255, 0-255
            H[i,j] =int( H[i,j] / 2)
            # S
            if max_ == 0:
                S[i, j] = 0
            else:
                S[i, j] =int( (max_ - min_)/max_*255)
            # V
            V[i, j] =int( max_*255)
    # 合併輸出
    # HSV = cv2.merge([H,S,V])
    # HSV=np.array(HSV,dtype='uint8')
    # return HSV
    # 不合併輸出
    return H, S, V


img = cv2.imread('maya.jpg',-1)  # 讀檔
h, s, v  = rgb2hsv(img)
hsv = cv2.merge([h, s, v])      # 合併HSV
img_rgb2hsv=np.array(hsv,dtype='uint8')

img_COLOR_RGB2HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

cv2.imshow('img',img_rgb2hsv)
cv2.imshow('img1',img_COLOR_RGB2HSV)
cv2.waitKey()





# python实现RGB转换HSV : https://blog.csdn.net/weixin_43360384/article/details/84871521
# RGB转到HSV和HSL公式 : https://blog.csdn.net/Sunshine_in_Moon/article/details/45131285
# 由RGB到HSV的转换详解 : https://zhuanlan.zhihu.com/p/105886300
# UE製作PBR材質攻略Part 1 - 色彩知識
# https://www.itread01.com/content/1570932365.html

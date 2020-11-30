import os
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('j.png',0)
img_op = cv2.imread('opening.png',0)
img_cl = cv2.imread('closing.png',0)
kernel = np.ones((3,3),np.uint8)

erosion = cv2.erode(img,kernel,iterations = 1)  # 侵蝕 , iterations 迭代(數字越大侵蝕、擴張越多)
dilation = cv2.dilate(img,kernel,iterations = 1)  # 擴張

opening = cv2.morphologyEx(img_op, cv2.MORPH_OPEN, kernel)  # 斷開，先侵蝕 後擴張，去除圖像中的小亮點
closing = cv2.morphologyEx(img_cl, cv2.MORPH_CLOSE, kernel)  # 閉合，先擴張 後侵蝕，去除圖像中的小黑點

cv2.imshow('img_00',erosion)
cv2.imshow('img_01',dilation)
# cv2.imshow('img_02_o',img_op)
# cv2.imshow('img_03-o',img_cl)
# cv2.imshow('img_02',opening)
# cv2.imshow('img_03',closing)
cv2.waitKey()



# Morphological Transformations : https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html

import os
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

def read_img(img_path):     # 讀檔, input = 影像路徑, output = 圖像,檔名,副檔名
    img_filepath = os.path.splitext(img_path)[0]    # 拆分路徑 & 副檔名，0 為路徑
    img_fileextension = os.path.splitext(img_path)[1]  # 1 為副檔名
    img_filename = os.path.basename(img_filepath)     # 取出檔名不含副檔名
    # img = cv2.imread(img_path, -1)  # 讀檔(中文路徑會爆掉)
    img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1) # 讀檔(含有中文路徑)
    return img, img_filename, img_fileextension

# 讀檔
img, file_name, file_extension = read_img('E:/Program_File/PYTHON/數位影像處理作業/HW_3/Test_Img/03.jpg')

# 黑白
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 定義內核大小並應用高斯平滑處理
kernel_size = 3
blur_img_gray = cv2.GaussianBlur(img_gray,(kernel_size, kernel_size), 0)

# 定義Canny的參數並應用
edges = cv2.Canny(blur_img_gray, threshold1=50, threshold2=200, apertureSize=3)

edges_HoughLinesP = edges.copy()
# 定義Hough轉換參數
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)
for line in [l[0] for l in lines]:  # 畫線
    leftx, boty, rightx, topy = line
    cv2.line(edges_HoughLinesP, (leftx, boty), (rightx,topy), (255, 255, 0), 2)

cv2.imshow('Canny', edges)
# cv2.imwrite('./Test_Img/' + file_name + '_canny_function.jpg',edges)
cv2.imshow('Result', edges_HoughLinesP)
cv2.waitKey(0)
cv2.destroyAllWindows()



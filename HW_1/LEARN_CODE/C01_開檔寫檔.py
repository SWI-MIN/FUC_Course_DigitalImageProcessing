# from numpy import *
import numpy as np
import cv2

# 1. from numpy import *  ,  2. import numpy as np  
# 差別 : 1. 可以直接使用numpy所有函數
#        2. 使用numpy函數前需要加 np.
#        EX :　a = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#              a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])



# -------------彩色轉黑白照片__方法__1-------------

# img_RGB2GRAY_00_00 = cv2.imread('P01_flower.jpg',cv2.IMREAD_UNCHANGED)  #  讀檔   !!!!!!!!!!!!!!!!!!!!!!
# ----------------------
# cv::ImreadModes {
#   cv::IMREAD_UNCHANGED = -1,   讀取圖片中所有的 channels，包含透明度的 channel。
#   cv::IMREAD_GRAYSCALE = 0,    以灰階的格式來讀取圖片。
#   cv::IMREAD_COLOR = 1,        預設值，讀取 RGB 的彩色圖片，忽略透明度的 channel
#   cv::IMREAD_ANYDEPTH = 2,
#   cv::IMREAD_ANYCOLOR = 4,
#   cv::IMREAD_LOAD_GDAL = 8,
#   cv::IMREAD_REDUCED_GRAYSCALE_2 = 16,
#   cv::IMREAD_REDUCED_COLOR_2 = 17,
#   cv::IMREAD_REDUCED_GRAYSCALE_4 = 32,
#   cv::IMREAD_REDUCED_COLOR_4 = 33,
#   cv::IMREAD_REDUCED_GRAYSCALE_8 = 64,
#   cv::IMREAD_REDUCED_COLOR_8 = 65,
#   cv::IMREAD_IGNORE_ORIENTATION = 128
# }

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# type(img_RGB2GRAY_00_00)  # 看資料型態
# print(img_RGB2GRAY_00_00.shape)   # 圖片的高度 x 寬度 x channel（RGB = 3，灰階 = 1）

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# img_RGB2GRAY_00_01 = cv2.cvtColor(img_RGB2GRAY_00_00, cv2.COLOR_RGB2GRAY)   # RGB2GRAY    !!!!!!!!!!!!!!!!

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# cv2.namedWindow('P01_flower.jpg', cv2.WINDOW_KEEPRATIO)   
# ----------------------
# 用法：cv2.namedWindow('窗口标题',默认参数)
# 默认参数： cv2.WINDOW_AUTOSIZE+cv2.WINDOW_KEEPRATIO+cv2.WINDOW_GUI_EXPANDED)
# 参数：
# （1）cv2.WINDOW_NORMAL ： 窗口大小可改变。
# （2）cv2.WINDOW_AUTOSIZE ： 窗口大小不可改变。
# （3）cv2.WINDOW_FREERATIO ： 自适应比例。
# （4）cv2.WINDOW_KEEPRATIO ： 保持比例。

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# cv2.resizeWindow('P01_flower.jpg', (1102, 826))  # 調整圖片大小

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# cv2.imshow('P01_flower.jpg',img_RGB2GRAY_00_01)  # show圖片
# cv2.waitKey()                                 # 暫停一下，不然圖片一閃即逝
# cv2.destroyAllWindows()  # 按下任意鍵則關閉所有視窗

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# cv2.imwrite('P01_flower_gray.jpg', img_RGB2GRAY_00_01)  # 寫檔   !!!!!!!!!!!!!!!!!!!!!!!!!!
# ----------------------
# 寫入不同圖檔格式
# cv2.imwrite('P01_flower_gray.png', img_RGB2GRAY_00_01)
# cv2.imwrite('P01_flower_gray.tiff', img_RGB2GRAY_00_01)




# -------------彩色轉黑白照片__方法__2-------------

img_RGB2GRAY_01_00 = cv2.imread('P01_flower.jpg',cv2.IMREAD_GRAYSCALE)  #  直接讀灰階檔
cv2.imwrite('P01_flower_gray.jpg', img_RGB2GRAY_01_00)  # 寫檔